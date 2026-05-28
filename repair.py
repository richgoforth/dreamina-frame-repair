#!/usr/bin/env python3
"""
video-repair: Fix frame drops and duplicate frames in AI-generated video.

Interpolation backends (in priority order):
  1. RIFE v4 via PyTorch MPS  — SOTA; auto-downloaded on first use via --setup-rife
  2. DIS optical flow (OpenCV) — excellent for smooth AI content; works immediately
  3. Simple pixel blend        — last resort, no external deps

Usage:
    python repair.py input.mp4 --repair repairs.json
    python repair.py input.mp4 --insert-after 13 --insert-after 29 --remove 36
    python repair.py input.mp4 --analyze
    python repair.py --setup-rife
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Repair:
    type: Literal["insert", "remove"]
    frame: int       # 1-indexed frame number in the ORIGINAL video
    note: str = ""


# ---------------------------------------------------------------------------
# Video metadata
# ---------------------------------------------------------------------------

def get_video_info(video_path: str | Path) -> dict:
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", "-select_streams", "v:0", str(video_path),
    ]
    data = json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)
    s = data["streams"][0]
    num, den = map(int, s["r_frame_rate"].split("/"))
    return {
        "fps": num / den,
        "width": s["width"],
        "height": s["height"],
        "nb_frames": int(s.get("nb_frames", 0)),
        "duration": float(s.get("duration", 0.0)),
    }


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def extract_frames(video_path: str | Path, out_dir: str | Path) -> list[Path]:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(video_path),
            "-q:v", "1", "-pix_fmt", "rgb24",
            str(Path(out_dir) / "frame_%06d.png"),
        ],
        check=True, capture_output=True,
    )
    return sorted(Path(out_dir).glob("frame_*.png"))


# ---------------------------------------------------------------------------
# Interpolation backends
# ---------------------------------------------------------------------------

def _load_rife():
    """Try to import the RIFE model from ~/.video-repair/rife/. Returns model or None."""
    rife_dir = Path.home() / ".video-repair" / "rife"
    if not rife_dir.exists():
        return None, None
    try:
        if str(rife_dir) not in sys.path:
            sys.path.insert(0, str(rife_dir))
        from model.RIFE_HDv3 import Model  # type: ignore
        import torch
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        model = Model()
        model.load_model(str(rife_dir / "train_log"), -1)
        model.eval()
        model.device()
        return model, device
    except Exception as e:
        print(f"  [rife] load failed: {e}", file=sys.stderr)
        return None, None


_rife_cache: tuple | None = None  # (model, device) or (None, None) if unavailable


def _get_rife():
    global _rife_cache
    if _rife_cache is None:
        _rife_cache = _load_rife()
    return _rife_cache


def _interpolate_rife(img_a: np.ndarray, img_b: np.ndarray) -> np.ndarray | None:
    """RIFE v4 interpolation using PyTorch MPS. Returns interpolated frame or None on failure."""
    model, device = _get_rife()
    if model is None:
        return None
    try:
        import torch
        def to_tensor(img: np.ndarray):
            return torch.from_numpy(img).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)

        I0 = to_tensor(img_a)
        I1 = to_tensor(img_b)
        with torch.no_grad():
            mid = model.inference(I0, I1)
        out = mid[0].permute(1, 2, 0).cpu().numpy()
        return (out * 255.0).clip(0, 255).astype(np.uint8)
    except Exception as e:
        print(f"  [rife] inference failed: {e}", file=sys.stderr)
        return None


def _interpolate_dis(img_a: np.ndarray, img_b: np.ndarray) -> np.ndarray:
    """
    Motion-compensated interpolation using OpenCV DIS optical flow.

    For each direction (A→B, B→A), we warp the source frame halfway toward
    the target using the estimated dense flow, then blend the two warped
    versions. A forward-backward consistency mask down-weights occluded
    regions so they fall back to a simple blend.
    """
    h, w = img_a.shape[:2]

    a_gray = cv2.cvtColor(img_a, cv2.COLOR_RGB2GRAY)
    b_gray = cv2.cvtColor(img_b, cv2.COLOR_RGB2GRAY)

    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    flow_ab = dis.calc(a_gray, b_gray, None)   # flow that maps A pixels → B
    flow_ba = dis.calc(b_gray, a_gray, None)   # flow that maps B pixels → A

    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)

    def warp(img: np.ndarray, flow: np.ndarray, t: float = 0.5) -> np.ndarray:
        map_x = (gx + flow[:, :, 0] * t).astype(np.float32)
        map_y = (gy + flow[:, :, 1] * t).astype(np.float32)
        return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    warped_a = warp(img_a, flow_ab, t=0.5)   # A pushed halfway toward B
    warped_b = warp(img_b, flow_ba, t=0.5)   # B pushed halfway toward A

    # Forward-backward consistency check: if warp(B by fwd flow) ≠ B position,
    # the pixel is likely occluded. Use residual magnitude as confidence penalty.
    fwd_in_b = warp(flow_ab, flow_ab, t=1.0)   # flow_ab remapped through itself
    consistency = np.sqrt(
        (flow_ab[:, :, 0] + flow_ba[:, :, 0]) ** 2 +
        (flow_ab[:, :, 1] + flow_ba[:, :, 1]) ** 2
    )
    # Confidence: 1 where flows are perfectly consistent, 0 where very inconsistent
    max_err = np.percentile(consistency, 95) + 1e-6
    conf = np.clip(1.0 - consistency / max_err, 0.0, 1.0)[:, :, np.newaxis]

    blended = (
        conf * 0.5 * warped_a.astype(np.float32) +
        conf * 0.5 * warped_b.astype(np.float32) +
        (1.0 - conf) * 0.5 * img_a.astype(np.float32) +
        (1.0 - conf) * 0.5 * img_b.astype(np.float32)
    )
    return blended.clip(0, 255).astype(np.uint8)


def _interpolate_blend(img_a: np.ndarray, img_b: np.ndarray) -> np.ndarray:
    """Simple 50% pixel blend. Fallback when nothing else is available."""
    return ((img_a.astype(np.float32) + img_b.astype(np.float32)) / 2).clip(0, 255).astype(np.uint8)


def interpolate_frame(path_a: Path, path_b: Path, out_path: Path) -> str:
    """Generate an interpolated frame between path_a and path_b. Returns method name."""
    img_a = np.array(Image.open(path_a).convert("RGB"))
    img_b = np.array(Image.open(path_b).convert("RGB"))

    # Try RIFE first (SOTA)
    result = _interpolate_rife(img_a, img_b)
    if result is not None:
        Image.fromarray(result).save(str(out_path), "PNG")
        return "RIFE"

    # Fall back to DIS optical flow
    try:
        result = _interpolate_dis(img_a, img_b)
        Image.fromarray(result).save(str(out_path), "PNG")
        return "DIS"
    except Exception as e:
        print(f"  [dis] failed: {e}", file=sys.stderr)

    # Last resort: simple blend
    result = _interpolate_blend(img_a, img_b)
    Image.fromarray(result).save(str(out_path), "PNG")
    return "blend"


# ---------------------------------------------------------------------------
# Repair logic
# ---------------------------------------------------------------------------

def apply_repairs(
    frames: list[Path],
    repairs: list[Repair],
    interp_dir: Path,
) -> list[Path]:
    """
    Apply repairs to the frame list. All frame numbers are 1-indexed and refer
    to the ORIGINAL video's frame positions. Returns the repaired frame list.
    """
    interp_dir.mkdir(parents=True, exist_ok=True)
    sorted_repairs = sorted(repairs, key=lambda r: r.frame)
    result = list(frames)
    offset = 0          # Cumulative index shift from earlier insertions/removals
    interp_n = 0

    for rep in sorted_repairs:
        orig_idx = rep.frame - 1            # 0-indexed in original
        curr_idx = orig_idx + offset        # Current position in result

        if rep.type == "insert":
            if curr_idx >= len(result) - 1:
                print(f"  Warning: cannot insert after frame {rep.frame} (out of bounds)")
                continue
            interp_path = interp_dir / f"interp_{interp_n:04d}.png"
            method = interpolate_frame(result[curr_idx], result[curr_idx + 1], interp_path)
            result.insert(curr_idx + 1, interp_path)
            offset += 1
            interp_n += 1
            note = f"  — {rep.note}" if rep.note else ""
            print(f"  [insert] after orig #{rep.frame} [{method}]{note}")

        elif rep.type == "remove":
            if curr_idx >= len(result):
                print(f"  Warning: cannot remove frame {rep.frame} (out of bounds)")
                continue
            result.pop(curr_idx)
            offset -= 1
            note = f"  — {rep.note}" if rep.note else ""
            print(f"  [remove] orig #{rep.frame}{note}")

    return result


# ---------------------------------------------------------------------------
# Video assembly
# ---------------------------------------------------------------------------

def assemble_video(
    frame_list: list[Path],
    audio_source: str | Path,
    output_path: str | Path,
    fps: float,
) -> None:
    """
    Assemble frame list into output video, copying audio from the source.

    Uses a sequentially-numbered symlink directory as input so ffmpeg receives
    a clean constant-framerate image sequence — avoids concat-demuxer rounding
    issues that can introduce duplicate or dropped frames.
    """
    # Represent fps as an integer fraction string (e.g. "24/1")
    fps_num = round(fps * 1000)
    fps_den = 1000
    from math import gcd
    g = gcd(fps_num, fps_den)
    fps_str = f"{fps_num // g}/{fps_den // g}"

    with tempfile.TemporaryDirectory() as seq_dir:
        seq_path = Path(seq_dir)
        for i, src in enumerate(frame_list):
            dst = seq_path / f"{i + 1:08d}.png"
            try:
                dst.symlink_to(Path(src).resolve())
            except (OSError, NotImplementedError):
                shutil.copy2(str(src), str(dst))

        subprocess.run(
            [
                "ffmpeg", "-y",
                "-framerate", fps_str,
                "-i", str(seq_path / "%08d.png"),
                "-i", str(audio_source),
                "-map", "0:v", "-map", "1:a",
                "-c:v", "libx264", "-preset", "slow", "-crf", "18",
                "-pix_fmt", "yuv420p",
                "-r", fps_str,
                "-c:a", "copy",
                "-shortest",
                str(output_path),
            ],
            check=True, capture_output=True,
        )


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze(video_path: str | Path, thumb_width: int = 480) -> None:
    """
    Scan a video for likely frame drops and duplicates.

    Computes per-frame pixel diff at a reduced resolution.
    Flags transitions that are statistical outliers (z-score thresholds):
      - Very low diff  → probable duplicate (doubled frame)
      - Very high diff → probable skip (dropped frame)
    """
    info = get_video_info(video_path)
    print(f"\n{Path(video_path).name}")
    print(f"  {info['nb_frames']} frames @ {info['fps']:.4f} fps  {info['width']}x{info['height']}")

    with tempfile.TemporaryDirectory() as tmpdir:
        print("  Extracting frames (scaled)...")
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(video_path),
                "-vf", f"scale={thumb_width}:-1",
                "-q:v", "2",
                os.path.join(tmpdir, "f_%06d.png"),
            ],
            check=True, capture_output=True,
        )
        frames = sorted(Path(tmpdir).glob("f_*.png"))

        print("  Computing frame diffs...")
        prev = None
        diffs: list[float] = []
        for f in frames:
            arr = np.array(Image.open(f).convert("L"), dtype=np.float32)
            if prev is not None:
                diffs.append(float(np.mean(np.abs(arr - prev))))
            prev = arr

    diffs_arr = np.array(diffs)
    mean_d = float(np.mean(diffs_arr))
    std_d = float(np.std(diffs_arr))
    print(f"  mean diff = {mean_d:.3f}  std = {std_d:.3f}\n")

    issues_found = 0
    for i, d in enumerate(diffs):
        z = (d - mean_d) / (std_d + 1e-9)
        frame_before = i + 1   # 1-indexed
        frame_after  = i + 2
        if d < 0.5:
            print(f"  DUPLICATE  between #{frame_before} and #{frame_after}: diff={d:.4f}  z={z:.2f}")
            issues_found += 1
        elif z < -1.8:
            print(f"  likely dup between #{frame_before} and #{frame_after}: diff={d:.4f}  z={z:.2f}")
            issues_found += 1
        elif z > 3.5:
            print(f"  SKIP       between #{frame_before} and #{frame_after}: diff={d:.4f}  z={z:.2f}")
            issues_found += 1
        elif z > 2.0:
            print(f"  likely skip between #{frame_before} and #{frame_after}: diff={d:.4f}  z={z:.2f}")
            issues_found += 1

    if issues_found == 0:
        print("  No significant anomalies detected.")
    print(f"\n  {issues_found} issue(s) flagged.")


# ---------------------------------------------------------------------------
# RIFE setup
# ---------------------------------------------------------------------------

def setup_rife() -> None:
    """
    Download practical-RIFE model code and weights to ~/.video-repair/rife/.
    Requires git and internet access.
    """
    install_dir = Path.home() / ".video-repair" / "rife"
    install_dir.mkdir(parents=True, exist_ok=True)

    repo_dir = install_dir
    if not (repo_dir / "model").exists():
        print("Cloning practical-RIFE model code...")
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/hzwer/Practical-RIFE.git", str(repo_dir)],
            check=True,
        )
    else:
        print("practical-RIFE already cloned.")

    weights_dir = repo_dir / "train_log"
    if weights_dir.exists() and any(weights_dir.iterdir()):
        print("Model weights already present.")
    else:
        print("Downloading RIFE v4.6 weights (~50 MB)...")
        weights_dir.mkdir(exist_ok=True)
        import urllib.request
        base = "https://github.com/hzwer/Practical-RIFE/releases/download/model4.6/"
        for fname in ["flownet.pkl", "contextnet.pkl", "unet.pkl"]:
            url = base + fname
            dest = weights_dir / fname
            print(f"  {fname}...", end=" ", flush=True)
            urllib.request.urlretrieve(url, dest)
            print("done")

    print(f"\nRIFE installed to {install_dir}")
    print("Run repair.py normally — it will use RIFE automatically.")


# ---------------------------------------------------------------------------
# Repair spec loading
# ---------------------------------------------------------------------------

def load_json_spec(path: str | Path) -> list[Repair]:
    with open(path) as f:
        data = json.load(f)
    repairs = []
    for item in data.get("repairs", []):
        t = item["type"]
        note = item.get("note", "")
        if t == "insert":
            repairs.append(Repair("insert", int(item["after"]), note))
        elif t == "remove":
            repairs.append(Repair("remove", int(item["frame"]), note))
        else:
            print(f"  Warning: unknown repair type '{t}', skipping")
    return repairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repair frame drops and duplicates in AI-generated video.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # Repair using a JSON spec file
  python repair.py input.mp4 --repair repairs.json

  # Inline repairs
  python repair.py input.mp4 --insert-after 13 --insert-after 29 --remove 36

  # Analyze for issues (no changes made)
  python repair.py input.mp4 --analyze

  # Set up RIFE (SOTA interpolation, one-time download ~50MB)
  python repair.py --setup-rife

JSON spec format:
  {
    "repairs": [
      {"type": "insert", "after": 13, "note": "skip between 13 and 14"},
      {"type": "remove", "frame": 36, "note": "doubled frame"}
    ]
  }
        """,
    )
    parser.add_argument("video", nargs="?", help="Input video file")
    parser.add_argument("--output", "-o", help="Output path (default: <input>_repaired.mp4)")
    parser.add_argument("--repair", "-r", help="JSON repair spec file")
    parser.add_argument(
        "--insert-after", type=int, action="append", metavar="FRAME",
        help="Insert interpolated frame after this 1-indexed frame number (repeatable)",
    )
    parser.add_argument(
        "--remove", type=int, action="append", metavar="FRAME",
        help="Remove this 1-indexed frame number (repeatable)",
    )
    parser.add_argument(
        "--trim-to", type=int, metavar="FRAME",
        help="Trim output to this many frames (useful for frozen-tail artifacts)",
    )
    parser.add_argument("--analyze", action="store_true", help="Analyze video for frame issues")
    parser.add_argument("--setup-rife", action="store_true", help="Download and install RIFE weights")
    parser.add_argument("--fps", type=float, help="Override detected FPS")
    parser.add_argument(
        "--backend", choices=["auto", "rife", "dis", "blend"], default="auto",
        help="Force a specific interpolation backend (default: auto)",
    )
    args = parser.parse_args()

    if args.setup_rife:
        setup_rife()
        return

    if not args.video:
        parser.error("video argument is required unless using --setup-rife")

    video_path = Path(args.video).resolve()
    if not video_path.exists():
        sys.exit(f"Error: {video_path} not found")

    # Force backend if requested
    if args.backend == "dis":
        global _rife_cache
        _rife_cache = (None, None)   # Prevent RIFE from loading
    elif args.backend == "blend":
        _rife_cache = (None, None)

    if args.analyze:
        analyze(video_path)
        return

    # Collect repairs
    repairs: list[Repair] = []
    if args.repair:
        repairs.extend(load_json_spec(args.repair))
    for f in (args.insert_after or []):
        repairs.append(Repair("insert", f))
    for f in (args.remove or []):
        repairs.append(Repair("remove", f))

    if not repairs and args.trim_to is None:
        sys.exit("No repairs specified. Use --repair, --insert-after, --remove, or --trim-to.")

    # Output path
    if args.output:
        output_path = Path(args.output).resolve()
    else:
        output_path = video_path.with_name(video_path.stem + "_repaired" + video_path.suffix)

    info = get_video_info(video_path)
    fps = args.fps if args.fps else info["fps"]

    print(f"\nInput:  {video_path.name}")
    print(f"        {info['nb_frames']} frames @ {fps:.4f} fps  {info['width']}x{info['height']}")
    if repairs:
        print(f"Repairs: {len(repairs)}")
    if args.trim_to:
        print(f"Trim to: {args.trim_to} frames")

    # Check which interpolation backend is active
    if any(r.type == "insert" for r in repairs):
        model, _ = _get_rife()
        backend_name = "RIFE (MPS)" if model else "DIS optical flow"
        print(f"Backend: {backend_name}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        frames_dir = tmpdir / "frames"
        interp_dir = tmpdir / "interp"

        print("\nExtracting frames...")
        frames = extract_frames(video_path, frames_dir)
        print(f"  {len(frames)} frames extracted")

        if repairs:
            print("\nApplying repairs...")
            frames = apply_repairs(frames, repairs, interp_dir)
            print(f"  Result: {len(frames)} frames ({len(frames) / fps:.3f}s)")

        if args.trim_to and len(frames) > args.trim_to:
            removed = len(frames) - args.trim_to
            frames = frames[: args.trim_to]
            print(f"\nTrimmed {removed} frames → {len(frames)} frames ({len(frames) / fps:.3f}s)")

        print(f"\nAssembling → {output_path.name}")
        assemble_video(frames, video_path, output_path, fps)

    size_mb = output_path.stat().st_size / 1_048_576
    print(f"Done. {output_path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
