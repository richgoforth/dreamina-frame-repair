#!/usr/bin/env python3
"""
video-repair: Detect and fix frame drops / duplicates in AI-generated video.

Interpolation backends (priority order):
  1. RIFE via rife-ncnn-vulkan — SOTA; needs a GPU (Apple Silicon/Metal works).
                                 Set up once with --setup-rife. CPU-only machines
                                 fall through to DIS, which ghosts on fast motion.
  2. DIS optical flow (OpenCV) — motion-compensated warp; works without a GPU
  3. Simple pixel blend        — last resort fallback

Detection pipeline:
  - Optical flow magnitude (DIS) tracks actual per-pixel motion
  - SSIM measures perceptual similarity between frame pairs
  - Local velocity baseline (median in rolling window) normalises for scene dynamics
  - Temporal consistency check validates skip candidates against surrounding motion
  - Direction consistency check filters out scene-change false positives
  - Non-maximum suppression collapses clustered candidates to one event

Usage:
    python repair.py input.mp4 --detect                   # Analyse, print report, save spec
    python repair.py input.mp4 --detect --auto-repair     # Analyse + immediately repair
    python repair.py input.mp4 --repair repairs.json      # Repair from saved spec
    python repair.py input.mp4 --insert-after 13 --remove 36
    python repair.py --setup-rife                         # One-time RIFE weight download
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
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
    frame: int       # 1-indexed in the original video
    note: str = ""


@dataclass
class Detection:
    type: Literal["skip", "duplicate", "frozen_tail"]
    frame: int          # For skip: frame BEFORE the gap; for dup/tail: frame to remove / tail start
    confidence: float
    evidence: dict = field(default_factory=dict)

    def to_repair(self) -> Repair | None:
        note = (
            f"{self.type} detected "
            f"(conf={self.confidence:.2f}"
            + (f", flow_ratio={self.evidence.get('flow_ratio', ''):.2f}x" if self.type == "skip" else "")
            + (f", ssim={self.evidence.get('ssim', ''):.4f}" if self.type == "duplicate" else "")
            + ")"
        )
        if self.type == "skip":
            return Repair("insert", self.frame, note)
        if self.type == "duplicate":
            return Repair("remove", self.frame, note)
        return None   # frozen_tail is handled separately via --trim-to


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

    def _color(field: str, fallback: str) -> str:
        v = s.get(field, "")
        return v if v and v != "unknown" else fallback

    return {
        "fps": num / den,
        "width": s["width"],
        "height": s["height"],
        "nb_frames": int(s.get("nb_frames", 0)),
        "duration": float(s.get("duration", 0.0)),
        # Color metadata — preserved onto output to prevent color shift
        "color_primaries": _color("color_primaries", "bt709"),
        "color_trc":       _color("color_transfer",  "bt709"),
        "color_space":     _color("color_space",     "bt709"),
        "color_range":     _color("color_range",     "tv"),
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
# SSIM (no external deps — pure OpenCV/numpy)
# ---------------------------------------------------------------------------

def compute_ssim(a: np.ndarray, b: np.ndarray) -> float:
    """
    Mean SSIM between two images. Handles grayscale or RGB input.
    Range 0..1 where 1 = identical.
    """
    if a.ndim == 3:
        a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
        b = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    k = (11, 11)
    sigma = 1.5
    mu1 = cv2.GaussianBlur(a, k, sigma)
    mu2 = cv2.GaussianBlur(b, k, sigma)
    s1  = cv2.GaussianBlur(a * a, k, sigma) - mu1 ** 2
    s2  = cv2.GaussianBlur(b * b, k, sigma) - mu2 ** 2
    s12 = cv2.GaussianBlur(a * b, k, sigma) - mu1 * mu2
    num = (2 * mu1 * mu2 + C1) * (2 * s12 + C2)
    den = (mu1 ** 2 + mu2 ** 2 + C1) * (s1 + s2 + C2)
    return float(np.mean(num / (den + 1e-10)))


# ---------------------------------------------------------------------------
# Per-pair feature extraction
# ---------------------------------------------------------------------------

def _scale_image(img: np.ndarray, target_width: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w == target_width:
        return img
    new_h = int(h * target_width / w)
    return cv2.resize(img, (target_width, new_h), interpolation=cv2.INTER_AREA)


def compute_pair_features(frames: list[Path], analysis_width: int = 640) -> list[dict]:
    """
    Compute per consecutive-pair features at reduced resolution.

    Returns list of dicts (one per adjacent pair) containing:
      - pair       : (frame_before_1idx, frame_after_1idx)
      - flow_mag   : mean magnitude of DIS optical flow (pixels/frame)
      - flow_dx/dy : mean x and y displacement components
      - ssim       : structural similarity (0–1)
      - diff       : mean absolute pixel difference (0–255)
    """
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    features: list[dict] = []

    # Load and scale all frames once
    imgs: list[np.ndarray] = []
    for fp in frames:
        rgb = np.array(Image.open(fp).convert("RGB"))
        imgs.append(_scale_image(rgb, analysis_width))

    for i in range(len(imgs) - 1):
        a, b = imgs[i], imgs[i + 1]
        a_g = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
        b_g = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)

        flow = dis.calc(a_g, b_g, None)
        mag  = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)

        features.append({
            "pair":     (i + 1, i + 2),
            "flow_mag": float(np.mean(mag)),
            "flow_dx":  float(np.mean(flow[:, :, 0])),
            "flow_dy":  float(np.mean(flow[:, :, 1])),
            "ssim":     compute_ssim(a_g, b_g),
            "diff":     float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32)))),
        })

    return features


# ---------------------------------------------------------------------------
# Local velocity baseline
# ---------------------------------------------------------------------------

def local_median_baseline(values: np.ndarray, window: int = 9) -> np.ndarray:
    """
    For each index i, compute median of values in [i-window//2, i+window//2]
    EXCLUDING i itself. Uses median for robustness against the anomalies we're
    trying to detect (outliers inflate a mean baseline, hiding further events).
    """
    n = len(values)
    half = window // 2
    baseline = np.empty(n)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        neighbours = np.concatenate([values[lo:i], values[i + 1:hi]])
        baseline[i] = np.median(neighbours) if len(neighbours) else np.median(values)
    return baseline


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def _direction_consistency(
    candidate_dx: float, candidate_dy: float,
    surrounding: list[dict],
) -> float:
    """
    Returns 0–1: how well the candidate's motion direction aligns with the
    surrounding frame pairs. Cosine similarity mapped to [0, 1].
    Protects against confusing scene-change spikes (direction changes sharply)
    with genuine skips (direction is consistent, just magnitude is double).
    """
    if not surrounding:
        return 0.5
    sdx = float(np.mean([f["flow_dx"] for f in surrounding]))
    sdy = float(np.mean([f["flow_dy"] for f in surrounding]))
    smag = math.sqrt(sdx ** 2 + sdy ** 2) + 1e-9
    cmag = math.sqrt(candidate_dx ** 2 + candidate_dy ** 2) + 1e-9
    cosine = (candidate_dx * sdx + candidate_dy * sdy) / (cmag * smag)
    return (cosine + 1.0) / 2.0   # −1..1 → 0..1


def _temporal_fit(
    candidate_flow: float,
    before_flows: np.ndarray,
    after_flows: np.ndarray,
) -> float:
    """
    How well does the candidate flow fit the "double-step" hypothesis?

    A skipped frame means the playback jumped two frames' worth of motion in
    one step. If the surrounding velocity is V, the skip should show flow ≈ 2V.
    We score proximity to that prediction and penalise if the surrounding
    context is too sparse or noisy.
    """
    if len(before_flows) == 0 or len(after_flows) == 0:
        return 0.4   # Insufficient context — mild penalty

    surr = np.concatenate([before_flows, after_flows])
    local_v = float(np.median(surr))

    if local_v < 0.05:
        # Near-static scene: any spike is suspicious, but we can't confirm direction
        return 0.55

    predicted_skip = 2.0 * local_v
    relative_error = abs(candidate_flow - predicted_skip) / (predicted_skip + 1e-9)
    # Perfect match → 1.0 ; error > 100% → 0.0
    return float(np.clip(1.0 - relative_error, 0.0, 1.0))


def _detect_frozen_tail(ssims: np.ndarray, threshold: float = 0.995, min_length: int = 3) -> int | None:
    """
    Return the 0-indexed pair index where a frozen tail begins, or None.

    A frozen tail is a run of min_length+ consecutive pairs all with SSIM ≥ threshold
    that reaches the end of the video — a known Dreamina generation artifact.
    """
    n = len(ssims)
    # Walk backwards to find where the freeze ends and content stops
    run = 0
    for i in range(n - 1, -1, -1):
        if ssims[i] >= threshold:
            run += 1
        else:
            break
    if run >= min_length:
        return n - run   # Pair index of first frozen pair (0-indexed)
    return None


def _non_max_suppression(detections: list[Detection], radius: int = 3) -> list[Detection]:
    """
    Within any cluster of same-type detections closer than `radius` frames,
    retain only the one with the highest confidence. Prevents double-counting
    of a single event that looks noisy across adjacent pairs.
    """
    if not detections:
        return []
    result: list[Detection] = []
    for det in sorted(detections, key=lambda d: -d.confidence):
        too_close = any(
            d.type == det.type and abs(d.frame - det.frame) <= radius
            for d in result
        )
        if not too_close:
            result.append(det)
    return sorted(result, key=lambda d: d.frame)


# ---------------------------------------------------------------------------
# Main detection function
# ---------------------------------------------------------------------------

def detect_issues(
    video_path: str | Path,
    *,
    ssim_dup_threshold: float = 0.97,
    skip_flow_ratio: float = 1.75,
    baseline_window: int = 9,
    min_confidence: float = 0.70,
    analysis_width: int = 640,
    verbose: bool = False,
) -> list[Detection]:
    """
    Detect frame drops (skips) and duplicate frames in a video.

    Parameters
    ----------
    ssim_dup_threshold  : SSIM ≥ this → duplicate candidate (default 0.97)
    skip_flow_ratio     : flow / local_baseline ≥ this → skip candidate (default 1.75)
    baseline_window     : rolling window for local velocity baseline (default 9 pairs)
    min_confidence      : discard detections below this score (default 0.70)
    analysis_width      : pixel width to scale frames down to for analysis (default 640)
    verbose             : print per-pair feature table if True

    Returns
    -------
    List of Detection objects sorted by frame number.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        frames_dir = Path(tmpdir) / "frames"
        print("  Extracting frames for analysis...")
        frames = extract_frames(video_path, frames_dir)
        print(f"  Computing optical flow + SSIM for {len(frames)-1} frame pairs...")
        features = compute_pair_features(frames, analysis_width=analysis_width)

    n = len(features)
    if n == 0:
        return []

    flow_mags = np.array([f["flow_mag"] for f in features])
    ssims     = np.array([f["ssim"]     for f in features])

    baseline = local_median_baseline(flow_mags, window=baseline_window)

    if verbose:
        print(f"\n  {'Pair':>10}  {'flow':>7}  {'base':>7}  {'ratio':>6}  {'ssim':>7}")
        for i, f in enumerate(features):
            ratio = flow_mags[i] / (baseline[i] + 1e-9)
            print(
                f"  {f['pair'][0]:>4}->{f['pair'][1]:<4}  "
                f"{flow_mags[i]:>7.3f}  "
                f"{baseline[i]:>7.3f}  "
                f"{ratio:>6.2f}x  "
                f"{ssims[i]:>7.4f}"
            )

    detections: list[Detection] = []

    # ------------------------------------------------------------------
    # 1. Duplicate detection
    # ------------------------------------------------------------------
    for i in range(n):
        ssim_val  = ssims[i]
        flow_mag  = flow_mags[i]
        lv        = baseline[i]

        if ssim_val < ssim_dup_threshold:
            continue

        # SSIM-based confidence: 0 at threshold, 1 at perfect similarity
        ssim_conf = min(1.0, (ssim_val - ssim_dup_threshold) / (1.0 - ssim_dup_threshold))

        # Flow-based modifier: reward near-zero motion, penalise non-trivial motion
        if lv > 0.1:
            flow_ratio = flow_mag / lv
            flow_mod = max(0.0, 1.0 - 0.4 * min(flow_ratio, 1.0))
        else:
            flow_mod = 1.0

        confidence = ssim_conf * 0.80 + flow_mod * 0.20

        if confidence < min_confidence:
            continue

        # Remove the second frame of the pair (the duplicate)
        detections.append(Detection(
            type="duplicate",
            frame=i + 2,   # 1-indexed
            confidence=round(confidence, 3),
            evidence={
                "ssim":          round(float(ssim_val), 4),
                "flow_mag":      round(float(flow_mag), 3),
                "local_baseline": round(float(lv), 3),
            },
        ))

    # ------------------------------------------------------------------
    # 2. Skip (dropped-frame) detection
    # ------------------------------------------------------------------
    for i in range(n):
        flow_mag  = flow_mags[i]
        lv        = baseline[i]

        if lv < 0.05:
            continue

        ratio = flow_mag / lv

        if ratio < skip_flow_ratio:
            continue

        # Guard: is this actually a scene change rather than a skip?
        # Scene changes tend to have SSIM << 0.5 and flow magnitude >> 4× baseline.
        if ssims[i] < 0.35 and ratio > 5.0:
            continue   # Likely scene cut

        # Gather surrounding pairs for context
        ctx_radius = 4
        before_idx = [j for j in range(max(0, i - ctx_radius), i)]
        after_idx  = [j for j in range(i + 1, min(n, i + ctx_radius + 1))]
        before_flows = flow_mags[before_idx]
        after_flows  = flow_mags[after_idx]

        # Score 1: temporal fit to "double-step" hypothesis
        t_fit = _temporal_fit(float(flow_mag), before_flows, after_flows)

        # Score 2: direction consistency (skips preserve direction, scene changes don't)
        surrounding = [features[j] for j in before_idx + after_idx]
        dir_conf = _direction_consistency(features[i]["flow_dx"], features[i]["flow_dy"], surrounding)

        # Score 3: magnitude ratio score (how far above threshold)
        ratio_score = min(1.0, (ratio - skip_flow_ratio) / skip_flow_ratio)

        confidence = 0.55 * t_fit + 0.30 * dir_conf + 0.15 * ratio_score

        if confidence < min_confidence:
            continue

        detections.append(Detection(
            type="skip",
            frame=i + 1,   # Insert AFTER this 1-indexed frame
            confidence=round(confidence, 3),
            evidence={
                "flow_mag":       round(float(flow_mag), 3),
                "local_baseline": round(float(lv), 3),
                "flow_ratio":     round(float(ratio), 2),
                "temporal_fit":   round(float(t_fit), 3),
                "dir_consistency": round(float(dir_conf), 3),
            },
        ))

    # ------------------------------------------------------------------
    # 3. Frozen-tail detection
    # ------------------------------------------------------------------
    tail_start_pair = _detect_frozen_tail(ssims, threshold=0.995, min_length=3)
    if tail_start_pair is not None:
        # Convert pair index to first frozen frame (1-indexed)
        first_frozen_frame = tail_start_pair + 2   # pair i links frames i+1 and i+2
        frozen_count = n - tail_start_pair
        detections.append(Detection(
            type="frozen_tail",
            frame=first_frozen_frame,
            confidence=1.0,
            evidence={"frozen_frames": frozen_count},
        ))

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------
    # Separate frozen_tail before NMS (it's not a skip/dup, handle independently)
    frozen = [d for d in detections if d.type == "frozen_tail"]
    events = [d for d in detections if d.type != "frozen_tail"]

    events = _non_max_suppression(events, radius=3)

    return events + frozen


# ---------------------------------------------------------------------------
# Detection report
# ---------------------------------------------------------------------------

def print_detection_report(
    detections: list[Detection],
    video_path: str | Path,
    info: dict,
    fps: float,
) -> None:
    skips  = [d for d in detections if d.type == "skip"]
    dups   = [d for d in detections if d.type == "duplicate"]
    tails  = [d for d in detections if d.type == "frozen_tail"]

    print(f"\n{'─'*60}")
    print(f"DETECTION REPORT: {Path(video_path).name}")
    print(f"  {info['nb_frames']} frames @ {fps:.4g} fps  "
          f"{info['width']}×{info['height']}")
    print(f"{'─'*60}")

    if skips:
        print(f"\n  FRAME SKIPS ({len(skips)}):")
        for d in skips:
            ts = d.frame / fps
            ev = d.evidence
            print(
                f"    After #{d.frame:>4}  ({ts:>6.3f}s)  "
                f"conf={d.confidence:.2f}  "
                f"flow={ev.get('flow_mag', 0):.2f}px "
                f"({ev.get('flow_ratio', 0):.1f}× baseline)  "
                f"tfit={ev.get('temporal_fit', 0):.2f}  "
                f"dir={ev.get('dir_consistency', 0):.2f}"
            )

    if dups:
        print(f"\n  DUPLICATE FRAMES ({len(dups)}):")
        for d in dups:
            ts = d.frame / fps
            ev = d.evidence
            print(
                f"    Frame #{d.frame:>4}  ({ts:>6.3f}s)  "
                f"conf={d.confidence:.2f}  "
                f"ssim={ev.get('ssim', 0):.4f}  "
                f"flow={ev.get('flow_mag', 0):.2f}px"
            )

    if tails:
        t = tails[0]
        ts = t.frame / fps
        print(f"\n  FROZEN TAIL:")
        print(
            f"    Starts at frame #{t.frame} ({ts:.3f}s)  "
            f"{t.evidence.get('frozen_frames', '?')} frozen frames  — recommend --trim-to {t.frame - 1}"
        )

    total = len(skips) + len(dups)
    if total == 0 and not tails:
        print("\n  No frame issues detected.")
    print(f"{'─'*60}\n")


def save_repair_spec(
    detections: list[Detection],
    video_path: str | Path,
    out_path: str | Path,
) -> None:
    repairs = []
    for d in detections:
        if d.type == "skip":
            repairs.append({
                "type": "insert",
                "after": d.frame,
                "note": f"skip (conf={d.confidence:.2f}, ratio={d.evidence.get('flow_ratio', 0):.1f}x)",
            })
        elif d.type == "duplicate":
            repairs.append({
                "type": "remove",
                "frame": d.frame,
                "note": f"duplicate (conf={d.confidence:.2f}, ssim={d.evidence.get('ssim', 0):.4f})",
            })

    tails = [d for d in detections if d.type == "frozen_tail"]
    spec = {
        "video": str(video_path),
        "repairs": repairs,
    }
    if tails:
        spec["trim_to"] = tails[0].frame - 1
        spec["_note_trim"] = (
            "frozen_tail detected — apply --trim-to to remove it after repair"
        )

    with open(out_path, "w") as f:
        json.dump(spec, f, indent=2)
    print(f"  Repair spec saved to: {out_path}")


# ---------------------------------------------------------------------------
# Interpolation backends
# ---------------------------------------------------------------------------
#
# Priority:
#   1. rife-ncnn-vulkan binary (SOTA; set up once with --setup-rife)
#   2. DIS optical flow with ghost-detection quality gate
#   3. Simple pixel blend (last resort — clean, no artifacts)
#
# The DIS "ghost" problem: forward/backward warp creates see-through double
# exposures on fast-moving subjects. We detect this via two signals:
#   a) Edge density ratio — ghosting doubles every edge in the frame
#   b) Deviation from plain blend — large deviations signal warp failure
# When either fires, we fall back to blend (slight motion-blur, no ghost).


# --- rife-ncnn-vulkan binary -------------------------------------------------

def _rife_ncnn_dir() -> Path:
    return Path.home() / ".video-repair" / "rife-ncnn"


def _rife_ncnn_binary() -> Path | None:
    b = _rife_ncnn_dir() / "rife-ncnn-vulkan"
    return b if b.exists() and (b.stat().st_mode & 0o111) else None


def _rife_ncnn_model() -> str:
    """Return name of best available RIFE model in the install dir."""
    d = _rife_ncnn_dir()
    for name in ("rife-v4.6", "rife-v4", "rife-v3", "rife"):
        if (d / name).is_dir():
            return name
    # Fallback: first rife-* directory found
    for item in sorted(d.iterdir()):
        if item.is_dir() and "rife" in item.name.lower():
            return item.name
    return "rife-v4.6"


_rife_ok: bool | None = None   # cached availability flag


def _interpolate_rife_ncnn(
    img_a: np.ndarray, img_b: np.ndarray, out_path: Path
) -> bool:
    """
    Call rife-ncnn-vulkan binary for SOTA interpolation.
    Returns True on success. Falls back silently on failure.
    """
    global _rife_ok
    binary = _rife_ncnn_binary()
    if binary is None:
        _rife_ok = False
        return False

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        a_p = tmp / "a.png"
        b_p = tmp / "b.png"
        out_file = tmp / "out.png"

        Image.fromarray(img_a).save(str(a_p))
        Image.fromarray(img_b).save(str(b_p))

        import platform
        cmd = [
            str(binary),
            "-0", str(a_p),
            "-1", str(b_p),
            "-o", str(out_file),
            "-m", _rife_ncnn_model(),
        ]
        # CPU mode on Linux: -g -1 works correctly via native CPU path.
        # On macOS, -g -1 routes through MoltenVK and produces frame distortion — omit it there.
        if platform.system() == "Linux":
            cmd += ["-g", "-1"]
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode != 0:
            if _rife_ok is None:
                print(f"  [rife-ncnn] failed: {r.stderr.decode()[:200]}", file=sys.stderr)
            _rife_ok = False
            return False

        if not out_file.exists():
            _rife_ok = False
            return False

        shutil.copy2(str(out_file), str(out_path))
        _rife_ok = True
        return True


# --- DIS optical flow with ghost-detection gate ------------------------------

def _dis_warp(img_a: np.ndarray, img_b: np.ndarray) -> np.ndarray:
    """Raw DIS forward+backward warp, blended with occlusion confidence."""
    h, w = img_a.shape[:2]
    a_g = cv2.cvtColor(img_a, cv2.COLOR_RGB2GRAY)
    b_g = cv2.cvtColor(img_b, cv2.COLOR_RGB2GRAY)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    fwd = dis.calc(a_g, b_g, None)
    bwd = dis.calc(b_g, a_g, None)
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)

    def warp(img, flow, t=0.5):
        return cv2.remap(
            img,
            (gx + flow[:, :, 0] * t).astype(np.float32),
            (gy + flow[:, :, 1] * t).astype(np.float32),
            cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
        )

    wa = warp(img_a, fwd)
    wb = warp(img_b, bwd)

    consistency = np.sqrt(
        (fwd[:, :, 0] + bwd[:, :, 0]) ** 2 +
        (fwd[:, :, 1] + bwd[:, :, 1]) ** 2
    )
    conf = np.clip(
        1.0 - consistency / (np.percentile(consistency, 95) + 1e-6),
        0.0, 1.0
    )[:, :, np.newaxis]

    return (
        conf * 0.5 * wa.astype(np.float32) +
        conf * 0.5 * wb.astype(np.float32) +
        (1.0 - conf) * 0.5 * img_a.astype(np.float32) +
        (1.0 - conf) * 0.5 * img_b.astype(np.float32)
    ).clip(0, 255).astype(np.uint8)


def _ghost_score(warped: np.ndarray, img_a: np.ndarray, img_b: np.ndarray) -> float:
    """
    Returns 0–1: higher = more likely the DIS warp produced ghost artifacts.

    Uses two signals:
      1. Edge density ratio: ghosting creates duplicate edges, raising the
         edge count above either input. Ratio > 1.3 is a strong ghost signal.
      2. High-percentile deviation from simple blend: a correct warp should
         mostly agree with a plain blend; large deviations indicate failure.

    Both are evaluated at 1/4 resolution for speed.
    """
    scale = 4
    h, w = img_a.shape[:2]
    th, tw = max(1, h // scale), max(1, w // scale)

    def shrink(x):
        return cv2.resize(x, (tw, th), interpolation=cv2.INTER_AREA)

    ws = shrink(warped)
    as_ = shrink(img_a)
    bs = shrink(img_b)

    def edge_mag(img):
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        return float(np.mean(np.sqrt(gx ** 2 + gy ** 2)))

    em_w  = edge_mag(ws)
    em_ab = max(edge_mag(as_), edge_mag(bs)) + 1e-6
    edge_ratio = em_w / em_ab   # > 1 means warped has more edges than inputs

    blend = (as_.astype(np.float32) + bs.astype(np.float32)) / 2
    dev = np.abs(ws.astype(np.float32) - blend)
    p90_dev = float(np.percentile(dev, 90))   # 90th‑pct pixel diff from blend

    # Normalise to 0–1 scores
    edge_score = min(1.0, max(0.0, (edge_ratio - 1.0) / 0.5))   # 1.0→0, 1.5→1.0
    dev_score  = min(1.0, p90_dev / 25.0)                         # 25/255 → score=1

    return max(edge_score, dev_score)


def _interpolate_blend(img_a: np.ndarray, img_b: np.ndarray) -> np.ndarray:
    return (
        (img_a.astype(np.float32) + img_b.astype(np.float32)) / 2
    ).clip(0, 255).astype(np.uint8)


def _interpolate_dis_or_blend(
    img_a: np.ndarray, img_b: np.ndarray
) -> tuple[np.ndarray, str]:
    """
    Attempt DIS warp. If the ghost-score is high enough to indicate artifacts,
    fall back to pixel blend. Returns (result, method_name).
    """
    try:
        warped = _dis_warp(img_a, img_b)
        score  = _ghost_score(warped, img_a, img_b)
        if score < 0.40:
            return warped, f"DIS(gs={score:.2f})"
        # Ghost detected — blend is cleaner (slight motion-blur vs. double-exposure)
        return _interpolate_blend(img_a, img_b), f"blend(dis-ghost={score:.2f})"
    except Exception as e:
        return _interpolate_blend(img_a, img_b), f"blend(dis-err)"


# --- Top-level interpolator --------------------------------------------------

def interpolate_frame(path_a: Path, path_b: Path, out_path: Path) -> str:
    img_a = np.array(Image.open(path_a).convert("RGB"))
    img_b = np.array(Image.open(path_b).convert("RGB"))

    # 1. Try RIFE (SOTA, no artifacts on fast motion)
    if _rife_ncnn_binary() is not None:
        if _interpolate_rife_ncnn(img_a, img_b, out_path):
            return "RIFE"

    # 2. DIS with ghost-detection gate
    result, method = _interpolate_dis_or_blend(img_a, img_b)
    Image.fromarray(result).save(str(out_path), "PNG")
    return method


# ---------------------------------------------------------------------------
# Repair application
# ---------------------------------------------------------------------------

def apply_repairs(
    frames: list[Path],
    repairs: list[Repair],
    interp_dir: Path,
) -> list[Path]:
    interp_dir.mkdir(parents=True, exist_ok=True)
    # At the same frame number, a remove MUST run before an insert. Otherwise
    # (insert-then-remove) the two cancel out: the freshly inserted frame gets
    # popped and the duplicate is left in place. This happens on paired
    # skip+duplicate glitches (e.g. dup at frame N + skip after frame N).
    sorted_repairs = sorted(repairs, key=lambda r: (r.frame, 0 if r.type == "remove" else 1))
    result = list(frames)
    offset = 0
    interp_n = 0

    for rep in sorted_repairs:
        orig_idx = rep.frame - 1
        curr_idx = orig_idx + offset

        if rep.type == "insert":
            if curr_idx >= len(result) - 1:
                print(f"  Warning: cannot insert after frame {rep.frame} (out of bounds)")
                continue
            ipath = interp_dir / f"interp_{interp_n:04d}.png"
            method = interpolate_frame(result[curr_idx], result[curr_idx + 1], ipath)
            result.insert(curr_idx + 1, ipath)
            offset += 1
            interp_n += 1
            print(f"  [insert] after #{rep.frame} [{method}]" + (f"  {rep.note}" if rep.note else ""))

        elif rep.type == "remove":
            if curr_idx >= len(result):
                print(f"  Warning: cannot remove frame {rep.frame} (out of bounds)")
                continue
            result.pop(curr_idx)
            offset -= 1
            print(f"  [remove] #{rep.frame}" + (f"  {rep.note}" if rep.note else ""))

    return result


# ---------------------------------------------------------------------------
# Video assembly
# ---------------------------------------------------------------------------

def assemble_video(
    frame_list: list[Path],
    audio_source: str | Path,
    output_path: str | Path,
    fps: float,
    fmt: str = "prores",
    color_meta: dict | None = None,
) -> None:
    """
    Assemble frames into a video file.

    fmt:
      "prores"  — ProRes 422 HQ, 10-bit 4:2:2, .mov container (default)
                  No perceptible quality loss; safe for professional post.
      "h264"    — H.264 CRF 18, 8-bit 4:2:0, .mp4 container
                  Use only for web/delivery; introduces generation loss.

    color_meta: dict with keys color_primaries, color_trc, color_space,
                color_range — probed from source and forwarded to avoid
                colour-space shift in the re-encode.
    """
    fps_num = round(fps * 1000)
    fps_den = 1000
    from math import gcd
    g = gcd(fps_num, fps_den)
    fps_str = f"{fps_num // g}/{fps_den // g}"

    cm = color_meta or {}
    cp  = cm.get("color_primaries", "bt709")
    ct  = cm.get("color_trc",       "bt709")
    cs  = cm.get("color_space",     "bt709")
    cr  = cm.get("color_range",     "tv")
    # FFmpeg -color_range accepts "tv" (limited) or "pc" (full)
    cr_flag = "tv" if cr in ("tv", "limited", "mpeg") else "pc"

    if fmt == "prores":
        codec_flags = [
            "-c:v", "prores_ks",
            "-profile:v", "3",          # 422 HQ — broadcast quality
            "-pix_fmt", "yuv422p10le",  # 10-bit 4:2:2
            "-vendor", "apl0",          # Apple ProRes identifier
        ]
    else:
        codec_flags = [
            "-c:v", "libx264",
            "-preset", "slow",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
        ]

    # Does the source actually have an audio track? Many VFX/effects clips don't.
    # Mapping 1:a unconditionally makes ffmpeg fail on silent sources.
    has_audio = bool(
        subprocess.run(
            ["ffprobe", "-v", "quiet", "-select_streams", "a:0",
             "-show_entries", "stream=index", "-of", "csv=p=0", str(audio_source)],
            capture_output=True, text=True,
        ).stdout.strip()
    )

    with tempfile.TemporaryDirectory() as seq_dir:
        seq_path = Path(seq_dir)
        for i, src in enumerate(frame_list):
            dst = seq_path / f"{i + 1:08d}.png"
            try:
                dst.symlink_to(Path(src).resolve())
            except (OSError, NotImplementedError):
                shutil.copy2(str(src), str(dst))

        cmd = ["ffmpeg", "-y", "-framerate", fps_str, "-i", str(seq_path / "%08d.png")]
        if has_audio:
            cmd += ["-i", str(audio_source), "-map", "0:v", "-map", "1:a"]
        else:
            cmd += ["-map", "0:v"]
        cmd += [
            *codec_flags,
            "-color_primaries", cp,
            "-color_trc",       ct,
            "-colorspace",      cs,
            "-color_range",     cr_flag,
            "-r", fps_str,
        ]
        if has_audio:
            cmd += ["-c:a", "copy", "-shortest"]
        cmd += [str(output_path)]

        subprocess.run(cmd, check=True, capture_output=True)


# ---------------------------------------------------------------------------
# RIFE one-time setup  (rife-ncnn-vulkan binary — no Python/CUDA/SSL deps)
# ---------------------------------------------------------------------------

def setup_rife() -> None:
    """
    Download and install the rife-ncnn-vulkan standalone binary.

    Uses curl (system SSL) to avoid Python SSL certificate issues on macOS.
    The binary works on CPU (-g -1) without Vulkan/GPU drivers.
    """
    import platform, zipfile

    ncnn_dir = _rife_ncnn_dir()
    ncnn_dir.mkdir(parents=True, exist_ok=True)
    binary = ncnn_dir / "rife-ncnn-vulkan"

    if binary.exists():
        print(f"rife-ncnn-vulkan already installed at {ncnn_dir}")
        return

    sys_name = platform.system()
    if sys_name == "Darwin":
        url = (
            "https://github.com/nihui/rife-ncnn-vulkan/releases/download"
            "/20221029/rife-ncnn-vulkan-20221029-macos.zip"
        )
    elif sys_name == "Linux":
        url = (
            "https://github.com/nihui/rife-ncnn-vulkan/releases/download"
            "/20221029/rife-ncnn-vulkan-20221029-ubuntu.zip"
        )
    else:
        sys.exit(f"Unsupported platform: {sys_name}")

    zip_path = ncnn_dir / "rife-ncnn-vulkan.zip"
    print(f"Downloading rife-ncnn-vulkan (~100 MB) via curl...")
    subprocess.run(
        ["curl", "-L", "--progress-bar", "-o", str(zip_path), url],
        check=True,
    )

    print("Extracting...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(str(ncnn_dir))
    zip_path.unlink()

    # The zip extracts to a versioned subdirectory — flatten it
    for sub in ncnn_dir.iterdir():
        if sub.is_dir():
            bin_candidate = sub / "rife-ncnn-vulkan"
            if bin_candidate.exists():
                shutil.copy2(str(bin_candidate), str(binary))
                binary.chmod(0o755)
                for item in sub.iterdir():
                    dest = ncnn_dir / item.name
                    if not dest.exists():
                        if item.is_dir():
                            shutil.copytree(str(item), str(dest))
                        else:
                            shutil.copy2(str(item), str(dest))
                shutil.rmtree(str(sub))
                break

    print(f"\nRIFE installed at {ncnn_dir}")
    print("Run repair.py normally — it will use RIFE automatically.")


# ---------------------------------------------------------------------------
# Spec loading
# ---------------------------------------------------------------------------

def load_json_spec(path: str | Path) -> tuple[list[Repair], int | None]:
    """Returns (repairs, trim_to). trim_to may be None."""
    with open(path) as f:
        data = json.load(f)
    repairs = []
    for item in data.get("repairs", []):
        t, note = item["type"], item.get("note", "")
        if t == "insert":
            repairs.append(Repair("insert", int(item["after"]), note))
        elif t == "remove":
            repairs.append(Repair("remove", int(item["frame"]), note))
    trim_to = data.get("trim_to")
    return repairs, trim_to


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Detect and repair frame drops / duplicates in AI-generated video.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # Auto-detect frame issues, print report, save repair spec
  python repair.py input.mp4 --detect

  # Detect and immediately repair in one step
  python repair.py input.mp4 --detect --auto-repair

  # Repair from a saved spec
  python repair.py input.mp4 --repair repairs.json

  # Manual repairs
  python repair.py input.mp4 --insert-after 13 --remove 36 --trim-to 98

  # Set up RIFE for SOTA interpolation quality (one-time ~50 MB download)
  python repair.py --setup-rife
        """,
    )
    p.add_argument("video", nargs="?", help="Input video file")
    p.add_argument("--output", "-o",   help="Output path (default: <input>_repaired.mov)")
    p.add_argument("--format",         choices=["prores", "h264"], default="prores",
                   help="Output codec: prores = ProRes 422 HQ lossless-quality .mov (default); "
                        "h264 = H.264 CRF18 .mp4 for web delivery")
    p.add_argument("--detect",         action="store_true",
                   help="Analyse video and detect frame issues")
    p.add_argument("--auto-repair",    action="store_true",
                   help="Immediately repair after --detect (combines both steps)")
    p.add_argument("--spec-out",       metavar="FILE",
                   help="Where to save the detected repair spec JSON (default: <input>_repairs.json)")
    p.add_argument("--repair", "-r",   help="Apply repairs from a JSON spec file")
    p.add_argument("--insert-after",   type=int, action="append", metavar="FRAME")
    p.add_argument("--remove",         type=int, action="append", metavar="FRAME")
    p.add_argument("--trim-to",        type=int, metavar="FRAME",
                   help="Keep only the first N frames of the output")
    p.add_argument("--setup-rife",     action="store_true",
                   help="Download RIFE model weights for SOTA interpolation")
    p.add_argument("--fps",            type=float, help="Override detected FPS")
    p.add_argument("--backend",
                   choices=["auto", "rife", "dis", "blend"], default="auto",
                   help="Force interpolation backend")
    # Detection tuning
    p.add_argument("--ssim-dup",       type=float, default=0.97, metavar="THRESH",
                   help="SSIM threshold for duplicate detection (default 0.97)")
    p.add_argument("--skip-ratio",     type=float, default=1.75, metavar="RATIO",
                   help="Flow/baseline ratio threshold for skip detection (default 1.75)")
    p.add_argument("--min-confidence", type=float, default=0.70, metavar="CONF",
                   help="Minimum detection confidence to report (default 0.70)")
    p.add_argument("--verbose-detect", action="store_true",
                   help="Print per-frame-pair feature table during detection")

    args = p.parse_args()

    if args.setup_rife:
        setup_rife()
        return

    if not args.video:
        p.error("video argument required (or use --setup-rife)")

    video_path = Path(args.video).resolve()
    if not video_path.exists():
        sys.exit(f"Error: {video_path} not found")

    # Force backend (override RIFE check)
    if args.backend in ("dis", "blend"):
        global _rife_ok
        _rife_ok = False

    info = get_video_info(video_path)
    fps  = args.fps if args.fps else info["fps"]

    # ------------------------------------------------------------------
    # Detection pass
    # ------------------------------------------------------------------
    detected_repairs: list[Repair] = []
    spec_trim_to: int | None = None

    if args.detect:
        print(f"\nDetecting frame issues in {video_path.name}...")
        detections = detect_issues(
            video_path,
            ssim_dup_threshold=args.ssim_dup,
            skip_flow_ratio=args.skip_ratio,
            min_confidence=args.min_confidence,
            verbose=args.verbose_detect,
        )
        print_detection_report(detections, video_path, info, fps)

        # Save spec
        spec_path = Path(args.spec_out) if args.spec_out else \
                    video_path.with_name(video_path.stem + "_repairs.json")
        save_repair_spec(detections, video_path, spec_path)

        if not args.auto_repair:
            print("  Use --auto-repair to apply these repairs, or edit the spec and run:")
            print(f"  python repair.py {video_path.name} --repair {spec_path.name}")
            return

        # Convert detections to repairs for immediate use
        for d in detections:
            r = d.to_repair()
            if r:
                detected_repairs.append(r)
        tails = [d for d in detections if d.type == "frozen_tail"]
        if tails and spec_trim_to is None:
            spec_trim_to = tails[0].frame - 1

    # ------------------------------------------------------------------
    # Repair pass
    # ------------------------------------------------------------------
    repairs: list[Repair] = []

    if args.repair:
        loaded, file_trim = load_json_spec(args.repair)
        repairs.extend(loaded)
        if file_trim is not None and spec_trim_to is None:
            spec_trim_to = file_trim

    repairs.extend(detected_repairs)

    for f in (args.insert_after or []):
        repairs.append(Repair("insert", f))
    for f in (args.remove or []):
        repairs.append(Repair("remove", f))

    trim_to = args.trim_to if args.trim_to is not None else spec_trim_to

    if not repairs and trim_to is None:
        p.error("No repairs specified. Use --detect, --repair, --insert-after/--remove, or --trim-to.")

    out_ext = ".mov" if args.format == "prores" else ".mp4"
    output_path = (
        Path(args.output).resolve() if args.output
        else video_path.with_name(video_path.stem + "_repaired" + out_ext)
    )

    print(f"\nInput:  {video_path.name}")
    print(f"        {info['nb_frames']} frames @ {fps:.4g} fps  "
          f"{info['width']}×{info['height']}")
    if repairs:
        print(f"Repairs: {len(repairs)}")
    if trim_to:
        print(f"Trim to: {trim_to} frames")

    if any(r.type == "insert" for r in repairs):
        has_rife = _rife_ncnn_binary() is not None
        print(f"Backend: {'RIFE (ncnn)' if has_rife else 'DIS+blend (adaptive)'}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        frames = extract_frames(video_path, tmpdir / "frames")
        print(f"\nExtracting frames... {len(frames)} extracted")

        if repairs:
            print("\nApplying repairs...")
            frames = apply_repairs(frames, repairs, tmpdir / "interp")
            print(f"  → {len(frames)} frames ({len(frames)/fps:.3f}s)")

        if trim_to and len(frames) > trim_to:
            trimmed = len(frames) - trim_to
            frames = frames[:trim_to]
            print(f"\nTrimmed {trimmed} frames → {len(frames)} ({len(frames)/fps:.3f}s)")

        codec_label = "ProRes 422 HQ" if args.format == "prores" else "H.264 CRF18"
        print(f"\nAssembling → {output_path.name}  [{codec_label}]")
        assemble_video(frames, video_path, output_path, fps,
                       fmt=args.format, color_meta=info)

    mb = output_path.stat().st_size / 1_048_576
    print(f"Done.  {output_path}  ({mb:.1f} MB)")


if __name__ == "__main__":
    main()
