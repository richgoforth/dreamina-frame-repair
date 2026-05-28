# video-repair — Claude Code Instructions

## What This Is

A CLI tool for detecting and fixing frame-level artifacts in AI-generated video (Dreamina, Kling, etc.):
- **Frame skips**: Dreamina drops a frame, creating a motion jump → fix by interpolating a new frame
- **Duplicate frames**: Dreamina repeats a frame, creating a stutter → fix by removing the duplicate
- **Frozen tail**: Dreamina repeats the last frame for the rest of the video duration → fix by trimming

## Stack

- Python 3 / single-file CLI (`repair.py`)
- OpenCV (`cv2`) — DIS optical flow, frame I/O
- Pillow — PNG frame I/O
- NumPy — array ops
- FFmpeg — frame extraction and video assembly (subprocess)
- **rife-ncnn-vulkan** — SOTA neural frame interpolation binary (~/.video-repair/rife-ncnn/)
  - Run `python3 repair.py --setup-rife` once to install
  - Uses Apple Silicon GPU via MoltenVK (do NOT use -g -1 CPU mode — it produces distorted output)

## Key Design Decisions

- **Single file**: Everything in `repair.py`. No packages, no modules.
- **1-indexed frame numbers throughout**: All user-facing frame numbers (repair specs, CLI output) are 1-indexed. Internal list indexing is 0-indexed (`frame - 1`).
- **Symlink-based FFmpeg assembly**: Frames are assembled via a temp dir of sequentially named symlinks passed as `-framerate N -i %08d.png`. This avoids the concat demuxer bug that changes FPS to 25 and introduces duplicate frames.
- **RIFE GPU mode**: rife-ncnn-vulkan must be called WITHOUT `-g -1`. The CPU path produces catastrophic distortion on macOS. MoltenVK handles GPU correctly.

## Detection Pipeline

1. Extract frames at reduced width (640px) for analysis
2. Compute DIS optical flow magnitude + SSIM for every consecutive pair
3. Local median baseline (rolling window, excludes self) normalizes for scene dynamics
4. **Skip detection**: flow/baseline ratio ≥ 1.75 → skip candidate; scored by temporal fit + direction consistency
5. **Duplicate detection**: SSIM ≥ 0.97 → duplicate candidate; confidence weighted by SSIM + flow
6. **Frozen tail**: 3+ consecutive pairs with SSIM ≥ 0.995 at end of video
7. Non-maximum suppression collapses nearby same-type detections

## Interpolation Priority

1. **RIFE ncnn** (SOTA — use this whenever binary is present)
2. **DIS optical flow** with ghost-detection gate (edge density + p90 deviation from blend)
3. **Pixel blend** (last resort — clean, no artifacts, slight motion blur)

## Usage

```bash
python3 repair.py --setup-rife                    # one-time install
python3 repair.py input.mp4 --detect              # analyse + save spec
python3 repair.py input.mp4 --repair spec.json    # repair from spec
python3 repair.py input.mp4 --detect --auto-repair --output out.mp4
```

## Known Issues / Upcoming Work

- Frame count not always preserved: skips (+1 each) and duplicates (−1 each) only cancel when counts match. Planned: `--preserve-count` mode that replaces duplicates instead of deleting them (net 0 per paired glitch).
- `apply_repairs` sort is stable so remove-before-insert at same frame number works, but relies on JSON input order. Should add explicit type-priority sort key.
