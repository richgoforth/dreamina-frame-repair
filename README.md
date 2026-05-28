# Frame Repair

Detects and fixes the frame-level artifacts that AI video generators (Dreamina, etc.) leave in their output:

- **Frame skips** — a dropped frame causes a visible motion jump. Fixed by synthesizing the missing frame with RIFE neural interpolation.
- **Duplicate frames** — a repeated frame causes a stutter. Fixed by removing the duplicate.
- **Frozen tail** — the last frame repeats for the rest of the clip. Fixed by trimming.

Output is **ProRes 422 HQ** (10-bit 4:2:2) — no generation loss, ready for professional post.

## Install (macOS)

```bash
curl -fsSL https://raw.githubusercontent.com/richgoforth/dreamina-frame-repair/master/install.sh | bash
```

That's it. The installer sets up an isolated environment in `~/.frame-repair`, downloads the RIFE model, and adds a `frame-repair` command. It touches nothing else on your system.

## Use

```bash
frame-repair
```

Your browser opens. Drop in a video, watch it process, download the repaired `.mov`. **Your footage never leaves your machine** — everything runs locally.

## Why local (not a website)

Clean interpolation requires a GPU, and your Mac has one (Metal). A hosted server typically doesn't, so it would fall back to lower-quality interpolation that ghosts on fast motion. Running locally also means unreleased footage stays on your machine — nothing is uploaded anywhere.

## Command line

The web UI wraps a CLI you can also use directly:

```bash
# Detect issues, print a report, save a repair spec
python3 repair.py input.mp4 --detect

# Detect and repair in one step
python3 repair.py input.mp4 --detect --auto-repair

# Repair from a saved/edited spec
python3 repair.py input.mp4 --repair repairs.json

# Manual control
python3 repair.py input.mp4 --insert-after 14 --remove 16 --trim-to 99

# H.264 output for web delivery instead of ProRes
python3 repair.py input.mp4 --detect --auto-repair --format h264
```

## How detection works

For every adjacent frame pair: dense optical flow (DIS) measures real motion, SSIM measures similarity. A rolling median baseline normalizes for scene dynamics. Skips are confirmed against a "double-step" motion model plus a direction-consistency check (rejects scene cuts); duplicates are confirmed by near-perfect SSIM with near-zero flow; the frozen tail is a run of identical frames reaching the end. Non-maximum suppression collapses clustered detections to one event each.

## Interpolation backends (in priority order)

1. **RIFE** (rife-ncnn-vulkan) — SOTA neural interpolation. Needs a GPU; Apple Silicon and Intel Macs both work via Metal.
2. **DIS optical flow** with a ghost-detection gate — used if no GPU is available. Ghosts on fast motion.
3. **Pixel blend** — last-resort fallback.
