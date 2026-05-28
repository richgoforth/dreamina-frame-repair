#!/usr/bin/env bash
#
# Frame Repair — one-line installer (macOS).
#
#   curl -fsSL https://raw.githubusercontent.com/richgoforth/dreamina-frame-repair/master/install.sh | bash
#
# Installs everything into ~/.frame-repair (isolated — touches nothing else)
# and adds a `frame-repair` command. Footage never leaves the machine.

set -euo pipefail

REPO_URL="https://github.com/richgoforth/dreamina-frame-repair.git"
DEST="$HOME/.frame-repair"

say() { printf "\033[0;36m%s\033[0m\n" "$1"; }
ok()  { printf "\033[0;32m✓ %s\033[0m\n" "$1"; }
err() { printf "\033[0;31m✗ %s\033[0m\n" "$1" >&2; }

say "Frame Repair — installer"
echo

# 1. Python 3
if ! command -v python3 >/dev/null 2>&1; then
  err "Python 3 not found."
  echo "  Install it:  brew install python3   (or https://www.python.org/downloads/)"
  exit 1
fi
ok "Python 3: $(python3 --version 2>&1)"

# 2. FFmpeg
if ! command -v ffmpeg >/dev/null 2>&1; then
  if command -v brew >/dev/null 2>&1; then
    say "Installing FFmpeg via Homebrew (one-time)…"
    brew install ffmpeg
  else
    err "FFmpeg not found, and Homebrew isn't installed."
    echo "  Install Homebrew from https://brew.sh then re-run this installer."
    exit 1
  fi
fi
ok "FFmpeg: $(ffmpeg -version 2>&1 | head -1 | cut -d' ' -f1-3)"

# 3. Get the code
if [ -d "$DEST/.git" ]; then
  say "Updating existing install…"
  git -C "$DEST" pull --quiet --ff-only || true
else
  say "Downloading Frame Repair…"
  git clone --quiet "$REPO_URL" "$DEST"
fi
ok "Code installed at $DEST"

# 4. Isolated Python environment + dependencies
say "Setting up Python environment…"
python3 -m venv "$DEST/venv"
"$DEST/venv/bin/pip" install --quiet --upgrade pip
"$DEST/venv/bin/pip" install --quiet -r "$DEST/requirements.txt"
ok "Dependencies installed"

# 5. RIFE neural interpolation (~100 MB, one-time). Needs a GPU at run time —
#    every modern Mac has one (Metal via MoltenVK).
say "Installing RIFE neural interpolation (~100 MB, one-time)…"
"$DEST/venv/bin/python" "$DEST/repair.py" --setup-rife
ok "RIFE installed"

# 6. `frame-repair` launcher command
LAUNCHER="/usr/local/bin/frame-repair"
TMP_LAUNCHER="$(mktemp)"
cat > "$TMP_LAUNCHER" <<EOF
#!/usr/bin/env bash
exec "$DEST/venv/bin/python" "$DEST/app.py" "\$@"
EOF
chmod +x "$TMP_LAUNCHER"
if [ -w "$(dirname "$LAUNCHER")" ]; then
  mv "$TMP_LAUNCHER" "$LAUNCHER"
else
  say "Adding the 'frame-repair' command (may ask for your Mac password)…"
  sudo mv "$TMP_LAUNCHER" "$LAUNCHER"
fi
ok "Command installed: frame-repair"

echo
ok "Done."
echo
say "To start:   frame-repair"
echo "Your browser opens automatically. Drop a video, wait, download. That's it."
