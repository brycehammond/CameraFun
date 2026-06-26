#!/usr/bin/env bash
#
# Install (or reinstall) the Depth Anything display as an always-on macOS
# LaunchAgent. It starts at login and relaunches automatically if it dies.
#
#   ./install_launchd.sh            # install + start
#   ./install_launchd.sh uninstall  # stop + remove
#
# Notes:
#  - This is a per-user LaunchAgent (~/Library/LaunchAgents), because the app
#    needs the GUI session for the fullscreen window and camera.
#  - The terminal/agent must have Camera permission. On first run macOS will
#    prompt; if it doesn't, add the python binary under System Settings >
#    Privacy & Security > Camera.
#  - Expects the Core ML model at ./depth_anything_v2_518.mlpackage. Generate it
#    with:  .venv/bin/python convert_coreml.py
#    (Or edit the plist template to drop --coreml and use the MPS path.)

set -euo pipefail

LABEL="com.bryce.depthdisplay"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$PROJECT_DIR/.venv/bin/python"
TEMPLATE="$PROJECT_DIR/launchd/$LABEL.plist"
AGENTS_DIR="$HOME/Library/LaunchAgents"
PLIST="$AGENTS_DIR/$LABEL.plist"
UID_NUM="$(id -u)"

if [[ "${1:-}" == "uninstall" ]]; then
    echo "Stopping and removing $LABEL ..."
    launchctl bootout "gui/$UID_NUM/$LABEL" 2>/dev/null || true
    rm -f "$PLIST"
    echo "Removed $PLIST"
    exit 0
fi

# Sanity checks before we wire it into launchd.
[[ -x "$PYTHON" ]] || { echo "ERROR: venv python not found at $PYTHON. Create it: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2; exit 1; }
[[ -f "$TEMPLATE" ]] || { echo "ERROR: plist template missing at $TEMPLATE" >&2; exit 1; }
if [[ ! -d "$PROJECT_DIR/depth_anything_v2_518.mlpackage" ]]; then
    echo "WARNING: depth_anything_v2_518.mlpackage not found." >&2
    echo "         Run: $PYTHON convert_coreml.py" >&2
    echo "         (Continuing — the agent will crash-loop until the model exists.)" >&2
fi

mkdir -p "$AGENTS_DIR" "$PROJECT_DIR/logs"

# Fill the template placeholders and write the real plist.
sed -e "s#__PYTHON__#$PYTHON#g" \
    -e "s#__PROJECT_DIR__#$PROJECT_DIR#g" \
    -e "s#__HOME__#$HOME#g" \
    "$TEMPLATE" > "$PLIST"
echo "Wrote $PLIST"

# Reload: bootout an old instance (ignore if absent), then bootstrap + kickstart.
launchctl bootout "gui/$UID_NUM/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$UID_NUM" "$PLIST"
launchctl kickstart -k "gui/$UID_NUM/$LABEL"

echo "Loaded and started $LABEL."
echo "Logs: $PROJECT_DIR/logs/depthdisplay.{out,err}.log"
echo "Status:  launchctl print gui/$UID_NUM/$LABEL | grep -E 'state|pid'"
echo "Stop:    ./install_launchd.sh uninstall"
