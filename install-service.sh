#!/usr/bin/env bash
# Install crowbuster as a systemd user service.
# After this runs, crowbuster will:
#   - start automatically at boot
#   - restart automatically if it crashes
#   - keep running when you log out (via linger)
#   - log to systemd's journal (queryable with journalctl)
#
# Usage (run on the machine that will host crowbuster, NOT on your dev mac):
#   ./install-service.sh

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
TARGET_DIR="$HOME/.config/systemd/user"
UNIT="$TARGET_DIR/crowbuster.service"

if [ ! -f "$HERE/crowbuster.py" ]; then
  echo "error: crowbuster.py not found in $HERE — run this from the repo root" >&2
  exit 1
fi

if [ ! -d "$HERE/.venv" ]; then
  echo "error: .venv missing — run 'python3 -m venv .venv && pip install -r requirements.txt' first" >&2
  exit 1
fi

mkdir -p "$TARGET_DIR"
cp "$HERE/crowbuster.service" "$UNIT"
echo "installed unit: $UNIT"

systemctl --user daemon-reload
systemctl --user enable crowbuster.service
systemctl --user restart crowbuster.service

# Allow the service to run when the user is not logged in.
if ! loginctl show-user "$USER" 2>/dev/null | grep -q '^Linger=yes'; then
  echo "enabling user lingering (requires sudo) so the service runs across logouts..."
  sudo loginctl enable-linger "$USER"
fi

echo ""
echo "✅ crowbuster installed as a systemd user service"
echo ""
echo "  status:    systemctl --user status crowbuster"
echo "  logs:      journalctl --user -u crowbuster -f"
echo "  restart:   systemctl --user restart crowbuster"
echo "  stop:      systemctl --user stop crowbuster"
echo "  uninstall: systemctl --user disable --now crowbuster && rm $UNIT"
echo ""
systemctl --user status crowbuster --no-pager || true
