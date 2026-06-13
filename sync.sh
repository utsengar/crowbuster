#!/usr/bin/env bash
# Continuously sync this folder to a remote machine on every file change.
#
# Requires: fswatch (macOS: brew install fswatch)
#
# Configuration — set these via env vars or put them in .env:
#   REMOTE_USER   username on the remote machine
#   REMOTE_HOST   IP or hostname of the remote machine
#   REMOTE_PATH   destination directory on remote (default: ~/crowbuster/)
#
# Usage:
#   cp .env.example .env       # then edit .env
#   ./sync.sh
# Stop with Ctrl+C.

set -e

LOCAL_PATH="$(cd "$(dirname "$0")" && pwd)/"

# Auto-load .env if present
if [ -f "${LOCAL_PATH}.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "${LOCAL_PATH}.env"
  set +a
fi

: "${REMOTE_USER:?Set REMOTE_USER in .env or as an env var}"
: "${REMOTE_HOST:?Set REMOTE_HOST in .env or as an env var}"
REMOTE_PATH="${REMOTE_PATH:-~/crowbuster/}"

RSYNC_ARGS=(
  -av
  --delete
  --exclude='.venv'
  --exclude='__pycache__'
  --exclude='events.log'
  --exclude='heartbeat'
  --exclude='captures'
  --exclude='.git'
  --exclude='yolov8n.pt'
  # Per-machine secrets — never sync. Each host owns its own .env. Previously
  # caused eva's CROWBUSTER_NTFY_TOPIC to be silently clobbered on every sync.
  --exclude='.env'
)

echo "Initial sync $LOCAL_PATH -> $REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"
rsync "${RSYNC_ARGS[@]}" "$LOCAL_PATH" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"

echo "Watching for changes... (Ctrl+C to stop)"
fswatch -o "$LOCAL_PATH" | while read -r _; do
  echo "[$(date +%H:%M:%S)] change detected — syncing"
  rsync "${RSYNC_ARGS[@]}" "$LOCAL_PATH" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"
done
