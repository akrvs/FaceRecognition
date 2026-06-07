#!/usr/bin/env bash
set -euo pipefail

DEST="${1:-data/lfw}"
URL="https://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz"
ARCHIVE="$(mktemp --suffix=.tgz)"

mkdir -p "$DEST"
echo "Downloading LFW (deep funneled) to $DEST"
curl -L "$URL" -o "$ARCHIVE"
tar -xzf "$ARCHIVE" -C "$DEST" --strip-components=1
rm -f "$ARCHIVE"
echo "Done. Run: visage evaluate $DEST --pairs 2000"
