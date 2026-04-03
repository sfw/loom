#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: build_macos_desktop_dmg.sh --app-bundle <path> --output <path> [--volume-name <name>]

Create a simple macOS drag-install DMG containing the built Loom Desktop app and
an Applications symlink. This avoids Tauri's Finder-automation DMG wrapper.

Pass --open to open the finished DMG in Finder after creation.
EOF
}

APP_BUNDLE=""
OUTPUT_PATH=""
VOLUME_NAME="Loom Desktop"
OPEN_DMG=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --app-bundle)
      APP_BUNDLE="${2:-}"
      shift 2
      ;;
    --output)
      OUTPUT_PATH="${2:-}"
      shift 2
      ;;
    --volume-name)
      VOLUME_NAME="${2:-}"
      shift 2
      ;;
    --open)
      OPEN_DMG=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$APP_BUNDLE" || -z "$OUTPUT_PATH" ]]; then
  usage >&2
  exit 1
fi

APP_BUNDLE="$(cd "$(dirname "$APP_BUNDLE")" && pwd)/$(basename "$APP_BUNDLE")"
OUTPUT_PATH="$(cd "$(dirname "$OUTPUT_PATH")" && pwd)/$(basename "$OUTPUT_PATH")"

if [[ ! -d "$APP_BUNDLE" ]]; then
  echo "App bundle not found: $APP_BUNDLE" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT_PATH")"
rm -f "$OUTPUT_PATH"

STAGING_DIR="$(mktemp -d "${TMPDIR:-/tmp}/loom-desktop-dmg.XXXXXX")"
cleanup() {
  rm -rf "$STAGING_DIR"
}
trap cleanup EXIT

cp -R "$APP_BUNDLE" "$STAGING_DIR/"
ln -s /Applications "$STAGING_DIR/Applications"

hdiutil create \
  -volname "$VOLUME_NAME" \
  -srcfolder "$STAGING_DIR" \
  -ov \
  -format UDZO \
  "$OUTPUT_PATH"

echo "Created DMG at $OUTPUT_PATH"

if [[ "$OPEN_DMG" == "1" ]]; then
  open "$OUTPUT_PATH"
fi
