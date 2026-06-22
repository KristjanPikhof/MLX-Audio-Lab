#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

DERIVED_DATA="${MLX_AUDIO_LAB_DERIVED_DATA:-$PWD/.derivedData}"
BUILD_DESTINATION="${MLX_AUDIO_LAB_DESTINATION:-platform=macOS,arch=arm64}"
PRODUCTS_DIR="$DERIVED_DATA/Build/Products/Debug"
APP_BUNDLE="$PWD/.run/MLX Audio Lab.app"
APP_CONTENTS="$APP_BUNDLE/Contents"
APP_MACOS="$APP_CONTENTS/MacOS"
APP_RESOURCES="$APP_CONTENTS/Resources"

xcodebuild \
  -quiet \
  -scheme MLXAudioLab \
  -destination "$BUILD_DESTINATION" \
  -derivedDataPath "$DERIVED_DATA" \
  COMPILER_INDEX_STORE_ENABLE=NO \
  build

rm -rf "$APP_BUNDLE"
mkdir -p "$APP_MACOS" "$APP_RESOURCES"

cp "$PRODUCTS_DIR/MLXAudioLab" "$APP_MACOS/MLXAudioLab"
cp "$PWD/Sources/MLXAudioLab/Info.plist" "$APP_CONTENTS/Info.plist"
cp "$PWD/Sources/MLXAudioLab/Resources/AppIcon.icns" "$APP_RESOURCES/AppIcon.icns"

for bundle in "$PRODUCTS_DIR"/*.bundle; do
  [ -d "$bundle" ] || continue
  cp -R "$bundle" "$APP_RESOURCES/"
done

METALLIB="$(find "$PRODUCTS_DIR/mlx-swift_Cmlx.bundle" -name default.metallib -print -quit)"
if [ -n "$METALLIB" ]; then
  cp "$METALLIB" "$APP_MACOS/mlx.metallib"
  cp "$METALLIB" "$APP_RESOURCES/default.metallib"
fi

codesign --force --deep --sign - "$APP_BUNDLE" >/dev/null
open -n "$APP_BUNDLE"

printf 'Opened %s\n' "$APP_BUNDLE"
printf 'App bundle: %s\n' "$APP_BUNDLE"
printf 'App folder: %s\n' "$(dirname "$APP_BUNDLE")"
