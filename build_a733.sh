#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
BUILD_DIR="$ROOT/native/a733_npu/build"
DEPS_DIR="$ROOT/native/a733_npu/.deps"
ARCHIVE="$DEPS_DIR/allwinner-model-zoo.tar.gz"
URL="https://dl.radxa.com/cubie/allwinner-model-zoo.tar.gz"

if [ "$(uname -m)" != "x86_64" ] && [ "$(uname -m)" != "aarch64" ]; then
    echo "Unsupported build host: $(uname -m)" >&2
    exit 1
fi
if [ ! -x /usr/bin/aarch64-linux-gnu-g++ ]; then
    echo "Missing /usr/bin/aarch64-linux-gnu-g++; install the AArch64 cross compiler." >&2
    exit 1
fi

mkdir -p "$DEPS_DIR"
if [ ! -f "$ARCHIVE" ]; then
    echo "Downloading the official Allwinner A733 model-zoo runtime..."
    curl -L --fail --retry 2 "$URL" -o "$ARCHIVE"
fi

ROOT_NAME=$(tar -tzf "$ARCHIVE" | sed -n 's#^\([^/]*/\)common/npuruntime/$#\1#p' | head -n 1 | sed 's#/$##')
if [ -z "$ROOT_NAME" ]; then
    echo "Could not find common/npuruntime in the model-zoo archive." >&2
    exit 1
fi
rm -rf "$DEPS_DIR/model-zoo"
mkdir -p "$DEPS_DIR/model-zoo"
tar -xzf "$ARCHIVE" -C "$DEPS_DIR/model-zoo" --strip-components=1 \
    "$ROOT_NAME/common/npuruntime"

rm -rf "$BUILD_DIR"
cmake -S "$ROOT/native/a733_npu" -B "$BUILD_DIR" \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
    -DCMAKE_C_COMPILER=/usr/bin/aarch64-linux-gnu-gcc \
    -DCMAKE_CXX_COMPILER=/usr/bin/aarch64-linux-gnu-g++ \
    -DA733_MODEL_ZOO_DIR="$DEPS_DIR/model-zoo"
cmake --build "$BUILD_DIR" --parallel
cp "$BUILD_DIR/liba733_npu.so" "$ROOT/liba733_npu.so"
file "$ROOT/liba733_npu.so"
