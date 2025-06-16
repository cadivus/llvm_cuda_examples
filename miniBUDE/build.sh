#!/usr/bin/env bash

set -eu

#MODEL=? A=B

BUILD_DIR="./build"

rm -rf "$BUILD_DIR"

if [ "$model" = "cuda_native" ]; then
    flags="-DCMAKE_CUDA_COMPILER=$CMAKE_CUDA_COMPILER -DCUDA_ARCH=$GPU_ARCH"
    model="cuda"
fi

cmake -B "$BUILD_DIR" -H. \
    -DCMAKE_BUILD_TYPE=Release \
    -DMODEL="$model" $flags

cmake --build "$BUILD_DIR" -j "$(nproc)"
