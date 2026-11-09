#!/bin/bash
# SPDX-License-Identifier: MIT
# build.sh: compile decode_floor for gfx1100 (ROCM: a ROCm or TheRock install with hipcc)
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
ROCM=${ROCM:-/opt/rocm}
mkdir -p "$HERE/build"
# TILE_1RT=1: the one-round-trip INT4 tile (TILE-1RT.md), built as decode_floor_1rt
NAME=decode_floor; FLAGS=""
if [ "${TILE_1RT:-0}" = 1 ]; then NAME=decode_floor_1rt; FLAGS="-DTILE_1RT"; fi
"$ROCM/bin/hipcc" -O3 --offload-arch=gfx1100 -std=c++17 -Wall $FLAGS "$HERE/src/decode_floor.hip" -o "$HERE/build/$NAME"
echo "built $HERE/build/$NAME"
