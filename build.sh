#!/bin/bash
# SPDX-License-Identifier: MIT
# build.sh: compile decode_floor for gfx1100 (ROCM: a ROCm or TheRock install with hipcc)
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
ROCM=${ROCM:-$HOME/rocm-therock/10.2.0a20261011}
mkdir -p "$HERE/build"
"$ROCM/bin/hipcc" -O3 --offload-arch=gfx1100 -std=c++17 -Wall "$HERE/src/decode_floor.hip" -o "$HERE/build/decode_floor"
echo "built $HERE/build/decode_floor"
