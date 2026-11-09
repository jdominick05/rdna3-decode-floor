#!/bin/bash
# SPDX-License-Identifier: MIT
# run.sh smoke|xtx: the GPU runs of PLAN.md. Each stage runs under GUARD when it is set: a watchdog called as
# `GUARD LOG SECS CMD...` that writes CMD's output to LOG (ours also pins the clocks at power_dpm_force_performance_level
# "high", watches VRAM and stops on a GPU fault in the kernel log; exit 99 / 98 = fault). Without GUARD each stage runs
# under `timeout SECS` with its output in the same LOG, and the clocks are whatever the system sets (pin them for
# comparable numbers). A watchdog fault or a failed stage stops the script; a stage that ends with 6 (a persistent
# variant failed its wait or dependency check; its rows say "fail") is recorded and the script goes on.
#   ROCM   a ROCm install with hipcc and rocprofv3 (default /opt/rocm)
#   DEVICE the HIP device index of the card (default: smoke 1, xtx 0; --expect-cus refuses any other card model)
#   EXPECT_CUS  smoke only: the WGP count the card must report (default 42, the RX 7900 XT; 48 runs smoke on an XTX)
#   BIN / TAG  another build of decode_floor, and a suffix for the results folder (e.g. BIN=build/decode_floor_1rt TAG=1rt)
#   PCI    its PCI address, for the environment record (clocks, link state; default: smoke 0000:06:00.0, xtx 0000:03:00.0)
#   smoke  the RX 7900 XT alone (device 1, 84 CUs = 42 WGPs; HIP reports WGPs on RDNA3), small sizes: nothing reported
#   xtx    the RX 7900 XTX (device 0, 96 CUs = 48 WGPs): E0, the floor split, E1, sync, E2, E3 (+ occupancy, one-tile claims)
# Results: results/<date>-<mode>/ (CSV files with headers, rocprofv3 kernel traces, logs, environment, tables.md).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
MODE=${1:?usage: run.sh smoke|xtx}
BIN=${BIN:-$HERE/build/decode_floor}
GUARD=${GUARD:-}
ROCM=${ROCM:-/opt/rocm}
PROF=$ROCM/bin/rocprofv3
OUT=$HERE/results/$(date +%Y%m%d-%H%M%S)-$MODE${TAG:+-$TAG}
mkdir -p "$OUT"
[ -x "$BIN" ] || { echo "build first: ./build.sh"; exit 2; }
if [ -n "$GUARD" ] && [ ! -x "$GUARD" ]; then echo "GUARD=$GUARD is not executable"; exit 2; fi
[ -x "$PROF" ] || { echo "no rocprofv3 at $PROF (set ROCM)"; exit 2; }
[ -n "$GUARD" ] || echo "no GUARD: stages run under timeout only, clocks not pinned"

case $MODE in
  smoke) export HIP_VISIBLE_DEVICES=${DEVICE:-1}; DEV=(--expect-cus ${EXPECT_CUS:-42}); PCI=${PCI:-0000:06:00.0} ;;
  xtx)   export HIP_VISIBLE_DEVICES=${DEVICE:-0}; DEV=(--expect-cus 48); PCI=${PCI:-0000:03:00.0} ;;
  *) echo "usage: run.sh smoke|xtx"; exit 2 ;;
esac

stage() {   # stage NAME SECS CMD...
  local name=$1 secs=$2; shift 2
  echo "=== $(date +%H:%M:%S) $name" | tee -a "$OUT/run.log"
  if [ -n "$GUARD" ]; then "$GUARD" "$OUT/guard-$name.log" "$secs" "$@" > "$OUT/$name.out" 2>&1
  else timeout "$secs" "$@" > "$OUT/guard-$name.log" 2>&1; fi; local rc=$?
  echo "=== $(date +%H:%M:%S) $name rc=$rc $(grep -hE '^(RESULT|FAIL|PASS token)' "$OUT/guard-$name.log" 2>/dev/null | cut -c1-40 | sort | uniq -c | tr '\n' ' ')" | tee -a "$OUT/run.log"
  if [ $rc -eq 99 ] || [ $rc -eq 98 ]; then echo "WATCHDOG FAULT ($rc) in $name: stopped" | tee -a "$OUT/run.log"; exit $rc; fi
  if [ $rc -eq 6 ]; then echo "$name: a persistent variant failed (rows say fail); going on" | tee -a "$OUT/run.log"; return 0; fi
  if [ $rc -ne 0 ]; then echo "$name failed (rc $rc): stopped" | tee -a "$OUT/run.log"; exit $rc; fi
}
prof() {    # prof NAME: rocprofv3 kernel trace (GPU timestamps) of the command that follows
  echo "$PROF" --kernel-trace --output-format csv -d "$OUT/prof-$1" -o "$1" --
}
D=/sys/bus/pci/devices/$PCI
envinfo() {  # the environment, read under the watchdog (so at its clocks)
  echo "date $(date -Is)"; echo "kernel $(uname -r)"; echo "rocm $ROCM"; "$ROCM/bin/hipcc" --version 2>&1 | head -2
  echo "device $PCI $(cat $D/power_dpm_force_performance_level 2>/dev/null)"
  echo "sclk:"; cat $D/pp_dpm_sclk 2>/dev/null; echo "mclk:"; cat $D/pp_dpm_mclk 2>/dev/null
  echo "runpm $(cat /sys/module/amdgpu/parameters/runpm 2>/dev/null)"
  echo "aspm policy: $(cat /sys/module/pcie_aspm/parameters/policy 2>/dev/null)"
  for x in $(readlink -f $D | tr '/' '\n' | grep -E '^0000:'); do
    echo "link $x: $(cat /sys/bus/pci/devices/$x/current_link_speed 2>/dev/null) x$(cat /sys/bus/pci/devices/$x/current_link_width 2>/dev/null) l1_aspm=$(cat /sys/bus/pci/devices/$x/link/l1_aspm 2>/dev/null || echo -)"
  done
  echo "displays (connector on PCI device):"
  for c in /sys/class/drm/card*-*; do if [ "$(cat $c/status 2>/dev/null)" = connected ]; then echo "  $(basename $c) on $(basename "$(readlink -f $c/device/device)")"; fi; done
  "$BIN" 2>&1 | head -1; md5sum "$BIN" "$HERE/src/decode_floor.hip"; git -C "$HERE" rev-parse HEAD 2>/dev/null; git -C "$HERE" status --short 2>/dev/null | head
}
export -f envinfo; export D PCI ROCM BIN HERE

stage env 60 bash -c envinfo
stage check 600 "$BIN" check "${DEV[@]}"

if [ "$MODE" = smoke ]; then
  stage e0-f16 600 $(prof e0-f16-cold) "$BIN" shapes --fmt f16 --reps 40 "${DEV[@]}" --csv "$OUT/e0.csv"
  stage e0-i4 600 $(prof e0-i4-cold) "$BIN" shapes --fmt i4 --reps 40 "${DEV[@]}" --csv "$OUT/e0.csv"
  stage floor-cold 600 $(prof floor-cold) "$BIN" floor --reps 40 "${DEV[@]}" --csv "$OUT/floor.csv"
  for g in 1 2 0; do stage sync-g$g 600 "$BIN" sync --stages 64 --reps 120 --grid-per-cu $g "${DEV[@]}" --csv "$OUT/sync.csv"; done
  stage e2-i4 600 "$BIN" pshapes --fmt i4 --stages 16 --reps 120 "${DEV[@]}" --csv "$OUT/e2.csv"
  stage e3-i4 600 "$BIN" token --fmt i4 --layers 2 --tokens 2 --reps 100 "${DEV[@]}" --csv "$OUT/e3.csv"
  stage e3-i4-warm 600 "$BIN" token --fmt i4 --warm --layers 8 --tokens 2 --reps 100 "${DEV[@]}" --csv "$OUT/e3.csv"
  stage e3-f16 600 "$BIN" token --fmt f16 --layers 2 --tokens 2 --reps 100 "${DEV[@]}" --csv "$OUT/e3.csv"
  stage floor-sweep 600 $(prof floor-sweep) env FLOOR_GRIDS=256,384,512 FLOOR_MV_GRIDS=384 "$BIN" floor --reps 40 "${DEV[@]}" --csv "$OUT/floor-sweep.csv"
  stage e2-i4-ka 600 env DF_LINK_KEEPALIVE=1 "$BIN" pshapes --fmt i4 --stages 16 --reps 120 "${DEV[@]}" --csv "$OUT/e2-ka.csv"
  stage e3-i4-window-ka 600 env DF_WINDOW_DIAG=1 DF_WINDOW_REPS=10 DF_LINK_KEEPALIVE=1 "$BIN" token --fmt i4 --layers 2 --tokens 2 --reps 100 "${DEV[@]}" --csv "$OUT/e3-window-ka.csv"
else
  # E0: their shapes, one kernel per launch, warm (their condition) and cold (the floor), kernel times from rocprofv3
  for fmt in f16 i4; do
    stage e0-$fmt-warm 900 $(prof e0-$fmt-warm) "$BIN" shapes --fmt $fmt --warm --reps 200 "${DEV[@]}" --csv "$OUT/e0.csv"
    stage e0-$fmt-cold 900 $(prof e0-$fmt-cold) "$BIN" shapes --fmt $fmt --reps 200 "${DEV[@]}" --csv "$OUT/e0.csv"
  done
  # the floor decomposed: empty kernels, and INT4 matvecs at K = 128 and 2048 on 1, 2, 8 blocks, cold and warm
  for c in cold warm; do stage floor-$c 600 $(prof floor-$c) "$BIN" floor $( [ $c = warm ] && echo --warm ) --reps 200 "${DEV[@]}" --csv "$OUT/floor.csv"; done
  # the 384-block anomaly: empty kernels from 256 to 768 blocks (384 blocks of 8 waves = all 3,072 wave slots), and
  # INT4 matvecs next to 384
  stage floor-sweep 600 $(prof floor-sweep) env FLOOR_GRIDS=256,288,320,336,352,368,376,380,382,383,384,385,386,388,392,400,416,448,480,512,576,768 FLOOR_MV_GRIDS=376,384,392 "$BIN" floor --reps 200 "${DEV[@]}" --csv "$OUT/floor-sweep.csv"
  # E1: the N sweep at K=2048, cold
  for fmt in i4 f16; do
    stage e1-$fmt 900 $(prof e1-$fmt) "$BIN" sweep --fmt $fmt --reps 200 "${DEV[@]}" --csv "$OUT/e1.csv"
  done
  # the synchronization alone, against the grid (blocks per WGP; 0 = the occupancy limit)
  for g in 1 2 4 0; do stage sync-g$g 900 "$BIN" sync --stages 256 --reps 200 --grid-per-cu $g "${DEV[@]}" --csv "$OUT/sync.csv"; done
  # E2: each shape as 256 stages: launches, graph, four persistent variants; cold at two grids, and warm
  stage e2-i4 1200 "$BIN" pshapes --fmt i4 --stages 256 --reps 200 "${DEV[@]}" --csv "$OUT/e2.csv"
  stage e2-i4-g2 1200 "$BIN" pshapes --fmt i4 --stages 256 --reps 200 --grid-per-cu 2 "${DEV[@]}" --csv "$OUT/e2.csv"
  stage e2-f16 1200 "$BIN" pshapes --fmt f16 --stages 256 --reps 200 "${DEV[@]}" --csv "$OUT/e2.csv"
  stage e2-i4-warm 1200 "$BIN" pshapes --fmt i4 --warm --stages 256 --reps 200 "${DEV[@]}" --csv "$OUT/e2.csv"
  # the same with the PCIe link kept awake (a host thread reading device memory in a loop): a single launch of a few ms
  # otherwise pays a link wake-up after the kernel that the GPU clocks do not see
  KA="env DF_LINK_KEEPALIVE=1"
  stage e2-i4-ka 1200 $KA "$BIN" pshapes --fmt i4 --stages 256 --reps 200 "${DEV[@]}" --csv "$OUT/e2-ka.csv"
  stage e2-f16-ka 1200 $KA "$BIN" pshapes --fmt f16 --stages 256 --reps 200 "${DEV[@]}" --csv "$OUT/e2-ka.csv"
  stage e2-i4-warm-ka 1200 $KA "$BIN" pshapes --fmt i4 --warm --stages 256 --reps 200 "${DEV[@]}" --csv "$OUT/e2-ka.csv"
  # E3: the 1,393-matvec token (48 layers + lm_head) in every mode, interleaved; INT4 at two grids, warm, f16
  S="env STAGES_CSV=$OUT/stages.csv"
  stage e3-i4 1200 $S "$BIN" token --fmt i4 --tokens 8 --reps 200 "${DEV[@]}" --csv "$OUT/e3.csv"
  stage e3-i4-g2 1200 $S "$BIN" token --fmt i4 --tokens 8 --reps 200 --grid-per-cu 2 "${DEV[@]}" --csv "$OUT/e3.csv"
  stage e3-i4-warm 1200 $S "$BIN" token --fmt i4 --warm --tokens 8 --reps 200 "${DEV[@]}" --csv "$OUT/e3.csv"
  stage e3-f16 1200 $S "$BIN" token --fmt f16 --tokens 8 --reps 200 "${DEV[@]}" --csv "$OUT/e3.csv"
  SK="env STAGES_CSV=$OUT/stages-ka.csv DF_LINK_KEEPALIVE=1"
  stage e3-i4-ka 1200 $SK "$BIN" token --fmt i4 --tokens 8 --reps 200 "${DEV[@]}" --csv "$OUT/e3-ka.csv"
  stage e3-i4-warm-ka 1200 $SK "$BIN" token --fmt i4 --warm --tokens 8 --reps 200 "${DEV[@]}" --csv "$OUT/e3-ka.csv"
  stage e3-f16-ka 1200 $SK "$BIN" token --fmt f16 --tokens 8 --reps 200 "${DEV[@]}" --csv "$OUT/e3-ka.csv"
  # where event time goes outside the kernels: each mode timed by events (host waiting / spinning) and by GPU stamps
  # around the run, with the link left to power-save and kept awake
  stage e3-i4-window 1200 env DF_WINDOW_DIAG=1 DF_WINDOW_REPS=60 "$BIN" token --fmt i4 --tokens 8 --reps 100 "${DEV[@]}" --csv "$OUT/e3-window.csv"
  stage e3-i4-window-ka 1200 env DF_WINDOW_DIAG=1 DF_WINDOW_REPS=60 DF_LINK_KEEPALIVE=1 "$BIN" token --fmt i4 --tokens 8 --reps 100 "${DEV[@]}" --csv "$OUT/e3-window-ka.csv"
  # the plain variants at their own occupancy (the shared grid is capped by the prefetch variants' registers)
  stage e3-i4-occ7 1200 env STAGES_CSV="$OUT/stages-occ.csv" VARIANTS=barrier,dataflow RESIDENCY_PROBE=1 "$BIN" token --fmt i4 --tokens 8 --reps 200 --grid-per-cu 7 "${DEV[@]}" --csv "$OUT/e3-occ.csv"
  stage e3-i4-occ8 1200 env STAGES_CSV="$OUT/stages-occ.csv" VARIANTS=dataflow RESIDENCY_PROBE=1 "$BIN" token --fmt i4 --tokens 8 --reps 200 --grid-per-cu 8 "${DEV[@]}" --csv "$OUT/e3-occ.csv"
  # a test of the expert-down explanation: dataflow with one-tile claims on the short-row stages (K <= 768)
  stage e3-i4-chunk1 1200 env STAGES_CSV="$OUT/stages-chunk1.csv" VARIANTS=dataflow,dataflow-pf DF_SHORTK_CHUNK1=1 "$BIN" token --fmt i4 --tokens 8 --reps 200 "${DEV[@]}" --csv "$OUT/e3-chunk1.csv"
  # GRID4=1: the token at 4 blocks per WGP with the link kept awake (a fixed persistent grid for comparing two builds
  # whose register use gives different occupancy limits; TILE-1RT.md)
  if [ "${GRID4:-0}" = 1 ]; then
    stage e3-i4-ka-g4 1200 env STAGES_CSV="$OUT/stages-g4.csv" DF_LINK_KEEPALIVE=1 "$BIN" token --fmt i4 --tokens 8 --reps 200 --grid-per-cu 4 "${DEV[@]}" --csv "$OUT/e3-g4.csv"
  fi
  # one traced token run: the per-kernel breakdown of the per-op and grouped modes
  stage e3-i4-traced 1200 $(prof e3-i4) "$BIN" token --fmt i4 --tokens 4 --reps 100 "${DEV[@]}"
fi
if python3 -I "$HERE/analyze.py" "$OUT" > "$OUT/analyze.log" 2>&1; then echo "tables: $OUT/tables.md" | tee -a "$OUT/run.log"
else { echo "analyze.py failed (full output in $OUT/analyze.log):"; tail -5 "$OUT/analyze.log"; } | tee -a "$OUT/run.log"; fi
echo "=== $(date +%H:%M:%S) done: $OUT" | tee -a "$OUT/run.log"
