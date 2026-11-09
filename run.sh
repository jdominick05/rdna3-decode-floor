#!/bin/bash
# SPDX-License-Identifier: MIT
# run.sh smoke|xtx: the GPU runs of PLAN.md, each under a GPU watchdog (GUARD, default the Caminus guard.sh: VRAM lines,
# kernel-log fault watch, fixed high clocks). A watchdog fault (exit 99 / 98) or a failed stage stops the script; a stage
# that ends with 6 (a persistent variant failed its wait or dependency check; its rows say "fail") is recorded and the
# script goes on.
#   smoke  the RX 7900 XT alone (device 1, 84 CUs = 42 WGPs; HIP reports WGPs on RDNA3), small sizes: nothing reported
#   xtx    the RX 7900 XTX (device 0, 96 CUs = 48 WGPs): E0, the floor split, E1, sync, E2, E3 (+ occupancy, one-tile claims)
# Results: results/<date>-<mode>/ (CSV files with headers, rocprofv3 kernel traces, logs, environment, tables.md).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
MODE=${1:?usage: run.sh smoke|xtx}
BIN=$HERE/build/decode_floor
GUARD=${GUARD:-$HOME/src/caminus-side/tools/guard.sh}
ROCM=${ROCM:-$HOME/rocm-therock/10.2.0a20261011}
PROF=$ROCM/bin/rocprofv3
OUT=$HERE/results/$(date +%Y%m%d-%H%M%S)-$MODE
mkdir -p "$OUT"
[ -x "$BIN" ] || { echo "build first: ./build.sh"; exit 2; }
[ -x "$GUARD" ] || { echo "no watchdog at $GUARD (set GUARD)"; exit 2; }

case $MODE in
  smoke) export HIP_VISIBLE_DEVICES=1; DEV=(--expect-cus 42); PCI=0000:06:00.0 ;;
  xtx)   export HIP_VISIBLE_DEVICES=0; DEV=(--expect-cus 48); PCI=0000:03:00.0 ;;
  *) echo "usage: run.sh smoke|xtx"; exit 2 ;;
esac

stage() {   # stage NAME SECS CMD...
  local name=$1 secs=$2; shift 2
  echo "=== $(date +%H:%M:%S) $name" | tee -a "$OUT/run.log"
  "$GUARD" "$OUT/guard-$name.log" "$secs" "$@" > "$OUT/$name.out" 2>&1; local rc=$?
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
else
  # E0: their shapes, one kernel per launch, warm (their condition) and cold (the floor), kernel times from rocprofv3
  for fmt in f16 i4; do
    stage e0-$fmt-warm 900 $(prof e0-$fmt-warm) "$BIN" shapes --fmt $fmt --warm --reps 200 "${DEV[@]}" --csv "$OUT/e0.csv"
    stage e0-$fmt-cold 900 $(prof e0-$fmt-cold) "$BIN" shapes --fmt $fmt --reps 200 "${DEV[@]}" --csv "$OUT/e0.csv"
  done
  # the floor decomposed: empty kernels, and INT4 matvecs at K = 128 and 2048 on 1, 2, 8 blocks, cold and warm
  for c in cold warm; do stage floor-$c 600 $(prof floor-$c) "$BIN" floor $( [ $c = warm ] && echo --warm ) --reps 200 "${DEV[@]}" --csv "$OUT/floor.csv"; done
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
  # E3: the 1,393-matvec token (48 layers + lm_head) in every mode, interleaved; INT4 at two grids, warm, f16
  S="env STAGES_CSV=$OUT/stages.csv"
  stage e3-i4 1200 $S "$BIN" token --fmt i4 --tokens 8 --reps 200 "${DEV[@]}" --csv "$OUT/e3.csv"
  stage e3-i4-g2 1200 $S "$BIN" token --fmt i4 --tokens 8 --reps 200 --grid-per-cu 2 "${DEV[@]}" --csv "$OUT/e3.csv"
  stage e3-i4-warm 1200 $S "$BIN" token --fmt i4 --warm --tokens 8 --reps 200 "${DEV[@]}" --csv "$OUT/e3.csv"
  stage e3-f16 1200 $S "$BIN" token --fmt f16 --tokens 8 --reps 200 "${DEV[@]}" --csv "$OUT/e3.csv"
  # the plain variants at their own occupancy (the shared grid is capped by the prefetch variants' registers)
  stage e3-i4-occ7 1200 env STAGES_CSV="$OUT/stages-occ.csv" VARIANTS=barrier,dataflow RESIDENCY_PROBE=1 "$BIN" token --fmt i4 --tokens 8 --reps 200 --grid-per-cu 7 "${DEV[@]}" --csv "$OUT/e3-occ.csv"
  stage e3-i4-occ8 1200 env STAGES_CSV="$OUT/stages-occ.csv" VARIANTS=dataflow RESIDENCY_PROBE=1 "$BIN" token --fmt i4 --tokens 8 --reps 200 --grid-per-cu 8 "${DEV[@]}" --csv "$OUT/e3-occ.csv"
  # a test of the expert-down explanation: dataflow with one-tile claims on the short-row stages (K <= 768)
  stage e3-i4-chunk1 1200 env STAGES_CSV="$OUT/stages-chunk1.csv" VARIANTS=dataflow,dataflow-pf DF_SHORTK_CHUNK1=1 "$BIN" token --fmt i4 --tokens 8 --reps 200 "${DEV[@]}" --csv "$OUT/e3-chunk1.csv"
  # one traced token run: the per-kernel breakdown of the per-op and grouped modes
  stage e3-i4-traced 1200 $(prof e3-i4) "$BIN" token --fmt i4 --tokens 4 --reps 100 "${DEV[@]}"
fi
python3 -I "$HERE/analyze.py" "$OUT" > /dev/null 2>&1 && echo "tables: $OUT/tables.md" | tee -a "$OUT/run.log"
echo "=== $(date +%H:%M:%S) done: $OUT" | tee -a "$OUT/run.log"
