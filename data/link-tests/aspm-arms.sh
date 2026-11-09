#!/bin/bash
# Root link tests for the delay after long launches on the XTX. Run as: sudo GUARD=/path/to/watchdog bash aspm-arms.sh
# As run on 2026-11-09, with the user, home directory and watchdog path generalized for publication. X and UP are this
# machine's XTX (03:00.0) and the CPU-side upstream port of its switch (01:00.0): set them for another machine.
# WARNING: the policy-perf arm writes the system-wide ASPM policy; on this machine that wedged an unrelated USB4
# controller until a reboot (README.md here). Remove that arm unless you need it.
# Sets reversible PCIe/DPM knobs as root, runs every measurement as the invoking user under the watchdog, restores everything on exit.
set -u
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
U=${SUDO_USER:?run with sudo from your own account}; H=$(getent passwd "$U" | cut -d: -f6); DIR=$(cd "$(dirname "$0")" && pwd); OUT=$DIR/run-$(date +%H%M%S)
X=/sys/bus/pci/devices/0000:03:00.0; UP=/sys/bus/pci/devices/0000:01:00.0; POL=/sys/module/pcie_aspm/parameters/policy
BIN=$DIR/decode_floor; G=${GUARD:?set GUARD to a watchdog run as GUARD LOG SECS CMD}; STOPF=${GUARD_STOP_FILE:-/nonexistent}
if pgrep -f '^/[^ ]*build/decode_floor' >/dev/null || pgrep -f '^bash run\.sh' >/dev/null; then
    echo "a decode_floor run is still going: wait for it to finish"; exit 1; fi
[ -e "$STOPF" ] && { echo "GPU fault STOP file present: a person must inspect it first"; exit 1; }
(cd "$DIR" && md5sum -c --quiet decode_floor.md5) || { echo "binary changed"; exit 1; }
runuser -u "$U" -- mkdir -p "$OUT"; runuser -u "$U" -- touch "$OUT/arms.log"
log() { echo "$*" | tee -a "$OUT/arms.log"; }
L0_X=$(cat $X/link/l1_aspm); L0_UP=$(cat $UP/link/l1_aspm); P0=$(grep -o '\[[a-z]*\]' $POL | tr -d '[]')
restore() {
    echo 1 > $X/link/l1_aspm 2>/dev/null; echo 1 > $UP/link/l1_aspm 2>/dev/null
    echo "$P0" > $POL 2>/dev/null; echo "$L0_X" > $X/link/l1_aspm; echo "$L0_UP" > $UP/link/l1_aspm
    echo auto > $X/power_dpm_force_performance_level; /usr/local/sbin/gpu-perf auto
}
trap "exit 130" INT TERM
trap 'restore; log "end state after restore:"; snap; log "restored: l1_aspm 03=$(cat $X/link/l1_aspm) 01=$(cat $UP/link/l1_aspm) policy=$(cat $POL) level=$(cat $X/power_dpm_force_performance_level)"' EXIT
snap() {  # between blocks only; also every other link's LnkCtl, to check the policy round trip restores them
    for d in $(ls /sys/bus/pci/devices | sed 's/^0000://'); do echo "$d $(lspci -vvv -s $d 2>/dev/null | grep -E 'LnkCtl:' | tr -s ' \t' ' ')"; done | grep LnkCtl >> "$OUT/lnkctl-all.log"; echo "--" >> "$OUT/lnkctl-all.log"
    log "  state: l1_aspm 01=$(cat $UP/link/l1_aspm) 03=$(cat $X/link/l1_aspm) policy='$(cat $POL)' level=$(cat $X/power_dpm_force_performance_level) power=$(cat $X/power/control)"
    for f in pcie sclk mclk fclk socclk; do log "  pp_dpm_$f: $(grep '\*' $X/pp_dpm_$f | tr -s ' ')"; done
    for d in 00:01.1 01:00.0 02:00.0 03:00.0; do log "  $d $(lspci -vvv -s $d 2>/dev/null | grep -E 'LnkCtl:' | tr -s ' \t' ' ' | cut -c1-60)"; done
}
run_as() { runuser -u "$U" -- env HOME=$H XDG_RUNTIME_DIR=/run/user/$(id -u "$U") HIP_VISIBLE_DEVICES=0 "$@"; }
block() {  # block NAME [extra env...]
    local name=$1; shift; snap
    for part in window e0; do
        if [ $part = window ]; then cmd=(env "$@" DF_WINDOW_DIAG=1 DF_WINDOW_REPS=60 $BIN token --fmt i4 --tokens 8 --reps 100 --expect-cus 48)
        else cmd=(env "$@" $BIN shapes --fmt f16 --warm --reps 200 --expect-cus 48 --csv $OUT/$name-e0.csv); fi
        run_as ${GH_ENV:-} $G $OUT/$name-$part.log 300 "${cmd[@]}" > /dev/null 2>&1; rc=$?
        log "$(date +%T) $name $part rc=$rc"
        if [ $rc = 99 ] || [ $rc = 98 ] || [ -e "$STOPF" ]; then log "GUARD STOP: stopping all arms"; exit 1; fi
    done
}
setl1() { echo $1 > $UP/link/l1_aspm || log "  write 01 l1_aspm=$1 FAILED"; echo $2 > $X/link/l1_aspm || log "  write 03 l1_aspm=$2 FAILED"; }
log "start $(date) out=$OUT initial: l1_aspm 03=$L0_X 01=$L0_UP policy=$P0"
for round in 1 2; do
    log "== round $round"
    block r$round-base
    [ $round = 1 ] && block r$round-base-ka DF_LINK_KEEPALIVE=1
    setl1 0 0; block r$round-l1off-both; setl1 1 1
    block r$round-base2
    setl1 1 0; block r$round-l1off-gpu; setl1 1 1
    setl1 0 1; block r$round-l1off-root; setl1 1 1
    block r$round-base3
    echo performance > $POL || log "  write policy=performance FAILED"; block r$round-policy-perf; echo "$P0" > $POL
    block r$round-base4
    # pcie-pin: manual DPM with every clock at its top level (= the runs' "high") and the link held at its top level;
    # guard runs the command without gpu-high so nothing resets the level
    echo on > $X/power/control; echo auto > $X/power_dpm_force_performance_level; echo manual > $X/power_dpm_force_performance_level
    for f in sclk mclk fclk socclk pcie; do echo "$(awk -F: 'END {print $1}' $X/pp_dpm_$f)" > $X/pp_dpm_$f || log "  pin $f FAILED"; done
    GH_ENV="CAMINUS_GPU_HIGH=/usr/bin/env" block r$round-pcie-pin
    echo auto > $X/power_dpm_force_performance_level
    block r$round-base5
done
log "done $(date)"
