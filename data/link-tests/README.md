# Root link tests: what removes the delay after long launches

Run 2026-11-09, 14:05–14:14, on the RX 7900 XTX with both monitors attached to it, two rounds. Written down before
running: [PREDICTIONS.md](PREDICTIONS.md) (what each outcome would mean). Script: [aspm-arms.sh](aspm-arms.sh), as run,
with the user, home directory and watchdog path generalized. Logs: [run-140555/](run-140555/). Table:
`python3 parse.py run-140555`.

- **Binary.** `decode_floor` built from a development branch without `-DTILE_1RT`: the same kernels as
  `src/decode_floor.hip` on main, plus a debug output switch that is off unless set. md5 in `decode_floor.md5`.
- **Each block:** the window diagnostic on the INT4 cold token (60 windows per mode, as in the final runs) and E0 f16
  warm shapes (HIP events, no profiler).
- **Arms:** base; base with the keep-alive (round 1 only); ASPM L1 off on both XTX links (`l1_aspm=0` on 01:00.0, the
  CPU-side link to the card's switch, and on 03:00.0, the switch-to-GPU link); base; L1 off on 03:00.0 only; L1 off on
  01:00.0 only; base; the system ASPM policy set to `performance`; base; manual DPM with sclk, mclk, fclk, socclk and
  the PCIe link (`pp_dpm_pcie`) at their top levels, ASPM as found; base.
- **No reads during a block.** The script reads GPU sysfs and config space only between blocks, since a read is itself
  bus traffic. The "level" column is the DPM level between blocks. The watchdog sets DPM `high` for each run, except in
  pcie-pin, where the script holds manual top levels.
- **The links on these devices offer only ASPM L1.** There are no L0s or L1-substate controls, and clock PM is off.

## Results (hidden = events minus stamps, median of 60 windows; in brackets, windows over 50 µs)

| arm | l1 01/03, policy, level, pcie | barrier-pf hidden med (over 50) | dataflow-pf | barrier | dataflow | grouped-graph | empty | b-pf stamp-to-stamp | E0 35B head ev | 30B head ev |
|---|---|---|---|---|---|---|---|---|---|---|
| r1-base | 1/1, default, auto | 243.3 (45/60) | 243.7 (55/60) | 243.1 (47/60) | 242.6 (51/60) | 10.7 (0/60) | 8.7 (0/60) | 3579.6 | 1383.4 | 715.5 |
| r1-base-ka | 1/1, default, auto | 9.4 (0/60) | 9.3 (0/60) | 9.4 (0/60) | 9.3 (0/60) | 10.7 (0/60) | 8.7 (0/60) | 3578.4 | 1383.1 | 715.2 |
| r1-l1off-both | 0/0, default, auto | 9.3 (0/60) | 9.2 (0/60) | 9.4 (0/60) | 9.2 (0/60) | 10.7 (0/60) | 8.8 (0/60) | 3583.4 | 1148.4 | 714.7 |
| r1-base2 | 1/1, default, auto | 243.3 (48/60) | 242.3 (44/60) | 243.5 (53/60) | 242.4 (52/60) | 10.7 (0/60) | 8.6 (0/60) | 3586.4 | 1381.3 | 715.5 |
| r1-l1off-gpu | 1/0, default, auto | 9.3 (0/60) | 9.3 (0/60) | 9.4 (0/60) | 9.2 (0/60) | 10.6 (0/60) | 8.8 (0/60) | 3596.0 | 1148.2 | 715.1 |
| r1-l1off-root | 0/1, default, auto | 9.5 (0/60) | 9.4 (0/60) | 9.6 (0/60) | 9.4 (0/60) | 10.7 (0/60) | 8.6 (0/60) | 3596.7 | 1148.2 | 715.5 |
| r1-base3 | 1/1, default, auto | 243.1 (53/60) | 243.1 (50/60) | 243.6 (58/60) | 242.4 (53/60) | 10.3 (0/60) | 8.8 (0/60) | 3579.0 | 1381.3 | 715.3 |
| r1-policy-perf | 0/0, performance, auto | 9.2 (0/60) | 9.2 (0/60) | 9.4 (0/60) | 9.2 (0/60) | 10.4 (0/60) | 8.8 (0/60) | 3579.0 | 1148.1 | 715.0 |
| r1-base4 | 1/1, default, auto | 242.0 (53/60) | 242.1 (53/60) | 243.2 (54/60) | 242.0 (51/60) | 10.4 (0/60) | 8.8 (0/60) | 3575.6 | 1382.7 | 715.2 |
| r1-pcie-pin | 1/1, default, manual | 241.6 (44/60) | 241.8 (50/60) | 242.8 (56/60) | 242.0 (53/60) | 10.7 (0/60) | 8.8 (0/60) | 3576.3 | 1382.4 | 715.2 |
| r1-base5 | 1/1, default, auto | 242.0 (50/60) | 241.9 (51/60) | 242.5 (57/60) | 242.1 (54/60) | 10.4 (0/60) | 8.6 (0/60) | 3579.0 | 1381.7 | 715.5 |
| r2-base | 1/1, default, auto | 242.7 (51/60) | 242.5 (50/60) | 242.2 (54/60) | 242.5 (56/60) | 10.5 (0/60) | 8.8 (0/60) | 3584.8 | 1381.9 | 715.3 |
| r2-l1off-both | 0/0, default, auto | 9.2 (0/60) | 9.2 (0/60) | 9.5 (0/60) | 9.2 (0/60) | 10.4 (0/60) | 8.8 (0/60) | 3592.4 | 1148.2 | 715.2 |
| r2-base2 | 1/1, default, auto | 242.4 (51/60) | 242.8 (51/60) | 243.1 (55/60) | 242.5 (49/60) | 10.6 (0/60) | 8.8 (0/60) | 3579.2 | 1381.7 | 715.7 |
| r2-l1off-gpu | 1/0, default, auto | 9.0 (0/60) | 9.2 (0/60) | 9.2 (0/60) | 9.0 (0/60) | 10.3 (0/60) | 8.7 (0/60) | 3590.2 | 1148.2 | 715.3 |
| r2-l1off-root | 0/1, default, auto | 9.0 (0/60) | 9.1 (0/60) | 9.2 (0/60) | 9.0 (0/60) | 10.4 (0/60) | 8.7 (0/60) | 3591.8 | 1147.9 | 715.2 |
| r2-base3 | 1/1, default, auto | 242.5 (49/60) | 242.2 (48/60) | 242.6 (52/60) | 242.2 (52/60) | 10.7 (0/60) | 8.8 (0/60) | 3584.0 | 1380.9 | 715.4 |
| r2-policy-perf | 0/0, performance, auto | 9.0 (0/60) | 9.1 (0/60) | 9.3 (0/60) | 9.0 (0/60) | 10.5 (0/60) | 8.7 (0/60) | 3585.2 | 1148.0 | 715.2 |
| r2-base4 | 1/1, default, auto | 241.2 (52/60) | 241.4 (52/60) | 241.8 (50/60) | 241.5 (51/60) | 10.7 (0/60) | 8.8 (0/60) | 3584.3 | 1381.1 | 715.3 |
| r2-pcie-pin | 1/1, default, manual | 240.8 (50/60) | 242.0 (52/60) | 241.6 (50/60) | 241.8 (55/60) | 10.7 (0/60) | 8.7 (0/60) | 3580.0 | 1381.1 | 715.1 |
| r2-base5 | 1/1, default, auto | 241.9 (49/60) | 242.2 (51/60) | 242.2 (50/60) | 242.2 (58/60) | 10.7 (0/60) | 8.8 (0/60) | 3584.6 | 1380.3 | 715.1 |

## Reading

- **ASPM L1 on the XTX's path is necessary for the delay.** With L1 off on either link alone, on both, or
  system-wide, the persistent kernels' hidden time drops from ~242 µs to ~9 µs, in 0 of 60 windows, in both rounds.
  That is the same level as the keep-alive arm and the empty kernel.
- **Link DPM is not the cause.** Holding the PCIe link at its top DPM level, with every clock at its top level, leaves
  the delay unchanged.
- **The plain kernel has the same cause.** The E0 f16 35B head reads 1,380–1,383 µs by events in every base block, and
  1,148 µs wherever L1 is off (its kernel time is ~1,129 µs).
- **The smaller part is not L1.** The 30B head (~715 µs by events, kernel ~697 µs) is unchanged in every arm.
- **The controls held.**
  - The grouped graph's hidden time stayed ~10 µs in every arm.
  - barrier-pf's stamp-to-stamp time stayed within 0.6 % of the base blocks, so no arm changed the GPU clocks.
  - The E0 rows in base-ka are not a keep-alive test: shapes mode has no keep-alive thread.
- **Which link?** Turning off either link's L1 is enough. A switch's upstream link may enter L1 only when its downstream
  links are in L1, so L1 off on the GPU link also keeps the CPU-side link out of L1. The data fit an exit from L1 on
  the CPU-side link, but nothing here measures which link entered L1 (no residency counters).
- **~240 µs is far longer than a normal L1 exit** (microseconds to tens of µs). So L1 is a necessary condition. It may
  enable something deeper, such as further power-down in the CPU's root complex; that was not measured.

## A side effect worth knowing before reproducing

The policy-perf arm writes the system-wide ASPM policy (`/sys/module/pcie_aspm/parameters/policy`). On this machine,
the first write wedged the motherboard's ASMedia ASM4242 USB4 controller (its xHCI and host-router functions):
- **From 14:08:33:** the kernel logged "Refused to change power state from D0 to D3hot" and config-space timeouts.
- **Its USB devices dropped,** a hub and a webcam.
- **Restoring the policy to `default` did not recover it,** and neither did a PCI remove and rescan. A reboot did.
- **Every other link** ended in its starting state (`run-140555/lnkctl-all.log`).

Use the per-link `l1_aspm` controls on the GPU's path only, unless you need the global policy.
