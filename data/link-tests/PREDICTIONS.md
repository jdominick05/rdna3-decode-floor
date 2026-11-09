# Root link tests: what each outcome would mean (written before running, 2026-11-09)

Binary: decode_floor from tile-1rt 27ebc9d, base tile (no TILE_1RT); md5 in decode_floor.md5. XTX = HIP device 0,
monitors on the XTX (the display masks the delay in some windows: 65 of 80 in the earlier test), every block under guard.

Each block runs the window diagnostic (INT4 cold token, DF_WINDOW_REPS=60, as in the final runs) and E0 f16 warm shapes
(events). Arms, in order, two rounds: base, [base + keep-alive, round 1 only], l1off-both (l1_aspm=0 on 01:00.0 and
03:00.0), base, l1off-gpu (03:00.0 only), l1off-root (01:00.0 only), base, policy-perf (pcie_aspm policy
"performance"), base, pcie-pin (manual DPM, sclk/mclk/fclk/socclk and pp_dpm_pcie at their top levels, ASPM as found),
base. Nothing reads GPU sysfs or config space during a block (that would itself be bus traffic); snapshots only between.

Primary metric: barrier-pf "events minus stamps" median and windows over 50 us. Controls: grouped-graph hidden (~10 us
in every arm), barrier-pf stamp to stamp (within 1 % of base in every arm, or the arm changed clocks), E0 f16 35B head
event median (base ~1,386 us, kernel ~1,129) and 30B head (base ~717, kernel ~697).

"Removed": median < 30 us and at most 10 % of windows over 50 us. "Unchanged": median > 200 us.

| Outcome | Reading |
|---|---|
| l1off-both and policy-perf remove it, pcie-pin does not | ASPM L1 on the XTX path is necessary for the delay. Not the exact mechanism: L1 entry may also let other parts power down. |
| only l1off-gpu (or only l1off-root) removes it | which link's L1 matters |
| pcie-pin removes it, l1off does not | firmware link DPM |
| none removes it | not ASPM L1 and not link DPM: CPU root complex / I/O die, the GPU's host interface, or the runtime |
| the E0 35B head drops to ~1,130-1,145 us where the token's delay goes | the plain kernel has the same cause |
| a median between 30 and 200 us in an arm | partial: the knob removes part of it or only in some windows; report the per-window distribution, no mechanism claim |
| l1off-both removes it, neither single-link arm does | either link's L1 is enough to produce it (both must be awake) |
| both single-link arms remove it | both links must be in L1 for it to occur |
| E0 35B head (judged only against this script's own base blocks, no profiler) | if the unprofiled base shows no ~+250 us over the kernel's ~1,129 us, the E0 part is uninformative, not "removed" |
| E0 in r1-base-ka | not a keep-alive test: shapes mode has no keep-alive thread |

Arms are read from their logged state lines: a failed sysfs write is only logged, and that block then measures the base
condition under the arm's name.
