# Development observations behind REPORT.md's link section

Single runs from the session that found the time outside the kernel, before the three final runs. They used
intermediate builds of decode_floor.hip (the window diagnostic as it grew; the same matvec and persistent kernels as
the final build). Each log is the program's output for one token-mode run with `DF_WINDOW_DIAG=1`.

| File | What it shows |
|---|---|
| `xt-2layer-window.log` | RX 7900 XT, 2-layer token: events within ~3 µs of the GPU stamps in every mode |
| `xtx-window-first.log` | XTX, full token: events ~240 µs above the stamps for the persistent variants only; spinning on `hipEventQuery` instead of blocking makes no difference |
| `xtx-window-reset-tests.log` | XTX: the reset of the sync state timed alone (23 µs), a synchronized reset before event a, and an empty kernel between the reset and event a: the extra time stays |
| `xtx-window-host-timing.log` | XTX: host time inside each call (the persistent launch returns in 1.4 µs; the host wait ends ~3,825 µs after submit), and two tokens in one event window (each pays it) |
| `xtx-window-two-monitors.log` | XTX driving two monitors: 65 of 80 persistent windows over 50 µs of hidden time |
| `xtx-window-no-display.log` | XTX with the monitors on the integrated GPU: 79 of 80 and 80 of 80 |
| `xtx-window-keepalive.log` | XTX with `DF_LINK_KEEPALIVE=1`: 0 of 80 for both persistent variants |
| `xtx-window-link-poll.log`, `xtx-root-port-link-poll.txt` | XTX: the window diagnostic with a 1 ms poll of the CPU-side root port's link speed and width (monotonic timestamps in both). Every sample during the run read 16 GT/s x16. A 1 ms poll of one port cannot exclude shorter transitions or changes on the card's downstream links. |

Not archived (development observations, stated in REPORT.md as such): the clock-starvation counts and pass rates
(method finding 1) and the chunk-rounding measurement (method finding 5), both from RX 7900 XT smoke runs.
