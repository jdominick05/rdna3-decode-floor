# rdna3-decode-floor

What the matrix-vector products of one Qwen3-30B-A3B decode token cost on an AMD RX 7900 XTX (gfx1100) when they are
launched four ways:
- one launch per matvec;
- one launch per dependent stage;
- those launches as a HIP graph;
- one persistent kernel per token, in four variants.

Single kernels run cold and warm in INT4 and f16; the per-shape and token runs cover INT4 cold, INT4 warm and f16
cold. The same tile kernel runs in every mode. It was written for the discussion
in [iree-org/iree#24165](https://github.com/iree-org/iree/issues/24165) and
[#23507](https://github.com/iree-org/iree/issues/23507).

- [REPORT.md](REPORT.md): the results, including where the plan's own stop rule tripped, and what changed after an
  independent review.
- [PLAN.md](PLAN.md): the experiments and predictions, written before measuring.
- [data/](data/): three full runs (CSV, logs, environment) and their spread.

## In three lines

- Grouping the MoE experts into one launch per stage takes the matvec token from 9.0 to 3.7 ms. After that, the gaps
  between launches are a minority of it (two model-based estimates: 0.23–0.35 ms), and about 1.4 ms of kernel time
  sits above a bandwidth baseline.
- A persistent kernel replaces each launch gap with a barrier of about the same cost. Here its best variant was 4 %
  slower than the graph with default settings, and 2 % faster with the PCIe link kept awake.
- On this XTX, single launches longer than ~1 ms showed ~250 µs more time by events and on the host than by GPU
  clocks. A host thread reading GPU memory over PCIe removed it; link power management is the leading hypothesis, not
  established.

## Build and run

Needs a ROCm install with `hipcc` and `rocprofv3`, a 7900 XTX with most of its 24 GB free, and Python 3 for the
tables.

```
ROCM=/opt/rocm ./build.sh
./run.sh smoke        # small sizes on a second card first (an XTX alone: DEVICE=0 EXPECT_CUS=48 ./run.sh smoke)
./run.sh xtx          # the measurements: results/<date>-xtx/tables.md
```

- **The watchdog is optional.** `GUARD=/path/to/watchdog` runs every stage under it. Without one, stages run under
  `timeout` and the clocks are not pinned; pin them (e.g. `power_dpm_force_performance_level` = `high`) for comparable
  numbers.
- **The binary runs single experiments directly.** Run `build/decode_floor` without arguments to list the modes and
  environment switches.

MIT licence.
