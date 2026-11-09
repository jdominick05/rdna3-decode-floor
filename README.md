# rdna3-decode-floor

What the matrix-vector products of one Qwen3-30B-A3B decode token cost on an AMD RX 7900 XTX (gfx1100) when they are
launched four ways:
- one launch per matvec;
- one launch per dependent stage;
- those launches as a HIP graph;
- one persistent kernel per token, in four variants.

Each runs cold and warm, in INT4 and f16, with the same tile kernel in every mode. It was written for the discussion
in [iree-org/iree#24165](https://github.com/iree-org/iree/issues/24165) and
[#23507](https://github.com/iree-org/iree/issues/23507).

- [REPORT.md](REPORT.md): the results, including where the plan's own stop rule tripped.
- [PLAN.md](PLAN.md): the experiments and predictions, written before measuring.
- [data/](data/): three full runs (CSV, logs, environment) and their spread.

## In three lines

- Grouping the MoE experts into one launch per stage takes the matvec token from 9.0 to 3.7 ms. After that, launch
  gaps are 6–10 % of it, and most of the rest is dependent memory latency inside small kernels.
- A persistent kernel replaces each launch gap with a barrier of about the same cost. Here its best variant was 4 %
  slower than the graph with default settings, and 2 % faster with the PCIe link kept awake.
- On this XTX, any single kernel longer than ~1 ms pays ~250 µs of PCIe link wake-up after it ends. No GPU clock sees
  it, but the host does.

## Build and run

Needs a ROCm install with `hipcc` and `rocprofv3`, a 7900 XTX with most of its 24 GB free, and Python 3 for the
tables.

```
ROCM=/opt/rocm ./build.sh
./run.sh smoke        # small sizes on a second card first (DEVICE=n picks the HIP device; run.sh documents the rest)
./run.sh xtx          # the measurements: results/<date>-xtx/tables.md
```

- **The watchdog is optional.** `GUARD=/path/to/watchdog` runs every stage under it. Without one, stages run under
  `timeout` and the clocks are not pinned; pin them (e.g. `power_dpm_force_performance_level` = `high`) for comparable
  numbers.
- **The binary runs single experiments directly.** Run `build/decode_floor` without arguments to list the modes and
  environment switches.

MIT licence.
