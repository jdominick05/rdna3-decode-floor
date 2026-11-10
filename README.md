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
- On this XTX, single launches longer than ~1 ms were followed by ~250 µs of idle GPU: a gap before the next dispatch
  in a kernel trace, excluded from kernel durations. It needs ASPM L1 on the card's PCIe path: turning L1 off on either
  of its links removed it (root tests, [data/link-tests/](data/link-tests/)), as did a host thread reading GPU memory
  over PCIe.

## The whole model

This study times one token's matvecs in isolation. The follow-up serves the whole model on the same card, from the
GGUF file to output tokens: [Qwen3-30B-A3B on Caminus's own code](https://github.com/jdominick05/caminus/blob/testing/docs/dev/qwen3-30b-a3b.md).

- One persistent kernel per token, with tiles that issue all their loads before using any (what this study pointed
  to): 246 tok/s at short context and 106 tok/s at the full 40,960-token context, 55 % to 74 % of a bound set by the
  bytes a token must read at the 858 GB/s measured here.
- Where the rest of a token goes: about half is what its 432 dependent stages cost before they load or compute
  anything (task claims, barriers, result stores, publishes). The hand-off between stages is 0.1 ms of it.
- The link delay this study found (ASPM L1 on the card's path) cost the whole model 0.2 ms on every token. One tape
  segment that touches host-mapped memory keeps the link awake, with no system setting.

The code lives in [Caminus](https://github.com/jdominick05/caminus) (AGPL-3.0); none of it is copied here.

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
