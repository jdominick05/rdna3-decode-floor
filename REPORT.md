# Where a Qwen3-30B-A3B decode token's matvec time goes on an RX 7900 XTX

Measured 2026-11-09 on one RX 7900 XTX (gfx1100, 96 CUs / 48 WGPs, 960 GB/s GDDR6, 96 MB Infinity Cache).
- **Plan:** [PLAN.md](PLAN.md), with the predictions written before measuring. Its last section was revised after an
  RX 7900 XT smoke run, before any XTX measurement.
- **Data:** three full runs of the same build, in [data/](data/). [data/spread.md](data/spread.md) gives, for each
  quantity, the median of the three runs' values and their range. The numbers below are those medians.
- **Revised after an independent review.** The section "Revisions" at the end lists what changed since the first
  version (tag `report-2026-11-09`).

## The answer in short

1. **The "token" here is 1,393 matrix-vector products only:** 48 layers plus the vocab head, in 241 dependent stages.
   - Attention, norms, routing and sampling are not run.
   - The stages are synchronized but pass no data: every matvec reads a fixed activation vector, not the previous
     stage's output.
   - Compare the modes with each other, never with whole-token speeds such as IREE's 120 tok/s.
2. **Grouping the experts is the big lever.**

   | Mode | INT4 cold token |
   |---|---|
   | One launch per matvec | 9.02 ms |
   | One launch per dependent stage | 3.82 ms |
   | The same stages as a HIP graph | 3.68 ms |

3. **With experts grouped and launches graphed, the gaps between launches are a minority of the token.** Two
   model-based estimates put them at 0.23–0.35 ms of the graph's 3.66 ms of GPU time. Neither is a measured bound (see
   "Where the token's GPU time goes").

   | Part of the graph's 3.66 ms of GPU time | ms |
   |---|---|
   | The weight bytes, at the 858 GB/s the vocab head reaches | 1.88 |
   | Kernel time above that baseline | 1.43 |
   | Gaps between kernels (estimate) | 0.23–0.35 |

   - **What the 1.43 ms is made of is not measured.** It is time above a bandwidth baseline. It includes arithmetic,
     dequantization, reductions, metadata loads and dispatch, as well as memory latency.
   - **Serial loads are a likely contributor.** The generated code waits for each chunk's loads before issuing the
     next. A wave's two rows also run one after the other. An INT4 tile at K = 2048 therefore takes 4 dependent round
     trips, and an f16 tile 16.
4. **The persistent kernel's result depends on whether something keeps the PCIe link busy.**

   | INT4 cold token | As configured | Host thread reading GPU memory over PCIe |
   |---|---|---|
   | Grouped graph | 3.68 ms | 3.67 ms |
   | Best persistent variant (barrier-pf) | 3.83 ms, **4 % slower** | 3.60 ms, **2 % faster** |

   - **What was observed.** On this XTX, every tested single launch longer than about 1 ms showed 200–320 µs more time
     by HIP events (and on the host) than by GPU clocks. That covers the four persistent variants and one plain
     1.13 ms kernel.
   - **The keep-alive removed it in every persistent test.** The thread reads one word of GPU memory over PCIe in a
     loop.
   - **Launch sequences didn't show it.**
   - **It was paid on every launch, back-to-back launches too.** So a decode loop that runs one long kernel per token
     would pay it on every token.
   - **The leading hypothesis is PCIe link power management, but it is not established.**
   - **On GPU time, the persistent kernel recovers the gaps but spends more inside its stages.** The largest
     interval differences are in the experts' down projections (+0.52 ms) and in q|k|v and o (−0.31 ms, where the
     next stage's loads start early). These are differences between synchronization boundaries, not isolated costs.
   - **This is far from the pre-registered bar.** The refutation test asked for at least 1.25× faster than the graph.
     This persistent kernel did not deliver the gain.
5. **The plan's E0 stop rule tripped, and we kept going.** On large shapes our kernels are 2–5× faster than IREE's, so
   the results describe this implementation, not IREE's kernels. The next section says what that means.
6. **These are one machine's numbers.** On the RX 7900 XT (one run), the persistent kernel was 27 % slower than the
   graph even on GPU time. See Limitations.

## Deviation from the plan: the E0 stop rule tripped

**The rule.** PLAN.md's E0 gate reads: "If they come within about 2× of IREE's per-shape times, the comparison is on
comparable kernels. If not, we stop and report that first. A kernel far faster than theirs is a codegen result, not a
dispatch result."

**What happened.**
- Of IREE's 26 per-shape kernel times, six are clearly more than 2× ours:
  - f16 8192×2048 (4.8×), q 4096×2048 (4.3×), o 2048×4096 (3.5×), down 2048×768 (2.5×) and the 35B vocab head
    (2.3×);
  - INT4 o (2.3×).
- Two more are close to 2×: the f16 30B head (2.03×) and INT4 8192×2048 (1.98×).
- In every case ours is the faster kernel.
- run.sh runs every experiment in one pass, and nothing in it enforced the gate. We compared E0 against IREE's numbers
  after the full run, so nothing stopped.

**Why we continued.**
- **The question is the launch floor.** That lives mostly in the small shapes, and most of those are within 2×: INT4
  1.1–1.8× (one shape where IREE is faster, 2048×512 at 0.8×) and f16 1.0–1.9× except the down projection (2.5×).
- **The large-shape difference is codegen.** iree#23507 reports llama.cpp's f16 kernels 1.8–2.7× faster than IREE's on
  the large shapes, and ours are 1.1–2× faster again.

**What it means: a restricted interpretation.**
- **Our token times are not IREE's token times.** With IREE's kernels the large stages would take longer, and the launch
  share of the token would be smaller.
- **The results hold for this implementation, on this platform, with this synthetic schedule.** Similar standalone
  kernel times do not imply similar occupancy, dependencies, cache behaviour or persistence costs, and only one tile
  implementation was tested. The stages also pass no data, which a real decode step does.
- **The rule should have been in the script.** Recorded as a method finding below.

## Setup

- **Card and software:**
  - RX 7900 XTX alone (`HIP_VISIBLE_DEVICES=0`), Linux 7.0.0-38, TheRock ROCm 10.2.0a20261011 (HIP 7.17, AMD clang 24);
  - DPM forced high (SCLK top state 2,482 MHz, MCLK 1,249 MHz), amdgpu runtime PM off;
  - every process ran under a GPU watchdog (VRAM limit, kernel-log fault watch).
- **Display:**
  - No display on the XTX during the three runs (the monitors were on the integrated GPU), with one exception.
  - The monitors were moved back during run 3's last four stages: window with keep-alive, occupancy, one-tile claims,
    and traced.
  - Those stages' event-timed results match runs 1 and 2 within 0.3 %. In the traced stage, the grouped gate|up kernel
    mean is 1.2 % higher (27.01 against 26.69 µs) and the grouped kernel sum 0.3 % higher.
- **PCIe:**
  - The XTX is on CPU lanes at 16 GT/s x16, behind the card's own switch.
  - ASPM L1 is enabled on the card's links (sysfs `l1_aspm=1` on 01:00.0 and 03:00.0); the kernel's ASPM policy is
    `default`.
  - Each run's `environment.txt` records the link state at its start.
- **Model shapes:**
  - Qwen3-30B-A3B per layer: q 1×4096×2048, k and v 1×512×2048, o 1×2048×4096, router 1×128×2048, and 8 of 128 experts,
    each gate and up 1×768×2048 and down 1×2048×768;
  - vocab head 1×151936×2048;
  - random weights in the real shapes (timing does not depend on values);
  - the routed experts are drawn at random per token (8 routing samples).
- **Formats:**
  - f16 weights;
  - INT4 with groups of 128 and an f16 scale and zero point per group;
  - 1.616 GB per INT4 token, 6.083 GB per f16 token (f16 allocates only the 8 experts a token uses, since 128 would not
    fit);
  - E2 and E3 run INT4 cold, INT4 warm and f16 cold (no warm f16).
- **One matvec tile function in every mode:**
  - 16 rows per 256-thread block (8 waves, 2 rows per wave, one row at a time);
  - f32 accumulation with `v_dot2_f32_f16`;
  - INT4 nibbles become f16 via the 0x6400 exponent trick;
  - a lane reads 32 INT4 columns or 8 f16 columns per iteration.
- **Modes:**
  - per-op: one launch per matvec;
  - grouped: one launch per stage (q|k|v together, the 8 experts' gate|up together, their downs together);
  - graph: the same launches captured in a HIP graph;
  - four persistent variants, one launch per token:
    - *barrier*: static round-robin tiles, a split-phase device-wide barrier between stages;
    - *dataflow*: per-stage claim and done counters; each block claims at most one chunk of ⌈tiles / blocks⌉ tiles
      per stage and waits only for the stage before it;
    - *-pf*: prefetch the claimed tile's weights before the wait (not for the routed gate|up, whose experts the router
      picks).
- **Synthetic dependencies.** Each persistent stage waits for the stage before it (barrier or done counters), and a
  dependency check confirms the wait. But the stages pass no data: every matvec reads a fixed activation vector.
- **Cold and warm:**
  - cold: weight copies totalling at least 400 MB, walked in order, so a copy is re-read only after 400+ MB of other
    traffic;
  - warm: one copy reused;
  - warm token: every layer reuses one layer's weights and 8 experts (30.2 MB), no vocab head.
- **Timing:**
  - single kernels (E0, E1, floor): rocprofv3 `--kernel-trace` GPU timestamps, the same tool as iree#23507;
  - E3 (token): HIP events around each token; the modes interleaved, their order rotated each round; median of 10
    measured rounds after 2 warmup rounds;
  - E2 (one shape, 256 times): HIP events around each sequence; 5 measured rounds after 1 warmup round, always in the
    order launches, graph, barrier, barrier-pf, dataflow, dataflow-pf;
  - persistent stages: the GPU wall clock (100 MHz) at stage boundaries;
  - token windows (the time-outside-the-kernel diagnostic): also a one-thread "stamp" kernel just before and just
    after the run (GPU clock, no events);
  - E2 and E3 each run twice: as configured, and with a host thread reading one word of device memory over PCIe in a
    loop (`DF_LINK_KEEPALIVE=1`).

## Correctness

- **Every kernel against a naive reference.** 26 shape/format pairs, max error 1.63e-4 (f16) and 2.92e-4 (INT4). The
  error is stabilized: |got − ref| / (|ref| + 0.01).
- **The reference is checked against double precision.** A double-precision host computation of rows 0 and N − 1 of
  each shape. Non-finite values fail.
- **Every persistent variant's token outputs are bitwise equal to the per-op outputs.** That holds in the cold, warm
  and f16 tokens, at every grid used, with and without the keep-alive.
- **A dependency check ran in the same runs.** Every producer adds its tiles' ids to a per-stage sum before it
  publishes, and every consumer checks the complete sum after its wait. No failure.
- **No wait timed out in any reported run.**

## Results

### E0: single kernels, their shapes (rocprofv3 kernel time, µs)

iree#23507's numbers were taken warm ("Both benchmarks run thousands of iterations, so L2/Infinity Cache is warm"), so
they compare with our warm columns. The ratio is IREE's time over ours; **bold** rows are clearly beyond the plan's 2×
gate.

| Shape (1×N×K) | ours INT4 warm | IREE INT4 | ratio | ours INT4 cold | ours f16 warm | IREE f16 | ratio | llama.cpp f16 | ours f16 cold |
|---|---|---|---|---|---|---|---|---|---|
| **8192×2048** | 9.2 | 18.2 | 2.0 | 20.1 | 15.6 | 74.6 | **4.8** | 31.5 | 48.6 |
| **4096×2048 (q)** | 6.2 | 10.3 | 1.7 | 12.7 | 9.0 | 38.6 | **4.3** | 15.7 | 29.9 |
| **2048×4096 (o)** | 6.1 | 13.9 | **2.3** | 10.2 | 10.0 | 34.8 | **3.5** | 12.7 | 22.6 |
| 512×2048 (k, v) | 3.0 | 5.2 | 1.7 | 4.2 | 5.0 | 5.1 | 1.0 | 3.1 | 7.4 |
| 2048×512 (35B shared down) | 3.0 | 2.5 | 0.8 | 3.6 | 3.5 | 6.2 | 1.8 | 5.1 | 7.5 |
| 128×2048 (router) | 2.9 | 5.0 | 1.7 | 3.6 | 4.8 | 5.3 | 1.1 | 2.1 | 6.5 |
| 32×2048 | 2.9 | 4.9 | 1.7 | 3.4 | 4.7 | 9.0 | 1.9 | 2.0 | 6.3 |
| 768×2048 (gate, up) | 3.2 | 5.6 | 1.8 | 4.6 | 5.0 | 5.0 | 1.0 | 14.7 | 8.3 |
| **2048×768 (down)** | 3.0 | 3.2 | 1.1 | 3.6 | 3.6 | 8.9 | **2.5** | 4.9 | 7.6 |
| 151936×2048 (30B head) | 192.4 | 231.8 | 1.2 | 193.0 | 696.9 | 1413 | 2.03 | 784.9 | 696.5 |
| **248320×2048 (35B head)** | 306.8 | 346.3 | 1.1 | 307.0 | 1129.4 | 2570 | **2.3** | 1285 | 1128.6 |

Against llama.cpp's f16 kernels (also from iree#23507), ours are 1.1–2× faster on the large shapes (1.13–1.14× on the
two heads) and 1.6–2.4× slower on the smallest. The gate|up shape is the exception (5.0 against 14.7 µs). All rows,
and the 256- and 512-row shapes: [data/spread.md](data/spread.md).

### The floor of a small kernel (rocprofv3, µs)

| Kernel | 1 block | 2 | 8 | 48 |
|---|---|---|---|---|
| empty | 1.08 | 1.08 | 1.08 | 1.12 |
| INT4 matvec K = 128, cold | 2.32 | 2.32 | 2.32 | 2.36 |
| INT4 matvec K = 2048, cold | 3.20 | 3.28 | 3.60 | 4.64 |
| INT4 matvec K = 128, warm | 2.08 | 2.08 | 2.08 | 2.12 |
| INT4 matvec K = 2048, warm | 2.52 | 2.52 | 2.52 | 2.64 |

- **Dispatch, wave launch and teardown cost 1.1 µs.**
- **The serial structure comes from the generated code.** A wave works through its two rows one after the other, and
  through each row's 1,024-column chunks one after the other. The code waits for all of a chunk's loads
  (`s_waitcnt vmcnt(0)`) before it issues the next chunk's. So K = 128 is 2 dependent round trips per wave, and
  K = 2048 is 4.
- **Going from K = 128 to K = 2048 adds 0.88 µs cold (0.44 warm).** That suggests a few tenths of a µs per extra round
  trip, but the comparison is not controlled. K = 128 also uses only 4 of a wave's 32 lanes per row, and moves fewer
  bytes and less arithmetic. A controlled test, the same tile with its loads issued together, is the follow-up.
- **The f16 floor is higher.** An f16 row of 2,048 takes 8 dependent iterations, against 2 for INT4 (16 and 4 per wave).

**A grid-size anomaly.** Empty kernels at 380–384 blocks take 10.2–10.3 µs. At 376 blocks they take 1.28 µs, and from
385 to 576 blocks 1.52–1.60 µs. The 384-block INT4 matvec at K = 128 takes 10.0 µs, against 4.5–4.7 µs at 376 and
392.
- 384 blocks of 8 waves is exactly the chip's 3,072 wave slots, so the anomaly sits just below a grid that fills the
  card.
- None of the token's launch grids falls in the band. The mechanism is not known.

### E1: the fixed cost of a single kernel (cold, K = 2048; fit t = t0 + bytes / BW)

| Format | Range | t0 | BW (fit) | Shapes under 1 MB |
|---|---|---|---|---|
| INT4 | N = 32 … 8192 (the plan's range) | 3.34 µs | 518 GB/s | 3.36–4.64 µs |
| INT4 | N = 32 … 32768 (the run's sweep) | 4.38 µs | 720 GB/s | |
| f16 | N = 32 … 8192 (the plan's range) | 5.25 µs | 755 GB/s | 6.28–6.60 µs |
| f16 | N = 32 … 32768 (the run's sweep) | 6.27 µs | 853 GB/s | |

- **The two ranges give different fits.** The plan specified N = 32 … 8192. The sweep in the code ran to 32768, and the
  first version of this report scored the prediction on that wider fit.
- **The two extra points existed before any XTX measurement.**
- **The predictions table scores the plan's range.**

### Synchronization inside one persistent kernel (µs per stage, GPU stamps)

| Blocks (per WGP) | barrier, empty stage | barrier, 1 tile | barrier-pf, 1 tile | dataflow, 1 tile | dataflow-pf, 1 tile |
|---|---|---|---|---|---|
| 48 (1) | 0.92 | 3.00 | 3.64 | 2.84 | 1.76 |
| 96 (2) | 1.00 | 3.24 | 4.24 | 2.92 | 2.06 |
| 192 (4) | 1.32 | 3.48 | 4.76 | 2.99 | 2.16 |
| 240 (5, the shared occupancy limit) | 1.52 | 3.64 | 5.28 | 3.01 | 2.19 |

- **The barrier grows with the number of blocks.** At the token's grid it costs about as much per stage as the
  estimated gap between graph launches (about 1–1.5 µs). So replacing launches with barriers saves little by itself.
- **Dataflow does not grow.**
- **Prefetch helps dataflow,** where a waiting block claims the next tile. It hurts the barrier variant in this test,
  because block 0 holds every one-tile stage.

### E2: one shape, 256 times in a row (INT4, cold, 240 blocks, link kept awake; µs per matvec)

Launches and graph are event times amortized over 256. Persistent stages are GPU-stamp medians: the median over
stages 1–255 of one launch's stage times.

| Shape | launches | graph | barrier | barrier-pf | dataflow | dataflow-pf |
|---|---|---|---|---|---|---|
| 8192×2048 | 22.20 | 21.88 | 18.52 | 15.56 | 18.02 | 14.51 |
| 4096×2048 (q) | 14.05 | 13.77 | 12.52 | 9.68 | 11.17 | 9.15 |
| 2048×4096 (o) | 11.30 | 10.80 | 9.28 | 11.08 | 9.40 | 8.79 |
| 512×2048 (k, v) | 5.59 | 5.09 | 4.24 | 5.52 | 4.48 | 3.31 |
| 128×2048 (router) | 5.05 | 4.54 | 3.88 | 5.40 | 3.55 | 2.96 |
| 32×2048 | 4.98 | 4.47 | 3.80 | 5.36 | 3.24 | 2.60 |
| 768×2048 (gate, up) | 6.00 | 5.53 | 4.52 | 5.48 | 4.16 | 3.36 |
| 2048×768 (down) | 4.99 | 4.52 | 3.68 | 4.76 | 3.60 | 3.52 |

- **In isolation, a persistent stage beats a graph launch.** On small shapes dataflow-pf is 1.0–2.2 µs per matvec
  cheaper. On the 4–9 MB shapes it saves 2.0–7.4 µs (o, the K = 4096 shape, gains least), where the next matrix's
  first loads overlap the current one.
- **E2's graph time minus the E0 kernel time of the same shape is 0.9–1.1 µs on small shapes** (0.56–1.83 µs across all
  shapes). It is one of the two gap estimates below. It assumes the repeated single-shape sequence behaves like the
  token's mixed stages, and that separately traced kernel times match untraced ones.
- **Persistent event times need the link condition.** In the default-link E2 tables, persistent runs whose duration
  proxy (256 × the median stage time) exceeds about 1.0 ms read about 1 µs per matvec slower by events than by stamps.
  All E2 rows, f16, warm and the 96-block grid: [data/spread.md](data/spread.md).

### E3: the 48-layer token (ms, 1,393 matvecs in 241 dependent stages)

| Mode | INT4 cold | INT4 cold, link awake | INT4 warm (30.2 MB, no head) | warm, awake | f16 cold | f16, awake |
|---|---|---|---|---|---|---|
| per-op | 9.02 | 8.87 | 6.94 | 6.78 | 14.81 | 14.66 |
| per-op graph | 8.21 | 8.18 | 6.10 | 6.09 | 13.99 | 13.97 |
| grouped | 3.82 | 3.79 | 2.18 | 2.15 | 8.93 | 8.90 |
| **grouped graph** | **3.68** | **3.67** | **2.03** | **2.04** | 8.82 | 8.80 |
| persistent barrier | 4.04 | 3.80 | 2.74 | 2.49 | 8.78 | **8.53** |
| persistent barrier-pf | 3.83 | **3.60** | 2.74 | 2.49 | 8.98 | 8.74 |
| persistent dataflow | 4.90 | 4.66 | 3.63 | 3.38 | 8.94 | 8.71 |
| persistent dataflow-pf | 4.61 | 4.37 | 3.49 | 3.24 | 8.90 | 8.67 |

- **Run-to-run spread is under 1 % on every row.**
- **Bandwidth bounds for the INT4 token:** 1.68 ms at 960 GB/s, 1.88 ms at the 858 GB/s our kernel reaches on the vocab
  head.
- **The f16 token is close to its bandwidth bound in every grouped or persistent mode** (690 GB/s).
- **Warm, the persistent kernel is 22 % slower than the graph even with the link awake.** The stages are short, and the
  synchronization dominates.
- **Other grids:**
  - plain barrier at its own occupancy (336 blocks): 3.86 ms;
  - dataflow at 384 blocks: 4.00 ms;
  - 96 blocks: 5.7–6.7 ms.

### Time outside the kernel (INT4 cold token, µs)

Each mode was timed three ways in one process:
- by events around the run;
- by GPU stamps just before and just after the run;
- by events around the stamps.

"Hidden" is the difference between the last two in each window: time that passes on the wall clock but between no GPU
stamps. Each cell is the median of the three runs' medians, each run's over 60 windows.

| Mode | event | stamp to stamp | hidden | windows with hidden > 50 µs | hidden, link kept awake |
|---|---|---|---|---|---|
| grouped graph | 3676.5 | 3658.1 | 10.4 | 0 of 180 | 10.6 (0 of 180) |
| persistent barrier-pf | 3816.5 | 3578.4 | 242.8 | 179 of 180 | 9.2 (0 of 180) |
| persistent dataflow-pf | 4589.1 | 4348.5 | 242.9 | 179 of 180 | 9.2 (0 of 180) |
| empty kernel | 8.7 | 6.3 | 8.7 | 0 of 180 | 8.7 (0 of 180) |

**What the single-run diagnostics showed.** The logs are in [data/dev-observations/](data/dev-observations/).
- **The host saw the same extra time.** The launch call returns in 1.4 µs, and the host wait ends ~3,825 µs after
  submit. rocprofv3's kernel duration matched the stamped span (3.572 against 3.562 ms).
- **Each launch paid it.** Two tokens in one event window cost 2 × 3,821 µs.
- **Spinning on `hipEventQuery` instead of blocking made no difference.**
- **The sync-state reset before the launch is not the cause.** Synchronizing after the reset, or placing an empty
  kernel between the reset and the event, made no difference.
- **The display was not needed.**
  - With two monitors on the XTX: 65 of 80 windows had it.
  - With the monitors on the integrated GPU: 79 of 80.
- **No link speed change was seen.** The CPU-side root port read 16 GT/s x16 at every 1 ms sample during the run; it
  reads 2.5 GT/s x1 only when the program is idle. A 1 ms poll of one port cannot exclude shorter transitions or
  changes on the card's downstream links.
- **A plain kernel showed it too.** E0 times plain kernels launched back to back (events around each, no host wait
  between). The f16 35B vocab head, 1,129 µs by rocprofv3, reads 1,385 µs by events (+257 µs). The 0.70 ms 30B head
  reads +21 µs, and the INT4 heads (0.19 and 0.31 ms) +12 µs. E0 has no keep-alive arm, so this kernel was not tested
  with the keep-alive.
- **There is an apparent transition near 1.0 ms.** The default-link E2 tables hold 528 persistent measurements (3 runs
  × 44 shape, grid and format rows × 4 variants). Each is a median, and the duration used is a proxy: 256 × the median
  stage time.
  - Below a proxy of 1.014 ms the extra time was 5–59 µs.
  - Above it, 200–323 µs.
  - At 1.014 ms both occur.

  Establishing a cutoff would need paired whole-launch GPU spans and event times.
- **The RX 7900 XT showed none.** One full-token run, on an x4 link with ASPM off: event 4,380 µs, stamps 4,373 µs. That
  comparison also changes the card and the topology.

**What it might be.**
- **The leading hypothesis is PCIe link power management after about 1 ms with no bus traffic.** ASPM L1 is enabled on
  the XTX's links and disabled on the XT's.
- **It is not established, for four reasons:**
  - ASPM was never changed on the same card; that needs root.
  - The keep-alive changes CPU activity and bus traffic together.
  - Other causes are not excluded: runtime completion and fence handling, the card's host-interface power management,
    CPU package states, or a combination.
  - The extra time lies somewhere between the events and the stamp kernels, at either end of the window. Traced runs
    show the persistent kernel starting ~4 µs after the preceding work, which points to the end. But tracing adds its
    own completion handling.
- **The keep-alive is a workaround, not a fix.** It is a host loop reading one 32-bit word of device memory through the
  BAR, and it costs a CPU core.

### Where the token's GPU time goes (INT4 cold, µs per stage)

- **Grouped:** rocprofv3 kernel time, mean per stage kind over the traced run.
  - The traced run also contains per-op tokens and warmup rounds.
  - Kernels are assigned to a stage kind by grid size. The router, o and vocab-head means therefore pool grouped and
    per-op executions.
  - The 128-block o class is told apart from the per-op expert down by duration.
- **Persistent:** GPU-stamp stage time, the mean over the token's stages of each stage's median over 10 runs. A
  persistent stage includes its synchronization; a kernel time does not include the gap to the next launch.

| Mode (blocks) | q\|k\|v | o | router | gate\|up ×8 | down ×8 | vocab head | sum (ms) |
|---|---|---|---|---|---|---|---|
| grouped launches | 14.90 | 9.82 | 3.20 | 26.69 | 10.29 | 192.75 | 3.310 |
| barrier (240) | 14.59 | 10.00 | 3.84 | 26.59 | 19.45 | 203.12 | 3.778 |
| **barrier-pf (240)** | **9.74** | **8.56** | 3.15 | 27.71 | **21.12** | 196.56 | 3.570 |
| dataflow (240) | 13.78 | 9.97 | 3.43 | 38.04 | 27.13 | 214.40 | 4.648 |
| dataflow-pf (240) | 10.24 | 8.52 | 2.94 | 38.38 | 26.29 | 206.57 | 4.353 |
| barrier (336) | 14.51 | 9.99 | 4.32 | 23.81 | 18.24 | 198.88 | 3.599 |
| dataflow (384) | 13.97 | 11.57 | 4.26 | 27.39 | 21.65 | 208.32 | 3.992 |
| dataflow, one-tile claims on down (240) | 13.98 | 10.00 | 3.44 | 38.33 | 21.54 | 213.83 | 4.402 |

**The grouped graph's 3.66 ms of GPU time (stamp to stamp), as a model:**

| Part | ms | How it is derived |
|---|---|---|
| The weight bytes | 1.88 | 1.616 GB at 858 GB/s, the vocab head's rate |
| Kernel time above that baseline | 1.43 | The traced kernel sum (3.31 ms) minus the bytes |
| Gaps, estimate 1 | 0.35 | Untraced stamp-to-stamp span minus the traced kernel sum |
| Gaps, estimate 2 | 0.23 | E2's ~0.95 µs per gap (graph per matvec minus E0 kernel time) × 241 |

- **These are two model-based estimates, not a measured interval.**
  - Estimate 1 subtracts a pooled traced kernel population from a separate untraced span, and tracing changes kernel
    timing.
  - Estimate 2 extrapolates repeated single-shape sequences to the token's mixed stages.
- **Neither bounds the true gaps.** A defensible decomposition needs matched graph invocations and a measured tracing
  effect.

**barrier-pf against the grouped launches, per layer.** These are differences between intervals that end at
synchronization boundaries, not isolated costs of the named projections.
- **The prefetch work sits in the previous interval.** Stage i's prefetch is issued before stage i − 1's end stamp is
  taken, so it is counted in stage i − 1's interval. The down interval therefore contains the issue of the next
  layer's q|k|v prefetch, and the q|k|v interval excludes it.
- **Gains:** q|k|v is 5.2 µs faster and o 1.3 µs faster.
- **Losses:** gate|up is 1.0 µs slower, and the down interval 10.8 µs slower.
- **Per token:** the stage sum is 0.26 ms above the grouped kernel sum. Against estimate 1's 0.35 ms of gaps, that
  gives an estimated net of 0.09 ms. The windows measured 0.08 ms directly (3.658 − 3.578 ms, stamp to stamp).

**The down interval (1,024 tiles of K = 768) is where the persistent kernel loses most.**
- **The problem isn't the prefetch shift.** The plain barrier variant, with no prefetch, is also slow there (19.45
  against 10.29 µs).
- **What a down tile is:** 16,384 short rows, each 384 bytes of INT4 weights plus 24 bytes of scales and zeros. A tile
  is one round trip per row, two per wave, by the generated code.
- **As a launch:** 1,024 one-tile blocks; the hardware refills each slot as a block exits.
- **In the persistent kernel at 240 blocks:** 64 blocks do 5 tiles one after another, and 176 do 4.

**What was tested:**
- **More blocks help little.** At 336 blocks, 16 blocks do 4 tiles and the rest 3, and the barrier stage only goes from
  19.45 to 18.24 µs.
- **One-tile claims help dataflow.** Its down stage goes from 27.1 to 21.5 µs.
- **Prefetch does not help.**

The stage takes 19–21 µs, and its bytes alone take 7.8 µs at 858 GB/s. A chain of 5 tiles of serial loads is a
candidate cause, not an established one.

**Dataflow loses more, also on gate|up.** A claim takes ⌈tiles / 240⌉ contiguous tiles, which leaves blocks idle:

| Stage | Tiles | Tiles per claim | Claims | Idle blocks |
|---|---|---|---|---|
| q\|k\|v | 320 | 2 | 160 | 80 |
| gate\|up | 768 | 4 | 192 | 48 |
| down | 1,024 | 5 | 205 | 35 |
| vocab head | 9,496 | 40 | 238 | 2 |

Its gate|up stage takes 38 µs against the barrier's 26.6 µs at the same per-block maximum of 4 tiles. Contiguous chunks
against round-robin tiles is the obvious difference, but it was not tested.

## Predictions against outcomes

**How they are scored.**
- **Where the plan named no metric, the choice is stated.** E2 is scored on default-link event times, an operational
  reading of the plan, with the awake-link GPU stamps beside them.
- **Boundaries are strict:** 1.46 is not ≥ 1.5, and 0.92 is outside 1–2.

| Prediction (PLAN.md) | Outcome |
|---|---|
| E0 gate: our kernels within about 2× of IREE's, else stop and report first | **Tripped.** Six of 26 shapes clearly beyond 2× and two near it, all faster than IREE's. The script did not stop. Reported above. |
| P1: single kernels t0 3–5 µs, BW 700–850 GB/s (cold, N = 32 … 8192) | INT4: t0 3.34 µs **held**, BW 518 GB/s **missed**. f16: t0 5.25 µs **missed** (just above 5), BW 755 GB/s **held**. (The wider sweep to N = 32768 gives 4.38 µs / 720 GB/s and 6.27 µs / 853 GB/s.) |
| P2 (revised): one-tile stage barrier 3.5–4.5, dataflow 2.8–3.5, dataflow-pf 2.0–2.6 µs | Barrier 3.00–3.64: **below** at low grids. Dataflow 2.84–3.01: **held**. Dataflow-pf 1.76–2.19: **below** at 48 blocks, **held** above. |
| Empty-stage barrier 1–2 µs | 0.92–1.52: **held** from 96 blocks, **missed** (below) at 48. |
| barrier-pf no better than barrier in sync | **Held.** It was worse: 3.64–5.28. |
| E2: dataflow-pf ≥ 1.5 µs below launches and graph on small shapes | Default-link event time. Below launches: **held** on seven of eight shapes (1.58–2.58), **missed** on down (1.46). Below the graph: **held** for the K = 2048 shapes (1.54–2.04), **missed** for K = 512 / 768 (0.89–0.96). Awake-link stamps: below launches 1.47–2.64 (down misses); below the graph 1.58–2.17 for K = 2048, 1.00–1.04 for K = 512 / 768. |
| E2: shapes over 4 MB within 15 % across modes | **Missed** for INT4. On default-link event time, dataflow-pf is 9–29 % faster than the graph, and barrier-pf ranges from 24 % faster to 12 % slower (o). On awake-link stamps, dataflow-pf is 19–34 % faster. |
| E3: per-op 7–9 ms | 9.02: **missed** by 0.02 ms (8.87 with the link awake). |
| E3: grouped and grouped graph 2.9–3.6 ms | 3.68–3.82: **missed**, slightly above. |
| E3: best persistent 2.6–3.4 ms | 3.83 (3.60 with the link awake): **missed**. |
| Refutation test: persistent ≥ 1.25× faster than the grouped graph | **Not met.** 0.96× as configured, 1.02× with the link awake. |
| Warm token: the graph-vs-best-persistent gap at least as large as cold, in µs per stage | The plan gives no sign. Signed as the persistent kernel's saving per stage: cold −0.62 / +0.30 µs (default / awake), warm −2.93 / −1.90 µs. **Missed** as a saving, which is the plan's stated reasoning ("no DRAM latency to hide behind"). The unsigned gap is larger warm because the persistent kernel gets slower. |

## What this says about the floor in iree#24165

- **Wall time vs kernel time.**
  - iree#24165's ~37 µs per small matvec is benchmark wall time.
  - As warm kernel time the small shapes are 2.5–9.0 µs in iree#23507's tables and 2.9–5.0 µs for our kernels.
  - These are not paired measurements of that benchmark, so the ~37 µs is not decomposed here.
- **Once the experts are grouped and the launches graphed:**
  - the gaps are a minority of the matvec token (two estimates, 0.23–0.35 ms of 3.66);
  - 1.43 ms of the kernel time is above a bandwidth baseline;
  - serial dependent loads in the tile are a likely, untested contributor.
- **This persistent kernel replaces each launch gap with a barrier of about the same cost.** What it can add is
  starting the next stage's weight loads before the barrier. What it can lose is any stage whose tiles a resident grid
  works through one after another, such as the MoE down projections.
- **On this XTX, single launches longer than ~1 ms showed ~250 µs of event- and host-visible time that GPU clocks
  don't.** A long persistent kernel pays it, and launch sequences didn't. Time any persistent kernel by GPU stamps as
  well as by events.
- **What we would try next, in this order:**
  1. group the experts;
  2. cut dependent round trips per tile: issue all of a tile's loads before using any, and let blocks batch their
     tiles;
  3. only then fuse into one launch, with cross-stage prefetch, and with the extra completion time understood.

## Method findings (each cost a debugging cycle; recorded for anyone building a persistent kernel on gfx1100)

Findings 1 and 5 come from RX 7900 XT smoke runs whose logs are not archived.

1. **Don't read the clock on every poll in a device wait.**
   - **What happened:** the wait loops read the wall clock (`s_sendmsg_rtn` GET_REALTIME) on every poll. With about 150
     waiting waves, one wave's reply stalled for 200 ms while its neighbours polled 620,000 times; it only un-stalled
     when they gave up.
   - **The fix:** read the clock once per 4,096 polls. Three variants went from 0 of 4 to 4 of 4 passing runs.
   - **What didn't matter:** poll style (acquire load, relaxed load plus fence, atomic add of 0) and sleep length.
   - **Scope:** seen on the RX 7900 XT, one ROCm build.
2. **Put every polled or counted word on its own 128-byte line.**
3. **The occupancy API's count is not a residency guarantee.** A barrier needs every block resident. Probe it at every
   grid used: an empty-stage barrier either completes or times out.
4. **A device-wide barrier's cost grows with the block count:** 0.9 µs at 48 blocks, 1.5 µs at 240. Per-stage done
   counters (dataflow) don't grow.
5. **Chunk sizes must round up.** With ⌊9,496 / 210⌋ = 45 tiles per chunk, the vocab head had 212 chunks for 210 blocks.
   The two blocks with a second chunk made the stage 1.44× as long.
6. **Time a persistent kernel by GPU stamps as well as by events.**
   - On this XTX, single launches longer than ~1 ms showed ~250 µs that only events and the host saw.
   - rocprofv3's kernel duration doesn't show it either.
   - A host thread reading device memory over PCIe removed it in every persistent test (`DF_LINK_KEEPALIVE=1`).
7. **Grids of 380–384 blocks cost ~9 µs extra per launch on this XTX:** just under the 3,072 wave slots of 384 blocks
   of 8 waves. 376 and 385 blocks are normal. Not explained.
8. **Put the stop rules in the script.** The E0 gate was in the plan but not in run.sh, so it tripped without stopping
   anything.
9. **Make the planned fit range the code's range.** The plan's E1 range was N = 32 … 8192, the code swept to 32768,
   and the first version of this report scored the prediction on the wider fit.

## Limitations

- **Matvecs only,** as stated above.
- **Synthetic dependencies.** The stages are synchronized but pass no data: every matvec reads a fixed activation
  vector, which is always cached. A real decode step reads the previous stage's output.
- **Random weights in our INT4 layout:** GPTQ-like groups of 128 with f16 scales and zeros. IREE's exact INT4 layout may
  differ.
- **Our hand-written tile kernel, not IREE codegen.** It is faster than IREE's on the large shapes (see the deviation
  section), and only one tile implementation was tested.
- **One machine, one board topology, one driver, kernel and ROCm build.** The extra completion time depends on
  something about this platform, and other boards and BIOS settings may differ.
- **The RX 7900 XT disagrees, in one run.** Full INT4 cold token, 42 WGPs, 210-block grid, x4 link with ASPM off:
  - grouped graph 3.44 ms, barrier-pf 4.37 ms (27 % slower), dataflow-pf 5.07 ms;
  - events and stamps agree there, so this is GPU time;
  - not investigated;
  - log in [data/20261109-xt-window/](data/20261109-xt-window/), from the build just before the final one (same
    kernels).
- **"Cold" means cache-cold, not DRAM-cold.** The copies are walked in address order, so consecutive launches touch
  adjacent DRAM pages. The token's stages read non-adjacent weights.
- **The gap estimates are models,** as described above.
- **Traces are not in data/.** The rocprofv3 traces (~370 MB per run) are not included, so the kernel times in E0, E1,
  the floor and the grouped breakdown can be checked against each run's `tables.md` but not re-derived from data/.
  run.sh regenerates them.

## Reproduce

```
ROCM=/opt/rocm ./build.sh                    # hipcc from a ROCm install
./run.sh smoke                               # a second card first, small sizes (DEVICE=n picks the HIP device;
                                             # an XTX alone: DEVICE=0 EXPECT_CUS=48 ./run.sh smoke)
./run.sh xtx                                 # the measurements: results/<date>-xtx/ with tables.md
python3 analyze.py --spread spread.md results/*-xtx    # medians and ranges across runs
```

- **The watchdog is optional.** Set `GUARD=/path/to/watchdog`: run.sh calls it as `GUARD LOG SECS CMD...`. Without it,
  each stage runs under `timeout` and the clocks are not pinned. Pin them for comparable numbers.
- **The pieces run directly:** `decode_floor check | shapes | sweep | floor | sync | pshapes | token`. Run it without
  arguments to print usage, including the environment switches.
- **Licence:** MIT.

## Revisions

**After the first version** (tag `report-2026-11-09`):
- **The round-trip cost.** It was given as about 1 µs per dependent round trip. The generated code runs a wave's two
  rows in series, so the floor comparison spans two round trips per K step. The per-round-trip figure is now presented
  as uncontrolled.

**After an independent review:**
- **The extra completion time.**
  - "Any kernel longer than 1 ms pays it" became an observation on the tested launches.
  - Link power management is now labelled a hypothesis.
  - The all/none 1 ms cutoff became an apparent transition in a duration proxy, with the corrected low range of
    5–59 µs.
  - The single-run diagnostics are archived in [data/dev-observations/](data/dev-observations/).
- **P1.** Scored on the plan's N = 32 … 8192 range: INT4 bandwidth is now a miss (518 GB/s).
- **The gap allocation.** "6–10 % of the token" became two model-based estimates with their assumptions.
- **The residual.** The 1.43 ms above the bandwidth baseline is no longer described as mostly memory latency.
- **The E0 deviation.** The interpretation is restricted to this implementation and schedule, and the synthetic
  dependencies (no data between stages) are disclosed.
- **Smaller corrections:**
  - stage differences relabelled as interval differences, with the prefetch shift disclosed;
  - the down row size now counts scales and zeros;
  - E2 and E3 timing documented separately;
  - signed and strict prediction scoring;
  - correctness maxima and metric;
  - the display exception in Setup;
  - format coverage;
  - aggregation wording;
  - 30.2 MB;
  - the near-2× E0 cases.
