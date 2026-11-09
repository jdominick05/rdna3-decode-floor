# Where a Qwen3-30B-A3B decode token's matvec time goes on an RX 7900 XTX

Measured 2026-11-09 on one RX 7900 XTX (gfx1100, 96 CUs / 48 WGPs, 960 GB/s GDDR6, 96 MB Infinity Cache). Plan and
pre-registered predictions: [PLAN.md](PLAN.md) (written before measuring; its last section was revised after an
RX 7900 XT smoke run and before any XTX measurement). Raw data: [data/20261109-xtx/](data/20261109-xtx/).

## The answer in short

- **The matvec "token" here is 1,393 matrix-vector products only** (48 layers + the vocab head): no attention, norms,
  routing or sampling. Compare the modes below with each other, never with whole-token speeds such as IREE's 120 tok/s.
- **Grouping the experts is the big lever.** One launch per matvec: 9.00 ms. One launch per dependent stage (q|k|v
  together, the 8 experts' gate|up together, their 8 downs together): 3.80 ms. A HIP graph of those: 3.68 ms.
- **With experts grouped and a graph, launch gaps are a small part of the token.** The grouped kernels add up to
  3.36 ms of GPU time (rocprofv3), so the gaps between them total about 0.32 ms: 1.33 µs per stage, 9 % of the token.
- **The fixed cost of a small kernel is inside the kernel, not in the launch.** Single-kernel times fit
  t = t0 + bytes / BW with t0 = 4.4 µs (INT4) and 6.2 µs (f16), cold, measured with rocprofv3 kernel timestamps.
- **A persistent kernel removes the gaps but did not beat grouped launches here.** Best of four persistent variants:
  3.83 ms (3.86 ms with more blocks), against 3.68 ms. In isolation a small matvec costs less as a persistent stage
  (2.6–3.7 µs against 4.5–5.5 µs as graph launches), but on the full token one stage type, the experts' down
  projections (K = 768, short rows), runs at 18–27 µs inside the persistent kernel against 10.3 µs as a launch,
  which costs more than the gaps saved. The pre-registered prediction for the persistent kernel (2.6–3.4 ms) failed.

## Setup

- **Card and software:** RX 7900 XTX alone (`HIP_VISIBLE_DEVICES=0`), Linux 7.0.0-38, TheRock ROCm 10.2.0a20261011
  (HIP 7.17, AMD clang 24), DPM forced high (SCLK top state 2,482 MHz, MCLK 1,249 MHz), amdgpu runtime PM off. The XTX
  also drives the desktop. Every process ran under a GPU watchdog (VRAM limit, kernel-log fault watch).
- **Model shapes:** Qwen3-30B-A3B, per layer q 1×4096×2048, k and v 1×512×2048, o 1×2048×4096, router 1×128×2048, 8 of
  128 experts × (gate and up 1×768×2048, down 1×2048×768), vocab head 1×151936×2048. Random weights in the real shapes
  (timing does not depend on values). The routed experts are drawn at random per token (8 routing samples).
- **Formats:** f16 weights; INT4 with groups of 128 and an f16 scale and zero point per group (1.616 GB per INT4 token,
  6.083 GB per f16 token; f16 allocates only the 8 experts a token uses, since 128 would not fit).
- **One matvec tile function in every mode.** 16 rows per 256-thread block, f32 accumulation with `v_dot2_f32_f16`;
  INT4 nibbles become f16 via the 0x6400 exponent trick. Checked in the generated code: the same loads
  (`global_load_b128`) and the same instruction mix in every kernel up to scheduling.
- **Modes:**
  - per-op: one launch per matvec;
  - grouped: one launch per stage;
  - graph: the same launches captured in a HIP graph;
  - four persistent variants, one launch per token:
    - *barrier*: static round-robin tiles, a split-phase device-wide barrier between stages;
    - *dataflow*: per-stage claim and done counters, each block claims at most one chunk per stage and waits only for
      the stage before it;
    - *-pf*: prefetch the claimed tile's weights before the wait. Not for the routed experts' gate|up, whose experts
      the router picks.
- **Cold and warm:**
  - cold: weight copies totalling at least 400 MB, walked in order, so a copy is re-read only after 400+ MB of other
    traffic (each row records copies and reuse distance);
  - warm: one copy reused;
  - warm token: every layer reuses one layer's weights and 8 experts (31 MB), no vocab head.
- **Timing:**
  - single kernels: rocprofv3 `--kernel-trace` GPU timestamps, the same tool as iree#23507;
  - token and E2 modes: HIP events around the whole sequence, modes interleaved, order rotated each round, median of
    10 rounds after 2 warmup rounds;
  - persistent stages: the GPU wall clock (100 MHz) at stage boundaries.

## Correctness

- Every kernel against a naive reference, 26 shape/format pairs: max relative error 1e-4 (f16) and 3e-4 (INT4); the
  reference itself against a double-precision host computation; non-finite values fail.
- Every persistent variant's token outputs are **bitwise equal** to the per-op outputs, in the cold, warm and f16
  tokens and at every grid used.
- A dependency check in those runs: every producer adds its tiles' ids to a per-stage sum before it publishes; every
  consumer checks the complete sum after its wait. No failure.
- No wait timed out in any reported run.

## Results

### E0: single kernels, their shapes (rocprofv3 kernel time, µs)

iree#23507's numbers were taken warm ("Both benchmarks run thousands of iterations, so L2/Infinity Cache is warm"), so
they compare with our warm column.

| Shape (1×N×K) | MB (INT4) | ours INT4 warm | IREE INT4 (warm) | ours INT4 cold | ours f16 warm | IREE f16 (warm) | llama.cpp f16 (warm) | ours f16 cold |
|---|---|---|---|---|---|---|---|---|
| 8192×2048 | 8.91 | 9.2 | 18.2 | 20.1 | 15.7 | 74.6 | 31.5 | 48.5 |
| 4096×2048 (q) | 4.46 | 6.1 | 10.3 | 12.6 | 9.0 | 38.6 | 15.7 | 29.8 |
| 2048×4096 (o) | 4.46 | 6.1 | 13.9 | 10.0 | 10.0 | 34.8 | 12.7 | 22.9 |
| 512×2048 (k, v) | 0.56 | 3.0 | 5.2 | 4.0 | 5.0 | 5.1 | 3.1 | 7.4 |
| 128×2048 (router) | 0.14 | 2.9 | 5.0 | 3.6 | 4.8 | 5.3 | 2.1 | 6.5 |
| 32×2048 | 0.03 | 2.9 | 4.9 | 3.4 | 4.7 | 9.0 | 2.0 | 6.3 |
| 768×2048 (gate, up) | 0.84 | 3.2 | 5.6 | 4.6 | 5.0 | 5.0 | 14.7 | 8.2 |
| 2048×768 (down) | 0.84 | 3.0 | 3.2 | 3.5 | 3.6 | 8.9 | 4.9 | 7.6 |
| 151936×2048 (head) | 165 | 192.5 | 231.8 | 192.9 | 697.5 | 1413 | 784.9 | 697.0 |

Against IREE's per-shape kernels (warm against warm), our tile kernel is faster or equal on every shape but one (INT4
2048×512: 3.0 against 2.5 µs). On the small shapes, the ones that matter for the floor, IREE takes 0.8–1.8× our
time in INT4 and 1.0–2.5× in f16, so mostly inside the plan's "within about 2×" gate; the large f16 shapes differ by
4–5×, a codegen difference iree#23507 also reports against llama.cpp. Against llama.cpp's f16 kernels, ours is faster
on the large shapes (15.7 against 31.5 µs on 8192×2048) and about 2× slower on the small ones (4.7–5.0 against
2.0–3.1 µs); the gate|up shape is the exception (5.0 against 14.7). So the comparison is between kernels of a similar
class, not an outlier.

### E1: the fixed cost of a single kernel (cold, K = 2048, N = 32 … 32768)

| Format | t0 | BW (fit) | Shapes under 1 MB |
|---|---|---|---|
| INT4 | 4.4 µs | 721 GB/s (75 %) | 3.4–4.7 µs |
| f16 | 6.2 µs | 851 GB/s (89 %) | 6.3–6.6 µs |

The f16 floor is higher because of how the tile loop reads: each lane reads 8 columns per iteration, so an f16 row of
2,048 takes 8 dependent iterations (load, wait, accumulate), against 2 for INT4 (32 columns per iteration). The floor
of a tiny matvec here is mostly dependent memory latency inside the kernel.

### Synchronization inside one persistent kernel (µs per stage, GPU stamps)

| Blocks per WGP | barrier, empty stage | barrier, 1 tile | dataflow, 1 tile | dataflow-pf, 1 tile |
|---|---|---|---|---|
| 1 (48 blocks) | 0.92 | 3.04 | 2.88 | 1.76 |
| 2 | 1.04 | 3.24 | 2.88 | 2.04 |
| 4 | 1.32 | 3.40 | 3.00 | 2.16 |
| 5 (240, the shared occupancy limit) | 1.52 | 3.60 | 3.00 | 2.20 |

- **The barrier grows with the number of blocks.**
- **Dataflow, which has no barrier, does not grow.**
- **Prefetch helps dataflow**, where an already-waiting block claims the next tile. It cannot help the barrier
  variant in this test, because block 0 holds every one-tile stage.

### E2: one shape, 256 times in a row (µs per matvec, event time; INT4, cold, 240 blocks)

| Shape | launches | graph | barrier | barrier-pf | dataflow | dataflow-pf |
|---|---|---|---|---|---|---|
| 8192×2048 | 22.4 | 22.0 | 19.5 | 16.4 | 19.1 | 15.6 |
| 4096×2048 (q) | 14.1 | 13.8 | 13.6 | 10.7 | 12.2 | 10.2 |
| 512×2048 (k, v) | 5.7 | 5.1 | 4.3 | 6.5 | 5.5 | 3.4 |
| 128×2048 (router) | 5.1 | 4.6 | 4.0 | 6.5 | 3.6 | 3.0 |
| 32×2048 | 5.1 | 4.5 | 3.9 | 6.4 | 3.3 | 2.6 |
| 768×2048 (gate, up) | 6.1 | 5.5 | 5.5 | 6.5 | 5.4 | 3.6 |
| 2048×768 (down) | 5.1 | 4.6 | 3.8 | 5.8 | 3.7 | 3.7 |

In isolation, the dataflow-pf persistent stage beats launches by 1.4–2.5 µs per small matvec and by 2.4–6.8 µs on the
4–9 MB shapes, where the next matrix's first loads overlap the current one. Full table, f16 and warm rows, and the
96-block grid: [data/20261109-xtx/tables.md](data/20261109-xtx/tables.md).

### E3: the 48-layer token (ms, 1,393 matvecs in 241 dependent stages; median, p10–p90 in the data)

| Mode | INT4 cold | INT4 warm (31 MB, no head) | f16 cold |
|---|---|---|---|
| per-op | 9.00 | 6.92 | 14.81 |
| per-op graph | 8.20 | 6.09 | 13.99 |
| grouped | 3.80 | 2.18 | 8.93 |
| **grouped graph** | **3.68** | **2.03** | 8.81 |
| persistent barrier | 4.04 (3.86 at 7 blocks/WGP) | 2.73 | 8.77 (a tie with the graph: 0.04 ms, under the 0.05 ms threshold) |
| persistent barrier-pf | 3.83 | 2.73 | 8.98 |
| persistent dataflow | 4.90 (4.02 at 8 blocks/WGP) | 3.62 | 8.95 |
| persistent dataflow-pf | 4.60 | 3.49 | 8.90 |

Bandwidth bounds for the INT4 token: 1.68 ms at 960 GB/s, 2.24 ms at the 721 GB/s our kernel reaches cold. The f16
token (6.1 GB) is bandwidth-bound in every grouped or persistent mode (690 GB/s).

### Where the persistent kernel loses (INT4 cold, per stage, µs)

| Stage | grouped launch (rocprofv3) | persistent barrier stage (240 blocks) | (336 blocks) |
|---|---|---|---|
| q\|k\|v (320 tiles) | 14.9 | 14.5 | 14.4 |
| o (128 tiles) | 9.8 | 10.0 | 10.0 |
| router (8 tiles) | 3.2 | 3.8 | 4.4 |
| gate\|up ×8 (768 tiles) | 27.9 | 26.6 | 23.9 |
| **down ×8 (1,024 tiles, K = 768)** | **10.3** | **19.4** | **18.3** |
| vocab head (9,496 tiles) | 193.0 | 202.3 | 199.7 |

- **A down tile is a single memory round trip.** Its rows are 384 bytes in INT4, so each lane does one load.
- **A launch keeps the down tiles in flight.** It has 1,024 one-tile blocks, and the hardware refills each slot as a
  block exits.
- **The persistent grid serializes them.** 240–384 resident blocks work through 3–4 tiles each, one after another,
  and static round-robin leaves some blocks a fourth tile. That costs 8–9 µs per layer, about 0.4 ms per token, more
  than the 0.32 ms of gaps it removes.
- **More blocks help, but don't close it.** The persistent kernel's grid is capped by its most register-hungry variant
  (5 blocks per WGP for the prefetch kernels, at 119–132 VGPRs). Running the plain variants at their own limits
  (barrier 7, dataflow 8) narrows the gap but does not close it.

## Predictions against outcomes

| Prediction | Outcome |
|---|---|
| P1: single kernels t0 3–5 µs, BW 700–850 GB/s (cold) | INT4 4.4 µs, 721 GB/s: **held**. f16 6.2 µs, 851 GB/s: t0 **missed** (dependent loads, see E1) |
| P2 (revised): one-tile stage barrier 3.5–4.5, dataflow 2.8–3.5, dataflow-pf 2.0–2.6 µs | barrier 3.0–3.6 (**below** at low grids), dataflow 2.9–3.0 **held**, dataflow-pf 1.8–2.2 (**at or below**) |
| Empty-stage barrier 1–2 µs | 0.92–1.52 **held** |
| barrier-pf no better than barrier in sync | **held** (worse: 3.7–5.1) |
| E2: dataflow-pf ≥ 1.5 µs below launches and graph on small shapes | below launches: **held** except down (1.4); below the graph: **held** for K = 2048 shapes, **missed** for K = 512 / 768 (0.9) |
| E2: shapes over 4 MB within 15 % across modes | **missed** for INT4: dataflow-pf 21–30 % faster than launches (barrier-pf mixed: 24–27 % faster on two shapes, 7 % slower on o); f16 within 15 % except q, where barrier-pf is 22 % slower |
| E3: per-op 7–9 ms | 9.00, **held** at the edge |
| E3: grouped and grouped-graph 2.9–3.6 ms | 3.68–3.80, **missed** (slightly above) |
| E3: best persistent 2.6–3.4 ms | 3.83 (3.86 at higher occupancy), **missed** |
| Refutation test: persistent ≥ 1.25× faster than the grouped graph | **not met**: this implementation did not deliver the gain. The breakdown above shows why, and that it is not launch cost |
| Warm token: persistent-vs-graph gap at least as large as cold | cold +0.15 ms, warm +0.70 ms: **held** |

## What this says about the floor in iree#24165

- **Wall time vs kernel time.** iree#24165's ~37 µs per small matvec is benchmark wall time. As kernel time (iree#23507,
  and our E0) the same shapes are 3–5 µs.
- **Where the token's time goes once experts are grouped and launches are graphed:**
  - about 9 % is launch gaps;
  - the rest is kernel time, which for small matvecs is mostly a fixed in-kernel cost: dependent memory latency,
    4.4 µs INT4 / 6.2 µs f16 here.
- **What would move it, in this order:**
  1. group the experts;
  2. reduce dependent loads per tile (more columns per lane, or loads issued ahead in the K loop);
  3. only then fuse into one launch. A persistent kernel only pays off if every stage, especially short-row ones like
     the MoE down projections, keeps as many loads in flight as a full launch does.

## Method findings (each cost a debugging cycle; recorded for anyone building a persistent kernel on gfx1100)

1. **Don't read the clock on every poll in a device wait.**
   - What happened: the wait loops read the wall clock (`s_sendmsg_rtn` GET_REALTIME) on every poll. With about 150
     waiting waves, one wave's reply stalled for 200 ms while its neighbours polled 620,000 times; it only un-stalled
     when they gave up.
   - The fix: read the clock once per 4,096 polls. Three variants went from 0 of 4 to 4 of 4 passing runs.
   - What didn't matter: poll style (acquire load, relaxed load plus fence, atomic add of 0) and sleep length.
2. **Every polled or counted word on its own 128-byte line.**
3. **The occupancy API's count is not a residency guarantee.** A barrier needs every block resident; probe it (an
   empty-stage barrier either completes or times out) at every grid used.
4. **A device-wide barrier's cost grows with the block count** (0.9 µs at 48 blocks, 1.5 µs at 240). Per-stage done
   counters (dataflow) don't.
5. **Chunk sizes must round up.** With floor(9,496 / 210) = 45 tiles per chunk, the vocab head had 212 chunks for
   210 blocks; the two blocks with a second chunk made the stage 1.44× as long.

## Limitations

- **Matvecs only**, as stated above.
- **Random weights, our INT4 layout**: GPTQ-like groups of 128 with f16 scales and zeros; IREE's exact INT4 layout
  may differ.
- **Our hand-written tile kernel, not IREE codegen.** The question was the launch strategy, with the kernel held fixed.
- **One card, one driver and kernel version, one ROCm build**; the XTX was also driving the desktop.
- **Token and E2 times are event times** (they include the first launch's latency once per sequence). Single-kernel
  times are rocprofv3.

## Reproduce

```
./build.sh                       # needs a ROCm / TheRock with hipcc (ROCM=...)
GUARD=/path/to/a/gpu/watchdog ./run.sh smoke   # a second card first, small sizes (edit the device index in run.sh)
GUARD=... ./run.sh xtx           # the measurements; results/<date>-xtx/ with tables.md
```

`decode_floor check | shapes | sweep | sync | pshapes | token` run the pieces directly; `--help`-style usage is printed
without arguments. MIT licence.
