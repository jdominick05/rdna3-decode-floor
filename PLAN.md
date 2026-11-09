# Where a Qwen3-30B-A3B decode token's matvec time goes on an RX 7900 XTX

Status: plan, 2026-11-09. Nothing measured yet. Every number below that is not cited is a prediction, and is labelled as one.

## Question

On an RX 7900 XTX (gfx1100), how much of a Qwen3-30B-A3B decode token's matrix-vector time is the cost of moving its
weights, and how much is a fixed cost per kernel? Does running the token's matvecs inside one persistent kernel
remove that fixed cost?

## Background (from the IREE issues, all on a 7900 XTX)

- [iree#23507](https://github.com/iree-org/iree/issues/23507) (comment of 2026-04-15) measured single matvecs with
  rocprofv3 kernel timestamps. Small kernels sit at about 5 µs even when they move almost nothing.
  - IREE INT4: 1×128×2048 (the 30B router) takes 5.0 µs for 144 KB, which is 3.0 % of peak bandwidth.
  - IREE INT4: 1×768×2048 (a 30B expert's gate or up) takes 5.6 µs for 841 KB, 15.5 % of peak.
  - Large shapes run near bandwidth: the 30B lm_head reaches 74 % (IREE INT4) and 83 % (llama.cpp f16).
- The same comment shows llama.cpp f16 at 111–137 % of the 960 GB/s peak on 17–34 MB shapes. Repeated runs on the
  same weights are served from the 96 MB Infinity Cache. A real token never re-reads a weight, so cache state has to
  be controlled.
- [onnxruntime-ep-iree#34](https://github.com/iree-org/onnxruntime-ep-iree/issues/34) (2026-05-12): Qwen3-30B-A3B
  decode at 120 tok/s (8.3 ms per token) with hand-written INT4 kernels inserted into IREE, against llama.cpp's 99.4
  tok/s. Their gate+up+SwiGLU and down+reduce kernels already group the 8 experts.
- [iree#24165](https://github.com/iree-org/iree/issues/24165): wall-clock timings of about 37 µs per small matvec come
  from their benchmark tool. They are not kernel times, and this experiment does not use them.
- The iree#23507 kernel times were taken warm: "Both benchmarks run thousands of iterations, so L2/Infinity Cache is
  warm" (same comment). E0 compares them with our warm runs; our cold runs serve the floor question.

## Model and formats

Qwen3-30B-A3B: 48 layers, hidden 2048, 32 query heads and 4 KV heads of 128, 128 routed experts with 8 used per
token, expert width 768, no shared expert, vocabulary 151,936.

Per token and layer, the matvecs are:

| Matvec | Shape (M×N×K) | Count per layer |
|---|---|---|
| q_proj | 1×4096×2048 | 1 |
| k_proj, v_proj | 1×512×2048 | 2 |
| o_proj | 1×2048×4096 | 1 |
| router | 1×128×2048 | 1 |
| expert gate, up | 1×768×2048 | 16 (8 experts × 2) |
| expert down | 1×2048×768 | 8 |

That is 29 matvecs per layer, 1,392 for the 48 layers, plus the lm_head (1×151936×2048): 1,393 per token.

Weights use their two formats:
- **f16** (their Table 1). The 128 experts × 48 layers do not fit in 24 GB at f16, so the f16 runs allocate only the
  8 experts a token uses. The f16 token still reads about 6.1 GB (3.6 GB of it experts), far above the cache.
- **INT4 with groups of 128, f16 scales** (the GPTQ format they use; their Table 2). All 128 experts of every layer fit
  (about 15 GB). The routing is random per token, as in real decoding.

Weights are random values in the real shapes and formats. Timing doesn't depend on the values, and their
microbenchmarks did the same.

Out of scope: attention, the norms, RoPE, sampling and the top-k selection. This measures the matvec part of the
token only, and the report will say so.

**What the token numbers can be compared with.** The "token" here is the 1,393 matvecs, not a decode token. Its times
are compared only between the launch modes of this benchmark (per-op, grouped, graph, persistent), which run the same
matvec code on the same weights. They are never compared with whole-token speeds such as IREE's 120 tok/s or
llama.cpp's 99.4 tok/s, which also include attention, norms, routing and sampling: a matvec-only time of a few ms
would look several times faster than those without being so. The one outside comparison allowed is per kernel: our
single-dispatch times against the per-kernel times in iree#23507 (E0), shape for shape, cold cache, rocprofv3.

## Hypotheses

- **H1 (floor):** a small matvec's ~5 µs kernel time is mostly fixed per-kernel cost: wave launch ramp, first-load
  latency, the kernel's tail and its end-of-kernel cache writeback. Its bytes take a fraction of that.
- **H2 (persistent):** as a task inside one persistent kernel, the same matvec costs about its bytes ÷ effective
  bandwidth plus a smaller fixed cost (the device-wide sync and first-load latency). In-kernel prefetch of the next
  task can hide part of the latency.

## Predictions (written before any measurement)

INT4 bytes per token: about 30.5 MB per layer (q 4.5, k and v 0.56 each, o 4.5, router 0.14, 8 experts × 3 × 0.84),
so 1.46 GB for the layers plus 0.17 GB for the lm_head: **1.63 GB per token**.

- **P1:** single-dispatch kernel time follows t = t0 + bytes ÷ BW_eff, with t0 of 3–5 µs and BW_eff of 700–850 GB/s
  for data that is not in cache.
- **P2:** inside the persistent kernel, the same fit gives t0' of 1–3 µs per dependent stage.
- **P3, the token (INT4, matvecs only):**
  - **A, one dispatch per matvec:** about the sum of single-kernel times. If our kernels match IREE's Table 2, that is
    7–8 ms (48 × 154.8 µs + 231.8 µs = 7.66 ms, summed from their per-kernel times; matvecs only, like ours).
  - **A', experts grouped** (8 experts' gate+up in one dispatch, the 8 downs in one, like their extern kernels): lower
    than A. How much lower is the grouping gain, measured separately from persistence.
  - **B, A' as a HIP graph:** lower than A' by the launch gaps.
  - **C, persistent:** about 1.63 GB ÷ 780 GB/s = 2.1 ms, plus 5 dependent stages per layer × 48 × t0'
    (0.24–0.72 ms). Predicted **2.3–2.9 ms**.
- **Refutation:**
  - If C is no more than 1.25× faster than B, this implementation did not deliver the predicted gain. That alone does
    not show the floor is unrelated to launches: synchronization and lost parallelism inside the persistent kernel can
    consume the saving. The report then breaks the persistent stage time down (sync alone, per-stage stamps) instead.
  - If C beats the bandwidth bound (1.63 GB ÷ 960 GB/s = 1.7 ms), the cache control failed and the run is invalid.

## Method

### Controls
- **Same matvec code in every mode.** A, A', B and C call the same device functions per tile; only the launch
  strategy changes. Checked in the generated gfx1100 code: the weight loads are `global_load_b128` in every kernel
  (explicit global address-space pointers; without them the persistent kernel got `flat_load`), and the matvec loops
  have the same instruction mix (57 `v_dot2acc_f32_f16`, 32 `v_pk_add_f16` per kernel) up to scheduling: per loop body
  of 19–156 instructions, at most one instruction differs plus `s_delay_alu` hints.
- **Cache state.**
  - "Cold" (the default): weights rotate through buffers totalling more than 4× the 96 MB Infinity Cache, so nothing
    is reused between runs. The token runs already read 1.6 GB (INT4) and 6.1 GB (f16) each.
  - "Warm": the same buffer repeated, reported separately only to reproduce their >100 % effect.
- **Their measurement tool.** Per-kernel times come from rocprofv3 `--kernel-trace` GPU timestamps, as in
  iree#23507. Inside the persistent kernel, per-task times come from the GPU's realtime counter (`s_memrealtime`,
  100 MHz); their sum is checked against the kernel's rocprof duration.
- **Clocks and repeats.** Clocks are fixed high for the run, with SCLK and MCLK recorded. Strategies are interleaved
  (A A' B C, repeated at least 5 times) after warmup; each result is a median with its spread.
- **Same card and software for everything.** One RX 7900 XTX, one build, one ROCm version, all recorded.

### Correctness first
Every kernel's output is compared with a reference before anything is timed: the INT4 dequantization against a
reference dequantization exactly, then the matvec within 1e-2 relative to an f32 reference. A mode that fails is
not timed.

### Steps
- **E0 (gate):** our standalone kernels on their Table 1 and Table 2 shapes, cold and warm, timed with rocprofv3.
  - If they come within about 2× of IREE's per-shape times, the comparison is on comparable kernels.
  - If not, we stop and report that first. A kernel far faster than theirs is a codegen result, not a dispatch result.
- **E1 (floor):** an N sweep at K=2048 (N from 32 to 8192) and the token's shapes, single dispatch, cold. Fit t0 and
  BW_eff (tests P1).
- **E2 (in-kernel cost):** the same shapes as tasks in one persistent kernel with a device-wide sync between them.
  Fit t0' (tests P2).
- **E3 (token):** the 1,393-matvec INT4 token in modes A, A', B and C, then the f16 token in the same modes (tests
  P3).
- **E4 (report):** a write-up of the hypotheses, the predictions and the measurements side by side, including the
  ones that failed.

## Safety
Every GPU process runs under the Caminus guard (VRAM lines, kernel-log fault watch). New kernels run first on the RX
7900 XT alone with small inputs, and only then on the XTX, the card the comparison needs. Every device-side wait in
the persistent kernel has a time limit and sets an error flag instead of spinning forever. The INT4 runs use about 15
GB of the XTX, under its guard line, with 512 MiB kept free for the display.

## Deliverables
This folder: the HIP source, build and run scripts, the raw results (CSV and JSON), and REPORT.md. Nothing is
published until the report exists; whether it becomes public, under which licence, and the IREE comment are the
owner's decisions.

## Revisions after the XT smoke (2026-11-09, before any XTX measurement)

Everything in this section was decided after looking at RX 7900 XT smoke runs (2 layers, small sizes) and before any
measurement on the XTX. The predictions below are written before the XTX run.

### What the smoke showed (XT, not reported as results)
- Grouping the experts mattered most: INT4 token (2 layers) 0.63 ms per-op, 0.38 ms grouped.
- A first persistent kernel (one device-wide barrier, 8 blocks per WGP) was no faster than grouped: 0.40 ms.
- A barrier over empty stages costs 1.1 µs at 1 block per WGP and 2.8 µs at 8: one shared counter, ~all blocks arrive.

### Design changes
- **Four persistent variants over the same tile code** (`k_persistent<PF>`, `k_dataflow<PF>`):
  - *barrier*: static round-robin tiles, a split-phase device-wide barrier between stages;
  - *dataflow*: per-stage claim and done counters; a block claims a chunk of a stage (chunk = ceil(tiles / grid), so
    at most one chunk per block), waits only for the stage before, publishes with a release; no barrier;
  - *-pf*: the claimed or assigned tile's weights are loaded before the wait (between arrive and wait for the barrier).
    Only for stages whose weights are known before the dependency finishes: never the routed experts' gate|up (the
    router picks them); their down, and all static weights, yes.
- **Grid**: one grid for all four, the smallest of their occupancy limits; swept at 1, 2, 4 blocks per WGP and the limit.
- **Warm / cold A/B**: warm E0 and E2 (one copy reused) and a warm token (every layer reuses one layer's weights and
  8 experts, INT4 about 31 MB, no vocab head), next to the cold ones.
- **Graph baseline in E2**: the same S launches captured in a HIP graph.

### Method fixes (from the smoke and two outside reviews)
- **The timeout clock starves waves.** The waits read the wall clock (`s_sendmsg_rtn` GET_REALTIME) on every poll; with
  ~150 waiting waves, one wave's reply stalled for 200 ms while the others kept asking (measured: the frozen wave's gap
  between two loop iterations was 199.99 ms; its neighbours polled 620,000 times). With the clock read once per 4,096
  polls, all four variants passed 4 of 4 where three had passed 0 of 4. Poll style (acquire, relaxed + fence, atomic
  add 0) and sleep length made no difference.
- Every polled or counted word sits on its own 128-byte line; waits poll with acquire loads.
- The occupancy API's count is not a residency guarantee in general; a residency probe (an empty-stage barrier) is run at
  every grid used, and the barrier variants are only used where it passes.
- **Dependency check** (correctness runs): every producer adds (tile + 1) per finished tile to a per-stage sum before it
  publishes; every consumer checks the complete sum after its wait.
- **Cold traversal**: no copy cap (too little memory refuses), one advancing cursor across warmup and timed runs, rings
  that never wrap inside a run; every row records the copies and the reuse distance (at least 400 MB on the XT).
- Stage stamps: per-stage end = atomic max of its chunks' finish times, taken before each release; start = atomic min
  over all blocks; a backwards stamp is counted and excluded. Non-finite outputs fail the numeric check. The warm token
  compares only what the plan writes. A failing variant is recorded as "fail" and the run goes on.

### Predictions for the XTX (written before the run)
- **sync, one-tile stage**: barrier 3.5–4.5 µs, growing with the grid; dataflow 2.8–3.5 µs; dataflow-pf 2.0–2.6 µs;
  barrier-pf no better than barrier (block 0 holds every tile, nothing to overlap). Empty-stage barrier 1–2 µs.
- **E2, small shapes (< 1 MB)**: dataflow-pf below launches and below the graph by at least 1.5 µs per matvec; for
  shapes over 4 MB all modes within 15 % of each other (bandwidth-bound).
- **E3, the 48-layer INT4 token (241 dependent stages, 1.63 GB)**:
  - per-op 7–9 ms; grouped and grouped-graph 2.9–3.6 ms.
  - The best persistent variant 2.6–3.4 ms: about 2.2 ms of weight traffic at 750 GB/s plus 241 stages × 1.5–4.5 µs.
  - Revised from P3's 2.3–2.9 ms: the smoke showed a per-stage cost of 2–4.5 µs inside the persistent kernel, more
    than the 1–3 µs of P2.
- **Warm token**: the gap between grouped-graph and the best persistent variant at least as large as cold, in µs per
  stage (no DRAM latency to hide behind).
