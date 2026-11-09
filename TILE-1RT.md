# One round trip per tile (step 1 of the tile work): plan and predictions

Written 2026-11-09, before any measurement. REPORT.md (tag `report-2026-11-09`) stays as measured; results of this test
go in their own results folders and a separate write-up.

## The change

The INT4 tile, built with `-DTILE_1RT` (`build/decode_floor_1rt`), issues every load of a pass before any arithmetic:
both rows' weights, scales and zeros for up to two 1,024-column chunks, and x once for the tile instead of once per row.
K <= 2048 is one pass (one dependent round trip per wave instead of 4 at K = 2048, 2 at K = 128 / 768); K = 4096 is two
passes (2 instead of 8). The arithmetic order per row is unchanged, so the outputs must be bitwise equal to the base
binary's. f16 is unchanged (the control). Batching a persistent block's several tiles (the down stage) is step 2, not
this test.

## Method

- Base and 1rt binaries from the same commit; runs interleaved in one session (base, 1rt, base, 1rt, base, 1rt).
- Not compared with data/: the XTX drives the display again (it did not in the published runs).
- Persistent comparisons on the link-kept-awake arm only.
- Gates before any timing: `check` with the 1rt binary (incl. K = 4096 two-pass and K = 512 / 768 partial lanes), and
  the token outputs of both binaries bitwise equal (Y_DUMP), INT4 cold and warm; on the RX 7900 XT first.
- VGPRs of all five kernels recorded; k_launch should stay <= 96 (16 waves per SIMD). A persistent grid change is
  reported as part of the result.

## Predictions (each removed round trip ~0.44 us on an idle card: the floor table's second pair)

| Quantity | Base (published) | Predicted with 1rt |
|---|---|---|
| 1-block INT4 matvec, K = 128, cold | 2.32 us | 1.8-2.1 us |
| 1-block INT4 matvec, K = 2048, cold | 3.20 us | 1.8-2.2 us (about the K = 128 value) |
| E1 INT4 t0 (cold) | 4.38 us | 2.7-3.4 us |
| INT4 cold token, grouped graph | 3.68 ms | 3.00-3.45 ms (about -0.33 ms at idle latency, more if loaded latency is higher) |
| f16 rows (control) | - | within 1 % of the interleaved base |

## Decision rule

- Grouped-graph INT4 cold token less than 5 % faster than the interleaved base: the round-trip count is not the lever
  this benchmark suggested; do not port to K3 on this evidence.
- 5-10 %: worth checking K3's own per-stage times first.
- 10 % or more: port the pattern to K3's matvec kernels (after measuring where K3's time goes).

## Result of v1 (2026-11-09, recorded before v2 was measured)

Three interleaved base / 1rt pairs on the XTX (display attached, both builds from commit 1c64f0b): the 1rt tile was
slower everywhere in INT4 and unchanged in f16 (the control).

| Quantity | Base | 1rt v1 |
|---|---|---|
| 1-block INT4 matvec, K = 128, cold | 2.32 us | 2.60 us |
| 1-block INT4 matvec, K = 2048, cold | 3.28 us | 3.56 us |
| E1 INT4 t0 (wide range) | 4.39 us | 4.95 us |
| INT4 cold token, grouped graph | 3.683 ms | 4.517 ms (+22.6 %) |
| f16 cold token, grouped graph (control) | 8.830 ms | 8.836 ms |

By the decision rule: do not port v1. But v1 did not implement the design. Its generated code wraps each load in a
divergent branch and converts each row's scale and zero to float inside that branch, so it waits (`s_waitcnt
vmcnt(1)`) after each row's loads before issuing the next: the round trips stayed serial, with branch overhead added.
The design (one round trip per pass) is therefore still untested.

v2: every load issued unconditionally from a clamped valid address; scales and zeros kept as raw f16 until after the
single wait; bounds applied only in the arithmetic. Gates and predictions as above, unchanged; v2 must show a single
wait before the arithmetic in the generated code before it is timed.

## v2's pre-timing gate (recorded after the v2 runs, before scoring them)

The gate above says v2 must show a single wait before the arithmetic before it is timed. I checked only that the load
section compiled to straight-line code, then timed it. Read properly, the gate fails in part:
- 11 of the pass's 12 16-byte loads and 6 of its 8 scale/zero loads are issued before the first data wait.
- One row's weight, scale and zero loads for one chunk are sunk by the compiler into that row's `live` branch, after the
  chunk's x-sum arithmetic, and waited for there: a second dependent round trip for that row (base: 4 per wave at
  K = 2048; v2: 2).
- Two waits inside the load cluster are on loads issued before it (register reuse), not on the tile's own data.

So v2 tests "2 round trips instead of 4", not "1 instead of 4". The fix would be to keep the loads out of any branch
(mask dead rows in the arithmetic). The v2 timing is scored as it stands, with this label. One 1rt-v2 run
(20261109-135031) stopped at floor-cold when rocprofv3 segfaulted at startup (pool release, before the kernel ran); it
is excluded, so 1rt-v2 has 2 runs against base-v2's 3.

## v2 result (3 base / 2 v2 runs interleaved, display on the XTX; full table in results-v2-vs-base.md)

| Quantity | Base | v2 | Predicted (for 1 round trip) |
|---|---|---|---|
| 1-block INT4 matvec, K = 128, cold | 2.32 us | 2.44 us | 1.8-2.1: missed (slower) |
| 1-block INT4 matvec, K = 2048, cold | 3.24 us | 2.92 us | 1.8-2.2: missed |
| E1 INT4 t0 (wide range) | 4.38 us | 3.95 us | 2.7-3.4: missed |
| INT4 cold token, grouped graph | 3.683 ms | 3.431 ms (-6.8 %) | 3.00-3.45: held |
| f16 cold token, grouped graph (control) | 8.820 ms | 8.846 ms | within 1 %: held |

Per kernel: q|k|v -18 %, gate|up unchanged, o (K = 4096, two passes) +25 %, down (K = 768) +15 %. Decision rule:
5-10 % -> check K3 per-stage before porting. And v2 is the 2-round-trip version (gate above), so a true one-round-trip
tile, and a K = 4096 / K = 768 path that does not regress, are still untested.
