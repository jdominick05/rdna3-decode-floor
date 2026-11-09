// SPDX-License-Identifier: MIT
// Copyright (c) 2026 jdominick05
//
// decode_floor: where a Qwen3-30B-A3B decode token's matvec time goes on an RX 7900 XTX (gfx1100). See PLAN.md.
//
// One tile function computes every matvec (y[rows] = W[rows x K] . x[K], f32 accumulation) in every mode, so the
// modes differ only in how the tiles are launched:
//   per-op        one kernel launch per matvec
//   per-op-graph  the same launches captured in a HIP graph
//   grouped       one launch per dependent stage (q|k|v, o, router, 8 experts' gate|up, 8 experts' down)
//   grouped-graph the grouped launches in a HIP graph
//   persistent    one launch per token; the stages run in order with a device-wide sync between them
// Weights: f16 [rows][K], or INT4 groups of 128 columns: packed bytes [rows][K/2] (low nibble = even column), an f16
// scale and an f16 zero point per row and group; w = s * (q - z).
//
// usage: decode_floor MODE [options]
//   check                 correctness of the tile function against a naive reference (f16 and INT4, every shape)
//   shapes   --fmt F      one kernel per shape, repeated (E0 / E1): per-launch event times (rocprofv3 for kernel times)
//   pshapes  --fmt F      each shape as STAGES stages of one persistent kernel, and as STAGES back-to-back launches (E2)
//   sync                  the persistent kernel's per-stage cost with 0 and 1 tiles per stage, against --grid-per-cu
//   token    --fmt F      the Qwen3-30B-A3B decode token's matvecs in every mode (E3); --warm: every layer reuses one
//                         layer's weights and experts (INT4 about 31 MB, in cache), no vocab head
// options: --fmt f16|i4  --warm (reuse one weight copy; default cold: rotate copies totalling > 4 x 96 MB)
//          --reps N  --stages N  --layers N (token: fewer layers for a smoke run)  --tokens N (routing samples)
//          --csv FILE (append result rows)
//          --expect-cus N (refuse any other device; HIP counts WGPs on RDNA3: XTX 48, XT 42)  --grid-per-cu N (persistent blocks per CU; default the occupancy max)
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    fprintf(stderr, "HIP error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, hipGetErrorString(e_)); exit(2); } } while (0)

enum { FMT_F16 = 0, FMT_I4 = 1 };
constexpr int WG = 256;                 // 8 waves of 32
constexpr int RPW = 2;                  // rows per wave
constexpr int TILE_ROWS = (WG / 32) * RPW;
constexpr int GROUP = 128;              // INT4 group size (columns)
constexpr uint64_t CACHE_BYTES = 96ull << 20;   // the 7900 XTX's Infinity Cache

typedef _Float16 h2 __attribute__((ext_vector_type(2)));
// the matvec reads through explicit global-address-space pointers: without them the compiler emits flat loads in the
// persistent kernel (pointers reached through the stage table) and global loads in the per-launch kernel, and the
// two modes would no longer run the same instructions
#define GLOBAL __attribute__((address_space(1)))
template <class T> __device__ __forceinline__ const GLOBAL T* gp(const T* p) { return (const GLOBAL T*)p; }
template <class T> __device__ __forceinline__ GLOBAL T* gpw(T* p) { return (GLOBAL T*)p; }

struct Seg {                            // one matvec
    const void* W; const __half* S; const __half* Z; const __half* x; float* y;
    int rows, K, fmt, tile0;            // tile0: this segment's first tile within its launch / stage
};
struct Stage { int seg0, nseg, ntiles, pf, chunk; };   // pf: the stage's weights are known before its dependency finishes;
                                                       // chunk: tiles per dataflow claim (set by the host, set_chunks)
                                                // (static weights, or experts already routed), so they may be prefetched
// Every polled or counted word sits on its own 128-byte cache line: with the barrier's count and generation on one line,
// ~170 waves polling it starved the blocks that still had to arrive (XT, 4 blocks per WGP: 200 ms stalls, PLAN.md E2).
constexpr int LINE = 32;                       // unsigned words per 128-byte line
#define AT(a, i) ((a)[(size_t)(i) * LINE])     // the per-stage counter i, one line each
struct Sync { unsigned count; unsigned p0[LINE - 1]; unsigned gen; unsigned p1[LINE - 1]; int err; int pad; int dbg[4]; int p2[LINE - 6]; };
#ifdef DF_TRACE
constexpr bool TRACE = true;                   // (diagnosis build: per-block barrier progress and timestamps)
#else
constexpr bool TRACE = false;
#endif

__device__ __forceinline__ h2 as_h2(uint32_t u) { return __builtin_bit_cast(h2, u); }

__device__ __forceinline__ float wave_sum(float a) {
    for (int o = 16; o > 0; o >>= 1) a += __shfl_xor(a, o, 32);
    return a;
}

typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));    // (native vectors: HIP's uint4 / __half classes
                                                                   //  have no address-space-qualified copies)
// One lane-chunk of a row, shared by the plain and the prefetching paths so both run the same arithmetic in the same order.
// f16: 8 columns, acc += w . x
__device__ __forceinline__ float f16_chunk(u32x4 wv, const GLOBAL _Float16* x, int c, float acc) {
    const u32x4 xv = *reinterpret_cast<const GLOBAL u32x4*>(x + c);
    acc = __builtin_amdgcn_fdot2(as_h2(wv.x), as_h2(xv.x), acc, false);
    acc = __builtin_amdgcn_fdot2(as_h2(wv.y), as_h2(xv.y), acc, false);
    acc = __builtin_amdgcn_fdot2(as_h2(wv.z), as_h2(xv.z), acc, false);
    acc = __builtin_amdgcn_fdot2(as_h2(wv.w), as_h2(xv.w), acc, false);
    return acc;
}
// INT4: 32 columns (16 bytes of nibbles), always inside one group of 128: acc += s * (sum q.x - z * sum x).
// A nibble q becomes the f16 1024 + q (exponent bits 0x6400 | q), minus 1024 in one packed subtract.
__device__ __forceinline__ float i4_chunk(u32x4 wv, float sc, float zp, const GLOBAL _Float16* x, int c, float acc) {
    const h2 off = {(_Float16)1024.0f, (_Float16)1024.0f}, one = {(_Float16)1.0f, (_Float16)1.0f};
    const GLOBAL u32x4* xp = reinterpret_cast<const GLOBAL u32x4*>(x + c);
    const uint32_t wq[4] = {wv.x, wv.y, wv.z, wv.w};
    float aq = 0.f, ax = 0.f;
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const u32x4 xv = xp[j];
        const uint32_t xw[4] = {xv.x, xv.y, xv.z, xv.w};
        const uint32_t q = wq[j];
#pragma unroll
        for (int k = 0; k < 4; k++) {
            const uint32_t bits = 0x64006400u | ((q >> (8 * k)) & 0xFu) | (((q >> (8 * k + 4)) & 0xFu) << 16);
            aq = __builtin_amdgcn_fdot2(as_h2(bits) - off, as_h2(xw[k]), aq, false);
            ax = __builtin_amdgcn_fdot2(as_h2(xw[k]), one, ax, false);
        }
    }
    return acc + sc * (aq - zp * ax);
}

// Prefetch: the first NP lane-chunks of each of the wave's RPW rows of a tile, loaded into registers before the
// dependency wait (INT4 K=2048: the whole row; f16 K=2048: a quarter of it). Activations are never prefetched.
constexpr int NP = 2;
struct Pre { u32x4 w[RPW][NP]; float s[RPW][NP], z[RPW][NP]; };

struct RowPtr { const GLOBAL _Float16* wh; const GLOBAL uint8_t* wq; const GLOBAL _Float16* S; const GLOBAL _Float16* Z; };
__device__ __forceinline__ RowPtr row_ptr(const Seg& s, int r) {
    RowPtr p; const int ng = s.K / GROUP;
    p.wh = gp(static_cast<const _Float16*>(s.W)) + (size_t)r * s.K;
    p.wq = gp(static_cast<const uint8_t*>(s.W)) + (size_t)r * (s.K / 2);
    p.S = gp(reinterpret_cast<const _Float16*>(s.S)) + (size_t)r * ng;
    p.Z = gp(reinterpret_cast<const _Float16*>(s.Z)) + (size_t)r * ng;
    return p;
}

// one row: chunks [0, NP) from the prefetched registers when pre != nullptr, the rest loaded here; the same chunk order
// and arithmetic either way (so prefetching never changes a result bit)
template <bool PF>
__device__ __forceinline__ float dot_row(const Seg& s, int r, int lane, const Pre* pre, int i) {
    const GLOBAL _Float16* x = gp(reinterpret_cast<const _Float16*>(s.x));
    const RowPtr rp = row_ptr(s, r);
    float acc = 0.f;
    if (s.fmt == FMT_F16) {
        int c = lane * 8, j = 0;
        if constexpr (PF) {
#pragma unroll
            for (; j < NP; j++, c += 32 * 8) if (c < s.K) acc = f16_chunk(pre->w[i][j], x, c, acc);
        }
        for (; c < s.K; c += 32 * 8) acc = f16_chunk(*reinterpret_cast<const GLOBAL u32x4*>(rp.wh + c), x, c, acc);
    } else {
        int c = lane * 32, j = 0;
        if constexpr (PF) {
#pragma unroll
            for (; j < NP; j++, c += 32 * 32) if (c < s.K) acc = i4_chunk(pre->w[i][j], pre->s[i][j], pre->z[i][j], x, c, acc);
        }
        for (; c < s.K; c += 32 * 32)
            acc = i4_chunk(*reinterpret_cast<const GLOBAL u32x4*>(rp.wq + c / 2), (float)rp.S[c / GROUP], (float)rp.Z[c / GROUP], x, c, acc);
    }
    return acc;
}

__device__ __forceinline__ void prefetch_rows(const Seg& s, int t, Pre& p) {
    const int wave = threadIdx.x >> 5, lane = threadIdx.x & 31;
#pragma unroll
    for (int i = 0; i < RPW; i++) {
        const int r = t * TILE_ROWS + wave * RPW + i;
        const RowPtr rp = row_ptr(s, r < s.rows ? r : 0);
#pragma unroll
        for (int j = 0; j < NP; j++) {
            p.w[i][j] = u32x4{0u, 0u, 0u, 0u}; p.s[i][j] = 0.f; p.z[i][j] = 0.f;
            if (r >= s.rows) continue;
            if (s.fmt == FMT_F16) {
                const int c = lane * 8 + j * 32 * 8;
                if (c < s.K) p.w[i][j] = *reinterpret_cast<const GLOBAL u32x4*>(rp.wh + c);
            } else {
                const int c = lane * 32 + j * 32 * 32;
                if (c < s.K) { p.w[i][j] = *reinterpret_cast<const GLOBAL u32x4*>(rp.wq + c / 2); p.s[i][j] = (float)rp.S[c / GROUP]; p.z[i][j] = (float)rp.Z[c / GROUP]; }
            }
        }
    }
}

template <bool PF>
__device__ __forceinline__ void do_tile(const Seg& s, int t, const Pre* pre) {
    const int wave = threadIdx.x >> 5, lane = threadIdx.x & 31;
#pragma unroll
    for (int i = 0; i < RPW; i++) {
        const int r = t * TILE_ROWS + wave * RPW + i;
        if (r >= s.rows) break;                      // (uniform within the wave)
        float a = wave_sum(dot_row<PF>(s, r, lane, pre, i));
        if (lane == 0) gpw(s.y)[r] = a;
    }
}

__device__ __forceinline__ int ntiles_of(int rows) { return (rows + TILE_ROWS - 1) / TILE_ROWS; }

__device__ __forceinline__ int find_seg(const Seg* segs, int nseg, int g) {
    for (int i = 0; i < nseg; i++)                   // (a launch / stage holds at most a few dozen segments)
        if (g >= segs[i].tile0 && g < segs[i].tile0 + ntiles_of(segs[i].rows)) return i;
    return -1;
}
__device__ __forceinline__ void run_tile(const Seg* segs, int nseg, int g) {
    const int i = find_seg(segs, nseg, g);
    if (i >= 0) do_tile<false>(segs[i], g - segs[i].tile0, nullptr);
}
__device__ __forceinline__ void prefetch_tile(const Seg* segs, int nseg, int g, Pre& p) {
    const int i = find_seg(segs, nseg, g);
    if (i >= 0) prefetch_rows(segs[i], g - segs[i].tile0, p);
}
__device__ __forceinline__ void run_tile_pf(const Seg* segs, int nseg, int g, const Pre& p) {
    const int i = find_seg(segs, nseg, g);
    if (i >= 0) do_tile<true>(segs[i], g - segs[i].tile0, &p);
}

// a launch: block b computes tile base + b of the segments [segs, segs + nseg)
__global__ __launch_bounds__(WG) void k_launch(const Seg* segs, int nseg, int base) {
    run_tile(segs, nseg, base + (int)blockIdx.x);
}

// How waiting threads poll (PLAN.md E2): POLL 0 = acquire loads (each invalidates the near caches), 1 = relaxed loads and
// one acquire fence after, 2 = atomic add of 0 (served at L2) and one acquire fence after; SLEEPN = s_sleep units between
// polls (64 clocks each). On the XT, acquire polls with SLEEPN 1 froze a waiting wave for 200 ms while ~150 others polled.
#ifndef POLL
#define POLL 0
#endif
#ifndef SLEEPN
#define SLEEPN 1
#endif
// CLOCKEVERY: the timeout clock (s_sendmsg_rtn GET_REALTIME, through the message path) is read once per CLOCKEVERY polls.
// Read on every poll by ~150 waiting waves, it starved some waves' message replies for as long as the others kept asking:
// on the XT, 0 of 4 runs passed for barrier-pf, dataflow and dataflow-pf; once per 4096 polls, 4 of 4 for all (PLAN.md E2).
#ifndef CLOCKEVERY
#define CLOCKEVERY 4096
#endif
__device__ __forceinline__ unsigned poll_u32(unsigned* p) {
    if constexpr (POLL == 0) return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
    else if constexpr (POLL == 1) return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    else return __hip_atomic_fetch_add(p, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
__device__ __forceinline__ void poll_done() { if constexpr (POLL != 0) __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent"); }

// A split-phase device-wide barrier with a time limit. arrive: publish this block's work (fence) and count it in;
// wait: spin until the last block has arrived (on timeout: the error flag, and the kernel gives up; it never spins forever).
// Loads issued between the two overlap the wait. (They cannot go before arrive: its fence waits for every outstanding
// load, so a prefetch issued before it would be waited for in full.)
__device__ __forceinline__ void grid_arrive(Sync* sy, unsigned nb, unsigned* sh_gen, int* trace = nullptr) {
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned g = __hip_atomic_load(&sy->gen, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        const unsigned arrived = __hip_atomic_fetch_add(&sy->count, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
        if (arrived == nb - 1) {
            __hip_atomic_store(&sy->count, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            __hip_atomic_fetch_add(&sy->gen, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        }
        *sh_gen = g;
        if (TRACE && trace) { trace[0] = (int)g; trace[1] = (int)arrived; *reinterpret_cast<uint64_t*>(trace + 2) = wall_clock64(); }
    }
}
__device__ __forceinline__ bool grid_wait(Sync* sy, uint64_t limit, const unsigned* sh_gen, int* sh_bad, int stage, int* trace = nullptr) {
    if (threadIdx.x == 0) {
        *sh_bad = 0;
        const uint64_t t0 = wall_clock64();
        // acquire polls (each one invalidates the near caches): with relaxed polls and one fence after, a poller kept
        // reading a stale copy and dataflow timed out 3 of 3 on the XT (PLAN.md E2)
        unsigned it = 0;
        while (poll_u32(&sy->gen) == *sh_gen) {
            if ((++it % CLOCKEVERY) == 0 && wall_clock64() - t0 > limit) {
                *sh_bad = 1; __hip_atomic_store(&sy->err, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
                if (TRACE && trace) *reinterpret_cast<uint64_t*>(trace + 4) = wall_clock64();
                sy->dbg[0] = stage; sy->dbg[1] = (int)*sh_gen; sy->dbg[2] = (int)__hip_atomic_load(&sy->count, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT); sy->dbg[3] = blockIdx.x;
                break; }
            __builtin_amdgcn_s_sleep(SLEEPN);
        }
        poll_done();
    }
    __syncthreads();
    return !*sh_bad;
}

// The dependency check (validate = 1, the correctness runs only): every producer adds (tile + 1) for each tile it finished
// to vsum[i] before it publishes (the barrier, or its done-count release); a consumer of stage i + 1 reads vsum[i] after
// its wait and needs the full n (n + 1) / 2. A short sum means some producer's writes were not visible after the wait
// (a missing release / acquire on that path): err = 2. (The stages pass no data, so the bitwise output check cannot.)
__device__ __forceinline__ void vsum_add(unsigned* vsum, int i, unsigned add) {
    if (threadIdx.x == 0 && add) __hip_atomic_fetch_add(&AT(vsum, i), add, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
__device__ __forceinline__ void vsum_check(const unsigned* vsum, int i, int ntiles, int* err) {
    const unsigned want = (unsigned)ntiles * (unsigned)(ntiles + 1) / 2u;
    if (threadIdx.x == 0 && __hip_atomic_load(&AT(vsum, i), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) != want)
        __hip_atomic_store(err, 2, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
// the kernel's start: the earliest block's first instruction (stamps[nstages], set to ~0 by the host before each run)
__device__ __forceinline__ void stamp_start(uint64_t* stamps, int nstages) {
    if (threadIdx.x == 0) atomicMin((unsigned long long*)&stamps[nstages], (unsigned long long)wall_clock64());
}

// The persistent kernel, barrier variant: every stage's tiles round-robin over the resident blocks (block b: tiles b,
// b + grid, ...), then a device-wide barrier. PF: a block issues the loads of its first tile of stage i (if the stage
// allows it) between arriving at the barrier that ends stage i - 1 and waiting on it. Every block must be resident.
// stamps[i] (block 0): the wall clock after stage i's closing barrier (ordered by the barriers)
// (PF is a template parameter: the variants without prefetch compile without its registers, as a plain kernel would)
template <bool PF>
__global__ __launch_bounds__(WG) void k_persistent(const Stage* stages, int nstages, const Seg* segs, Sync* sy,
                                                   uint64_t limit, uint64_t* stamps, unsigned* vsum, int* prog, int validate) {
    __shared__ unsigned sh_gen; __shared__ int sh_bad;
    stamp_start(stamps, nstages);
    const int g0 = blockIdx.x;
    for (int i = 0; i <= nstages; i++) {             // (i == nstages: only the closing barrier)
        const bool last = i == nstages;
        const Stage st = last ? Stage{0, 0, 0, 0, 1} : stages[i];
        if (i > 0) { grid_arrive(sy, gridDim.x, &sh_gen, prog + 4096 + 6 * blockIdx.x); if (TRACE && threadIdx.x == 0) __hip_atomic_store(&prog[blockIdx.x], i * 4 + 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT); }
        const bool pf = PF && !last && st.pf && g0 < st.ntiles;
        Pre p;
        if constexpr (PF) { if (pf) prefetch_tile(segs + st.seg0, st.nseg, g0, p); }
        if (i > 0) {
            if (!grid_wait(sy, limit, &sh_gen, &sh_bad, i, prog + 4096 + 6 * blockIdx.x)) { if (TRACE && threadIdx.x == 0) __hip_atomic_store(&prog[blockIdx.x], i * 4 + 3, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT); return; }
            if (TRACE && threadIdx.x == 0) __hip_atomic_store(&prog[blockIdx.x], i * 4 + 2, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            if (blockIdx.x == 0 && threadIdx.x == 0) stamps[i - 1] = wall_clock64();
            if (validate && !last && g0 < st.ntiles && stages[i - 1].ntiles > 0) vsum_check(vsum, i - 1, stages[i - 1].ntiles, &sy->err);
        }
        if (last) break;
        int g = g0; unsigned add = 0;
        if constexpr (PF) { if (pf) { run_tile_pf(segs + st.seg0, st.nseg, g, p); add += g + 1; g += gridDim.x; } }   // (the prefetched first tile)
        for (; g < st.ntiles; g += gridDim.x) { run_tile(segs + st.seg0, st.nseg, g); add += g + 1; }
        if (validate) vsum_add(vsum, i, add);           // (before the next barrier's arrive, whose fence releases it)
    }
}

// The persistent kernel, dataflow variant (no barrier): a block claims a chunk of stage i's tiles (st.chunk tiles) from a
// counter (claim[i]); only a block holding a chunk waits, and only until done[i - 1] reaches stage i - 1's chunk count; a
// finished chunk adds one to done[i] (release). A block that gets no chunk moves on to the next stage. PF: the claimed
// chunk's first tile's loads are issued before the wait (if the stage allows it). Every stage needs at least one tile.
// stamps[i]: the latest finish of stage i's chunks, taken (atomic max) before each chunk's release, so it never trails
// the next stage's start
template <bool PF>
__global__ __launch_bounds__(WG) void k_dataflow(const Stage* stages, int nstages, const Seg* segs, unsigned* claim,
                                                 unsigned* done, unsigned* vsum, int* err, uint64_t limit,
                                                 uint64_t* stamps, int* prog, int validate) {
    __shared__ int sh_t, sh_bad;
    stamp_start(stamps, nstages);
    for (int i = 0; i < nstages; i++) {
        const Stage st = stages[i];
        const int nch = (st.ntiles + st.chunk - 1) / st.chunk;
        for (;;) {
            if (threadIdx.x == 0) sh_t = (int)__hip_atomic_fetch_add(&AT(claim, i), 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            __syncthreads();
            const int c = sh_t;
            __syncthreads();
            if (c >= nch) break;
            const int t0c = c * st.chunk, t1c = min(st.ntiles, t0c + st.chunk);
            if (TRACE && threadIdx.x == 0) __hip_atomic_store(&prog[blockIdx.x], (i << 8) | 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);   // claimed
            const bool pf = PF && st.pf;
            Pre p;
            if constexpr (PF) { if (pf) prefetch_tile(segs + st.seg0, st.nseg, t0c, p); }
            if (i > 0) {
                const Stage sp = stages[i - 1];
                if (threadIdx.x == 0) {
                    sh_bad = 0;
                    const unsigned target = (unsigned)((sp.ntiles + sp.chunk - 1) / sp.chunk);
                    const uint64_t t0 = wall_clock64();
                    uint64_t tprev = t0; unsigned polls = 0;
                    while (poll_u32(&AT(done, i - 1)) < target) {
                        if ((++polls % CLOCKEVERY) != 0) { __builtin_amdgcn_s_sleep(SLEEPN); continue; }
                        const uint64_t tnow = wall_clock64();
                        if (TRACE) { prog[8192 + 2 * blockIdx.x] = (int)(tnow - tprev); prog[8192 + 2 * blockIdx.x + 1] = (int)polls; }
                        tprev = tnow;
                        if (tnow - t0 > limit) {
                            sh_bad = 1; __hip_atomic_store(err, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
                            if (TRACE) { int* r = prog + 4096 + 4 * blockIdx.x; r[0] = i; r[1] = (int)__hip_atomic_load(&AT(done, i - 1), __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
                                         r[2] = (int)target; r[3] = (int)(((uintptr_t)&AT(done, i - 1) - (uintptr_t)done) / 4); }
                            err[1] = i; err[2] = (int)__hip_atomic_load(&AT(done, i - 1), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT); err[3] = (int)target;
                            err[4] = (int)__hip_atomic_load(&AT(claim, i - 1), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT); err[5] = blockIdx.x;
                            break; }
                        __builtin_amdgcn_s_sleep(SLEEPN);
                    }
                    poll_done();
                }
                __syncthreads();
                if (sh_bad) return;
                if (TRACE && threadIdx.x == 0) __hip_atomic_store(&prog[blockIdx.x], (i << 8) | 2, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);   // wait over
                if (validate) vsum_check(vsum, i - 1, sp.ntiles, err);
            }
            unsigned add = 0;
            int t = t0c;
            if constexpr (PF) { if (pf) { run_tile_pf(segs + st.seg0, st.nseg, t, p); add += t + 1; t++; } }
            for (; t < t1c; t++) { run_tile(segs + st.seg0, st.nseg, t); add += t + 1; }
            if (TRACE && threadIdx.x == 0) __hip_atomic_store(&prog[blockIdx.x], (i << 8) | 3, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);   // computed
            __threadfence();                          // every wave's outputs, before the release below
            __syncthreads();
            if (threadIdx.x == 0) {
                if (validate) __hip_atomic_fetch_add(&AT(vsum, i), add, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
                atomicMax((unsigned long long*)&stamps[i], (unsigned long long)wall_clock64());
                __hip_atomic_fetch_add(&AT(done, i), 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
                if (TRACE) __hip_atomic_store(&prog[blockIdx.x], (i << 8) | 4, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);   // published
            }
        }
    }
}

// the naive reference: one thread per row, f32, in column order
__global__ void k_reference(Seg s) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= s.rows) return;
    float acc = 0.f;
    if (s.fmt == FMT_F16) {
        const __half* w = static_cast<const __half*>(s.W) + (size_t)r * s.K;
        for (int c = 0; c < s.K; c++) acc += __half2float(w[c]) * __half2float(s.x[c]);
    } else {
        const uint8_t* w = static_cast<const uint8_t*>(s.W) + (size_t)r * (s.K / 2);
        const int ng = s.K / GROUP;
        for (int c = 0; c < s.K; c++) {
            const int q = (w[c / 2] >> ((c & 1) * 4)) & 0xF, g = c / GROUP;
            acc += __half2float(s.S[(size_t)r * ng + g]) * ((float)q - __half2float(s.Z[(size_t)r * ng + g])) * __half2float(s.x[c]);
        }
    }
    s.y[r] = acc;
}

// deterministic random fill (a hash of the index), no host copies
__device__ __forceinline__ uint32_t hash32(uint64_t i, uint32_t seed) {
    uint64_t z = i * 0x9E3779B97F4A7C15ull + seed;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull; z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return (uint32_t)(z ^ (z >> 31));
}
__global__ void k_fill_bytes(uint8_t* p, size_t n, uint32_t seed) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) p[i] = (uint8_t)hash32(i, seed);
}
__global__ void k_fill_half(__half* p, size_t n, uint32_t seed, float lo, float hi) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        p[i] = __float2half(lo + (hi - lo) * (hash32(i, seed) >> 8) * (1.0f / 16777216.0f));
}

// ---------------------------------------------------------------------------------------------------------- host
struct Tensor { void* W = nullptr; __half* S = nullptr; __half* Z = nullptr; int rows = 0, K = 0, fmt = 0; };

static size_t align_up(size_t v) { return (v + 255) & ~(size_t)255; }
static size_t tensor_bytes(int rows, int K, int fmt) {     // weight bytes the kernel reads (scales and zeros included)
    return fmt == FMT_F16 ? (size_t)rows * K * 2 : (size_t)rows * K / 2 + (size_t)rows * (K / GROUP) * 4;
}
static int host_ntiles(int rows) { return (rows + TILE_ROWS - 1) / TILE_ROWS; }

struct Arena {                                             // one allocation, carved
    uint8_t* base = nullptr; size_t cap = 0, used = 0;
    void init(size_t bytes, const char* what) {
        size_t freeb = 0, total = 0; CHECK(hipMemGetInfo(&freeb, &total));
        const size_t margin = (size_t)1536 << 20;        // leave room for the display and the runtime
        if (bytes + margin > freeb) { fprintf(stderr, "%s: needs %.2f GB, %.2f GB free (margin 1.5 GB): refused\n", what, bytes / 1e9, freeb / 1e9); exit(3); }
        CHECK(hipMalloc(&base, bytes)); cap = bytes;
    }
    void* take(size_t n) { n = align_up(n); if (used + n > cap) { fprintf(stderr, "arena overflow\n"); exit(3); } void* p = base + used; used += n; return p; }
    void release() { if (base) CHECK(hipFree(base)); base = nullptr; cap = used = 0; }
};
static size_t tensor_alloc_bytes(int rows, int K, int fmt) {
    return fmt == FMT_F16 ? align_up((size_t)rows * K * 2) : align_up((size_t)rows * K / 2) + 2 * align_up((size_t)rows * (K / GROUP) * 2);
}
static Tensor make_tensor(Arena& a, int rows, int K, int fmt, uint32_t seed) {
    Tensor t; t.rows = rows; t.K = K; t.fmt = fmt;
    if (fmt == FMT_F16) {
        t.W = a.take((size_t)rows * K * 2);
        hipLaunchKernelGGL(k_fill_half, dim3(1024), dim3(256), 0, 0, (__half*)t.W, (size_t)rows * K, seed, -0.05f, 0.05f);
    } else {
        t.W = a.take((size_t)rows * K / 2);
        t.S = (__half*)a.take((size_t)rows * (K / GROUP) * 2); t.Z = (__half*)a.take((size_t)rows * (K / GROUP) * 2);
        hipLaunchKernelGGL(k_fill_bytes, dim3(1024), dim3(256), 0, 0, (uint8_t*)t.W, (size_t)rows * K / 2, seed);
        hipLaunchKernelGGL(k_fill_half, dim3(256), dim3(256), 0, 0, t.S, (size_t)rows * (K / GROUP), seed + 1, 0.005f, 0.02f);
        hipLaunchKernelGGL(k_fill_half, dim3(256), dim3(256), 0, 0, t.Z, (size_t)rows * (K / GROUP), seed + 2, 7.0f, 9.0f);
    }
    CHECK(hipGetLastError());
    return t;
}
static Seg seg_of(const Tensor& t, const __half* x, float* y, int tile0) {
    Seg s; s.W = t.W; s.S = t.S; s.Z = t.Z; s.x = x; s.y = y; s.rows = t.rows; s.K = t.K; s.fmt = t.fmt; s.tile0 = tile0; return s;
}

struct Opts { std::string mode, fmt = "i4", csv; bool warm = false; int reps = 200, stages = 256, layers = 48, tokens = 8, expect_cus = 0, grid_per_cu = 0; };
static int g_grid_per_cu = 0;   // persistent blocks per CU (0: the occupancy maximum)
static int fmt_of(const std::string& f) { return f == "f16" ? FMT_F16 : FMT_I4; }

// a result row: printed, and appended to --csv; a new (empty) file gets the header first
static void csv_row(const Opts& o, const char* header, const std::string& row) {
    printf("%s\n", row.c_str());
    if (o.csv.empty()) return;
    FILE* f = fopen(o.csv.c_str(), "a");
    if (!f) return;
    if (ftell(f) == 0) fprintf(f, "%s\n", header);
    fprintf(f, "%s\n", row.c_str());
    fclose(f);
}
static double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v.empty() ? 0 : v[v.size() / 2]; }
static double pct(std::vector<double> v, double p) { std::sort(v.begin(), v.end()); return v.empty() ? 0 : v[std::min(v.size() - 1, (size_t)(p * v.size()))]; }

// the shapes from iree#23507 (Qwen MoE decode shapes, M = 1), as (N, K, label)
struct Shape { int N, K; const char* label; };
static const std::vector<Shape> TABLE_SHAPES = {
    {8192, 2048, "35B gdn_qkv_proj"}, {4096, 2048, "30B/35B q_proj"}, {2048, 4096, "30B/35B o_proj"},
    {512, 2048, "30B/35B k_proj v_proj"}, {2048, 512, "35B shared_down"}, {256, 2048, "35B router"}, {128, 2048, "30B router"},
    {32, 2048, "35B gdn_b gdn_a"}, {151936, 2048, "30B lm_head"}, {248320, 2048, "35B lm_head"}, {768, 2048, "30B moe gate/up"},
    {2048, 768, "30B moe down"}, {512, 2048, "35B moe gate/up"}};

// Weight copies of one shape. Cold: enough copies that they total more than 4 x the 96 MB Infinity Cache (no cap: too
// little memory is a refusal from the arena, never a quietly smaller set), at least `ring` (the copies one run reads)
// and a whole number of rings. Callers walk the copies in order with one advancing cursor (warmup included), so a copy
// is read again only after all n - 1 others. Warm: one copy, reused on purpose.
struct Copies { Arena a; std::vector<Tensor> t; size_t bytes = 0; };
static void make_copies(Copies& c, const Shape& sh, int fmt, bool warm, int ring = 1) {
    c.bytes = tensor_bytes(sh.N, sh.K, fmt);
    size_t n = 1;
    if (!warm) {
        n = (4 * CACHE_BYTES + c.bytes - 1) / c.bytes + 1;
        n = std::max(n, (size_t)ring);
        n = (n + ring - 1) / ring * ring;
    }
    c.a.init(tensor_alloc_bytes(sh.N, sh.K, fmt) * n + 4096, "weights");
    for (size_t i = 0; i < n; i++) c.t.push_back(make_tensor(c.a, sh.N, sh.K, fmt, 1000 + (uint32_t)i));
    CHECK(hipDeviceSynchronize());
}
// the bytes read between two reads of the same copy, walking the copies in order (rings consecutively)
static double reuse_mb(const Copies& c, int /*ring*/) { return c.t.size() <= 1 ? 0.0 : (double)(c.t.size() - 1) * c.bytes / 1e6; }

static int check_mode(const Opts& o) {
    int bad = 0;
    __half* x; float *y, *yr; const int Kmax = 4096, Nmax = 248320;
    CHECK(hipMalloc(&x, Kmax * 2)); CHECK(hipMalloc(&y, Nmax * 4)); CHECK(hipMalloc(&yr, Nmax * 4));
    hipLaunchKernelGGL(k_fill_half, dim3(16), dim3(256), 0, 0, x, (size_t)Kmax, 7, -1.f, 1.f);
    for (int fmt : {FMT_F16, FMT_I4}) for (const Shape& sh : TABLE_SHAPES) {
        Arena a; a.init(tensor_alloc_bytes(sh.N, sh.K, fmt) + 4096, "check");
        Tensor t = make_tensor(a, sh.N, sh.K, fmt, 42);
        Seg s = seg_of(t, x, y, 0), sr = seg_of(t, x, yr, 0);
        Seg* d; CHECK(hipMalloc(&d, sizeof(Seg))); CHECK(hipMemcpy(d, &s, sizeof(Seg), hipMemcpyHostToDevice));
        hipLaunchKernelGGL(k_launch, dim3(host_ntiles(sh.N)), dim3(WG), 0, 0, d, 1, 0);
        hipLaunchKernelGGL(k_reference, dim3((sh.N + 255) / 256), dim3(256), 0, 0, sr);
        CHECK(hipDeviceSynchronize());
        std::vector<float> hy(sh.N), hr(sh.N); CHECK(hipMemcpy(hy.data(), y, sh.N * 4, hipMemcpyDeviceToHost)); CHECK(hipMemcpy(hr.data(), yr, sh.N * 4, hipMemcpyDeviceToHost));
        // the reference itself against a double-precision host computation of two rows
        std::vector<__half> hx(sh.K); CHECK(hipMemcpy(hx.data(), x, sh.K * 2, hipMemcpyDeviceToHost));
        double host_err = 0;
        for (int r : {0, sh.N - 1}) {
            double acc = 0;
            if (fmt == FMT_F16) {
                std::vector<__half> w(sh.K); CHECK(hipMemcpy(w.data(), (const __half*)t.W + (size_t)r * sh.K, sh.K * 2, hipMemcpyDeviceToHost));
                for (int c = 0; c < sh.K; c++) acc += (double)__half2float(w[c]) * __half2float(hx[c]);
            } else {
                const int ng = sh.K / GROUP;
                std::vector<uint8_t> w(sh.K / 2); std::vector<__half> S(ng), Z(ng);
                CHECK(hipMemcpy(w.data(), (const uint8_t*)t.W + (size_t)r * (sh.K / 2), sh.K / 2, hipMemcpyDeviceToHost));
                CHECK(hipMemcpy(S.data(), t.S + (size_t)r * ng, ng * 2, hipMemcpyDeviceToHost)); CHECK(hipMemcpy(Z.data(), t.Z + (size_t)r * ng, ng * 2, hipMemcpyDeviceToHost));
                for (int c = 0; c < sh.K; c++) { int q = (w[c / 2] >> ((c & 1) * 4)) & 0xF; acc += (double)__half2float(S[c / GROUP]) * (q - (double)__half2float(Z[c / GROUP])) * __half2float(hx[c]); }
            }
            host_err = std::max(host_err, std::fabs(acc - hr[r]) / (std::fabs(acc) + 1e-3));
        }
        double max_rel = 0; int nonfinite = 0;   // (non-finite outputs or errors fail outright: std::max would skip a NaN)
        for (int r = 0; r < sh.N; r++) {
            const double e = (double)std::fabs(hy[r] - hr[r]) / (std::fabs(hr[r]) + 1e-2);
            if (!std::isfinite(hy[r]) || !std::isfinite(hr[r]) || !std::isfinite(e)) nonfinite++; else max_rel = std::max(max_rel, e);
        }
        const bool ok = nonfinite == 0 && std::isfinite(host_err) && max_rel < 2e-3 && host_err < 2e-3;
        bad += !ok;
        printf("%s check %-4s %7d x %5d %-22s tile vs reference max rel %.2e, reference vs host %.2e, non-finite %d\n", ok ? "PASS" : "FAIL", fmt ? "i4" : "f16", sh.N, sh.K, sh.label, max_rel, host_err, nonfinite);
        CHECK(hipFree(d)); a.release();
    }
    printf("RESULT %s\n", bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}

static const char* H_SHAPES = "mode,fmt,cache,N,K,label,weight_bytes,copies,reuse_MB,event_us_median,event_us_p10,event_us_p90,event_GBps";

static int shapes_mode(const Opts& o) {
    const int fmt = fmt_of(o.fmt);
    hipStream_t st; CHECK(hipStreamCreate(&st));
    __half* x; float* y; CHECK(hipMalloc(&x, 4096 * 2)); CHECK(hipMalloc(&y, 248320 * 4));
    hipLaunchKernelGGL(k_fill_half, dim3(16), dim3(256), 0, 0, x, (size_t)4096, 7, -1.f, 1.f);
    std::vector<Shape> shapes = TABLE_SHAPES;
    if (o.mode == "sweep") { shapes.clear(); for (int n : {32, 64, 128, 256, 512, 768, 1024, 2048, 4096, 8192, 16384, 32768}) shapes.push_back({n, 2048, "sweep K=2048"}); }
    for (const Shape& sh : shapes) {
        Copies c; make_copies(c, sh, fmt, o.warm);
        const size_t n = c.t.size();
        std::vector<Seg> hs; for (auto& t : c.t) hs.push_back(seg_of(t, x, y, 0));
        Seg* d; CHECK(hipMalloc(&d, hs.size() * sizeof(Seg))); CHECK(hipMemcpy(d, hs.data(), hs.size() * sizeof(Seg), hipMemcpyHostToDevice));
        const int nt = host_ntiles(sh.N), reps = o.reps;
        size_t k = 0;                                   // one cursor for warmup and timed launches
        for (int i = 0; i < 20; i++) hipLaunchKernelGGL(k_launch, dim3(nt), dim3(WG), 0, st, d + (k++ % n), 1, 0);
        std::vector<hipEvent_t> e(2 * reps); for (auto& v : e) CHECK(hipEventCreate(&v));
        for (int i = 0; i < reps; i++) {
            CHECK(hipEventRecord(e[2 * i], st));
            hipLaunchKernelGGL(k_launch, dim3(nt), dim3(WG), 0, st, d + (k++ % n), 1, 0);
            CHECK(hipEventRecord(e[2 * i + 1], st));
        }
        CHECK(hipStreamSynchronize(st));
        std::vector<double> us; for (int i = 0; i < reps; i++) { float ms; CHECK(hipEventElapsedTime(&ms, e[2 * i], e[2 * i + 1])); us.push_back(ms * 1e3); }
        const double b = (double)c.bytes, med = median(us);
        char row[512];
        snprintf(row, sizeof row, "shapes,%s,%s,%d,%d,\"%s\",%.0f,%zu,%.1f,%.2f,%.2f,%.2f,%.1f", o.fmt.c_str(), o.warm ? "warm" : "cold", sh.N, sh.K, sh.label, b,
                 n, reuse_mb(c, 1), med, pct(us, 0.1), pct(us, 0.9), b / (med * 1e-6) / 1e9);
        csv_row(o, H_SHAPES, row);
        for (auto& v : e) CHECK(hipEventDestroy(v));
        CHECK(hipFree(d)); c.a.release();
    }
    return 0;
}

// The four persistent variants (the same tile code; PLAN.md, E2 / E3): a device-wide barrier between stages or per-stage
// done counters (dataflow), each with or without prefetching a tile's weights before its wait.
enum { V_BARRIER = 0, V_BARRIER_PF, V_DATAFLOW, V_DATAFLOW_PF, NV };
static const char* VNAME[NV] = {"barrier", "barrier-pf", "dataflow", "dataflow-pf"};

static int g_occ[NV] = {0, 0, 0, 0};
// One grid for every variant: the smallest of the four kernels' occupancy limits (or --grid-per-cu), times the WGPs.
// (The barrier variants need every block resident; on the XT an empty-stage barrier completed one block per WGP above
// the occupancy count of barrier-pf and timed out two above it.)
static int persistent_grid(int* per_out = nullptr) {
    int dev; CHECK(hipGetDevice(&dev)); hipDeviceProp_t p; CHECK(hipGetDeviceProperties(&p, dev));
    CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&g_occ[V_BARRIER], k_persistent<false>, WG, 0));
    CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&g_occ[V_BARRIER_PF], k_persistent<true>, WG, 0));
    CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&g_occ[V_DATAFLOW], k_dataflow<false>, WG, 0));
    CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&g_occ[V_DATAFLOW_PF], k_dataflow<true>, WG, 0));
    const int occ = *std::min_element(g_occ, g_occ + NV);
    int per = std::max(1, occ);
    if (per_out) *per_out = per;
    if (g_grid_per_cu > 0) {
        if (g_grid_per_cu > occ && !getenv("RESIDENCY_PROBE")) { fprintf(stderr, "--grid-per-cu %d > the occupancy maximum %d: refused (blocks would not all be resident)\n", g_grid_per_cu, occ); exit(2); }
        per = g_grid_per_cu;
    }
    return per * p.multiProcessorCount;
}
static uint64_t wall_rate_hz() { int dev; CHECK(hipGetDevice(&dev)); int khz = 0; CHECK(hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, dev)); return (uint64_t)khz * 1000; }
// dataflow claim size per stage: about one chunk per block for large stages (the per-claim protocol, an atomic, three
// block syncs, a fence and a release, would otherwise run once per 16-row tile: 9,496 times for the 30B vocab head)
// (rounded up, so a stage never has more chunks than blocks: with floor(9496 / 210) = 45 the vocab head had 212 chunks
// for 210 blocks, and two blocks doing a second chunk made the stage 1.44x as long on the XT)
static void set_chunks(std::vector<Stage>& v, int grid) { const int g = std::max(1, grid); for (auto& s : v) s.chunk = std::max(1, (s.ntiles + g - 1) / g); }

struct Persist {
    Sync* sy = nullptr; unsigned* ds = nullptr; int* derr = nullptr; uint64_t* stamps = nullptr; int* prog = nullptr;
    int grid = 0, maxst = 0; uint64_t limit = 0, rate = 0; int nonmono = 0;
    void init(int max_stages) {
        maxst = max_stages;
        CHECK(hipMalloc(&sy, sizeof(Sync)));
        CHECK(hipMalloc(&ds, 3 * (size_t)maxst * LINE * sizeof(unsigned))); CHECK(hipMalloc(&derr, 8 * sizeof(int)));
        CHECK(hipMalloc(&stamps, (maxst + 1) * sizeof(uint64_t)));
        grid = persistent_grid(); rate = wall_rate_hz(); limit = rate / 5;     // 200 ms per wait at most
        CHECK(hipMalloc(&prog, 65536 * sizeof(int)));
    }
    // before every run, outside the timed span: every piece of sync state from zero (a timed-out run leaves the barrier's
    // count and generation inconsistent), the stage stamps at 0 and the start stamp at ~0 (atomic min / max targets)
    void prep(hipStream_t st) {
        CHECK(hipMemsetAsync(sy, 0, sizeof(Sync), st)); CHECK(hipMemsetAsync(derr, 0, 8 * sizeof(int), st));
        CHECK(hipMemsetAsync(ds, 0, 3 * (size_t)maxst * LINE * sizeof(unsigned), st));
        CHECK(hipMemsetAsync(stamps, 0, maxst * sizeof(uint64_t), st)); CHECK(hipMemsetAsync(stamps + maxst, 0xff, sizeof(uint64_t), st));
        if (TRACE) CHECK(hipMemsetAsync(prog, 0, 65536 * sizeof(int), st));
    }
    void run(hipStream_t st, const Stage* stages, int n, const Seg* segs, int v, int validate = 0) {
        unsigned* vs = ds + 2 * (size_t)maxst * LINE;
        if (v == V_BARRIER) hipLaunchKernelGGL(k_persistent<false>, dim3(grid), dim3(WG), 0, st, stages, n, segs, sy, limit, stamps, vs, prog, validate);
        else if (v == V_BARRIER_PF) hipLaunchKernelGGL(k_persistent<true>, dim3(grid), dim3(WG), 0, st, stages, n, segs, sy, limit, stamps, vs, prog, validate);
        else if (v == V_DATAFLOW) hipLaunchKernelGGL(k_dataflow<false>, dim3(grid), dim3(WG), 0, st, stages, n, segs, ds, ds + (size_t)maxst * LINE, vs, derr, limit, stamps, prog, validate);
        else hipLaunchKernelGGL(k_dataflow<true>, dim3(grid), dim3(WG), 0, st, stages, n, segs, ds, ds + (size_t)maxst * LINE, vs, derr, limit, stamps, prog, validate);
    }
    int err() { Sync h; int d = 0; CHECK(hipMemcpy(&h, sy, sizeof h, hipMemcpyDeviceToHost)); CHECK(hipMemcpy(&d, derr, sizeof d, hipMemcpyDeviceToHost)); return std::max(h.err, d); }
    // the last run's stage times (stage i ends at stamps[i]; stage 0 starts at stamps[n]); an end before the previous one
    // (which the stamping order should rule out) is counted in nonmono and returned as NaN
    std::vector<double> stage_us(int n) {
        std::vector<uint64_t> h(n + 1); CHECK(hipMemcpy(h.data(), stamps, (n + 1) * sizeof(uint64_t), hipMemcpyDeviceToHost));
        std::vector<double> u(n);
        for (int i = 0; i < n; i++) { const uint64_t a = i ? h[i - 1] : h[n]; if (h[i] < a) { nonmono++; u[i] = NAN; } else u[i] = (double)(h[i] - a) * 1e6 / rate; }
        return u;
    }
};
static std::vector<double> finite(const std::vector<double>& v) { std::vector<double> r; for (double x : v) if (std::isfinite(x)) r.push_back(x); return r; }
static bool report_err(Persist& P, const char* what) {
    const int e = P.err();
    if (e == 1) {
        int d[8]; CHECK(hipMemcpy(d, P.derr, sizeof d, hipMemcpyDeviceToHost));
        fprintf(stderr, "%s: a device wait timed out", what);
        if (d[0]) {
            fprintf(stderr, " (dataflow: block %d waiting in stage %d saw done[%d] = %d of %d, claim[%d] = %d)", d[5], d[1], d[1] - 1, d[2], d[3], d[1] - 1, d[4]);
            if (TRACE) {   // the blocks holding an unpublished tile: their stage, phase, and the counters of that stage and the one before
                std::vector<int> pr(P.grid); CHECK(hipMemcpy(pr.data(), P.prog, P.grid * sizeof(int), hipMemcpyDeviceToHost));
                std::vector<unsigned> dsh(3 * (size_t)P.maxst * LINE); CHECK(hipMemcpy(dsh.data(), P.ds, dsh.size() * sizeof(unsigned), hipMemcpyDeviceToHost));
                auto cnt = [&](int which, int st) { return st < 0 ? -1 : (int)dsh[((size_t)which * P.maxst + st) * LINE]; };
                const char* ph[5] = {"-", "claimed (prefetch / before the wait)", "wait over (computing)", "computed (publishing)", "published"};
                std::map<int, int> lo; for (int b = 0; b < P.grid; b++) if ((pr[b] & 0xff) && (pr[b] & 0xff) < 4) lo[pr[b] >> 8]++;
                {   std::vector<uint64_t> h(P.maxst + 1); CHECK(hipMemcpy(h.data(), P.stamps, h.size() * sizeof(uint64_t), hipMemcpyDeviceToHost));
                    fprintf(stderr, "\n    stage finished at (ms after the start):");
                    for (int k = 0; k < std::min(P.maxst, 10); k++) fprintf(stderr, " [%d] %.4f", k, h[k] >= h[P.maxst] ? (h[k] - h[P.maxst]) * 1e3 / P.rate : -1.0);
                }
                int shown = 0;
                for (int b = 0; b < P.grid && shown < 12; b++) {
                    const int stg = pr[b] >> 8, phs = pr[b] & 0xff;
                    if (!phs || phs == 4) continue;
                    if (stg > lo.begin()->first + 2) continue;   // only the lowest unfinished stages
                    fprintf(stderr, "\n    block %d: stage %d, %s; claim[%d]=%d done[%d]=%d done[%d]=%d", b, stg, ph[phs], stg, cnt(0, stg), stg, cnt(1, stg), stg - 1, cnt(1, stg - 1));
                    {   int r[4]; CHECK(hipMemcpy(r, P.prog + 4096 + 4 * b, sizeof r, hipMemcpyDeviceToHost));
                        int gp2[2]; CHECK(hipMemcpy(gp2, P.prog + 8192 + 2 * b, sizeof gp2, hipMemcpyDeviceToHost));
                        fprintf(stderr, "; at its timeout: stage %d, saw %d, target %d, word offset %d (= stage %d); polls %d, gap before the last check %.3f ms", r[0], r[1], r[2], r[3], r[3] / LINE, gp2[1], gp2[0] * 1e3 / P.rate); }
                    shown++;
                }
            }
        }
        Sync h; CHECK(hipMemcpy(&h, P.sy, sizeof h, hipMemcpyDeviceToHost));
        if (h.err) {
            fprintf(stderr, " (barrier: block %d waiting at the barrier before stage %d, its gen %d, now gen %u count %d of grid %d)", h.dbg[3], h.dbg[0], h.dbg[1], h.gen, h.dbg[2], P.grid);
            std::vector<int> pr(P.grid); CHECK(hipMemcpy(pr.data(), P.prog, P.grid * sizeof(int), hipMemcpyDeviceToHost));
            std::map<int, std::vector<int>> by; for (int b = 0; b < P.grid; b++) by[pr[b]].push_back(b);
            const char* ph[4] = {"not started", "arrived", "left the wait", "timed out"};
            std::vector<int> tr(6 * P.grid); CHECK(hipMemcpy(tr.data(), P.prog + 4096, 6 * P.grid * sizeof(int), hipMemcpyDeviceToHost));
            auto tarr = [&](int b) { return *reinterpret_cast<const uint64_t*>(&tr[6 * b + 2]); };
            auto tout = [&](int b) { return *reinterpret_cast<const uint64_t*>(&tr[6 * b + 4]); };
            uint64_t t00 = ~0ull; for (int b = 0; b < P.grid; b++) t00 = std::min(t00, tarr(b));
            for (auto& kv : by) {
                std::map<int, int> gens; int amin = 1 << 30, amax = -1;
                double a0 = 1e30, a1 = -1, o0 = 1e30, o1 = -1;
                for (int b : kv.second) { gens[tr[6 * b]]++; amin = std::min(amin, tr[6 * b + 1]); amax = std::max(amax, tr[6 * b + 1]);
                    const double ta = (tarr(b) - t00) * 1e3 / P.rate, to = (tout(b) - t00) * 1e3 / P.rate; a0 = std::min(a0, ta); a1 = std::max(a1, ta); o0 = std::min(o0, to); o1 = std::max(o1, to); }
                fprintf(stderr, "\n      (last arrival %.3f..%.3f ms, timed out %.3f..%.3f ms, after the earliest arrival)", a0, a1, o0, o1);
                fprintf(stderr, "\n    %3zu blocks: barrier before stage %d, %s; arrival numbers %d..%d; gen read:", kv.second.size(), kv.first / 4, ph[kv.first % 4], amin, amax);
                for (auto& g : gens) fprintf(stderr, " %d (x%d)", g.first, g.second);
            }
        }
        fprintf(stderr, "\n");
    }
    if (e == 2) fprintf(stderr, "%s: DEPENDENCY FAILURE: after its wait, a stage saw an incomplete sum from the stage before it\n", what);
    return e != 0;
}
static void print_grid(const Persist& P) {
    int dev; CHECK(hipGetDevice(&dev)); hipDeviceProp_t pr; CHECK(hipGetDeviceProperties(&pr, dev));
    printf("# persistent grid %d blocks of %d (%d per WGP; occupancy max per WGP: barrier %d, barrier-pf %d, dataflow %d, dataflow-pf %d), wall clock %.0f MHz\n",
           P.grid, WG, P.grid / pr.multiProcessorCount, g_occ[0], g_occ[1], g_occ[2], g_occ[3], P.rate / 1e6);
}

static const char* H_PSHAPES = "mode,fmt,cache,N,K,label,weight_bytes,stages,grid,copies,reuse_MB,launch_us,graph_us,"
    "barrier_us,barrier_stage_us,barrier_pf_us,barrier_pf_stage_us,dataflow_us,dataflow_stage_us,dataflow_pf_us,dataflow_pf_stage_us";

// each shape as S stages: S back-to-back launches, the same S launches as a HIP graph, and one persistent launch in every
// variant (E2). Every run reads the next ring of S copies (one cursor); a variant that fails (timeout, dependency check)
// is reported as "fail" and skipped from then on. Event times, amortized per stage; stage medians from the GPU stamps.
static int pshapes_mode(const Opts& o) {
    const int fmt = fmt_of(o.fmt), S = o.stages;
    hipStream_t st; CHECK(hipStreamCreate(&st));
    __half* x; float* y; CHECK(hipMalloc(&x, 4096 * 2)); CHECK(hipMalloc(&y, 248320 * 4));
    hipLaunchKernelGGL(k_fill_half, dim3(16), dim3(256), 0, 0, x, (size_t)4096, 7, -1.f, 1.f);
    Persist P; P.init(S); print_grid(P);
    for (const Shape& sh : TABLE_SHAPES) {
        if (sh.N > 16384) continue;                    // (the vocab heads: one per token, nothing to amortize)
        Copies c; make_copies(c, sh, fmt, o.warm, o.warm ? 1 : S);
        const int n = (int)c.t.size(), R = o.warm ? 1 : n / S;
        std::vector<Seg> hs; for (int i = 0; i < R * S; i++) hs.push_back(seg_of(c.t[o.warm ? 0 : i], x, y, 0));
        std::vector<Stage> hst; for (int i = 0; i < S; i++) hst.push_back({i, 1, host_ntiles(sh.N), 1, 1});
        set_chunks(hst, P.grid);
        Seg* d; Stage* ds; CHECK(hipMalloc(&d, hs.size() * sizeof(Seg))); CHECK(hipMalloc(&ds, hst.size() * sizeof(Stage)));
        CHECK(hipMemcpy(d, hs.data(), hs.size() * sizeof(Seg), hipMemcpyHostToDevice)); CHECK(hipMemcpy(ds, hst.data(), hst.size() * sizeof(Stage), hipMemcpyHostToDevice));
        auto launches = [&](const Seg* dr) { for (int i = 0; i < S; i++) hipLaunchKernelGGL(k_launch, dim3(host_ntiles(sh.N)), dim3(WG), 0, st, dr + i, 1, 0); };
        std::vector<hipGraphExec_t> graphs(R);
        for (int r = 0; r < R; r++) { hipGraph_t g; CHECK(hipStreamBeginCapture(st, hipStreamCaptureModeGlobal)); launches(d + (size_t)r * S); CHECK(hipStreamEndCapture(st, &g)); CHECK(hipGraphInstantiate(&graphs[r], g, nullptr, nullptr, 0)); CHECK(hipGraphDestroy(g)); }
        // the dependency check, once per variant, before timing
        bool vfail[NV] = {false, false, false, false};
        for (int v = 0; v < NV; v++) { P.prep(st); P.run(st, ds, S, d, v, 1); CHECK(hipStreamSynchronize(st)); vfail[v] = report_err(P, VNAME[v]); }
        std::vector<double> launch_us, graph_us; std::vector<std::vector<double>> v_us(NV), v_stage(NV);
        hipEvent_t a, b; CHECK(hipEventCreate(&a)); CHECK(hipEventCreate(&b));
        size_t k = 1;                                  // the ring cursor: run k reads ring k % R
        for (int rep = -1; rep < std::max(3, o.reps / 40); rep++) {   // (rep -1: warmup)
            float ms;
            CHECK(hipEventRecord(a, st)); launches(d + (size_t)(k++ % R) * S); CHECK(hipEventRecord(b, st)); CHECK(hipEventSynchronize(b));
            CHECK(hipEventElapsedTime(&ms, a, b)); if (rep >= 0) launch_us.push_back(ms * 1e3 / S);
            CHECK(hipEventRecord(a, st)); CHECK(hipGraphLaunch(graphs[k++ % R], st)); CHECK(hipEventRecord(b, st)); CHECK(hipEventSynchronize(b));
            CHECK(hipEventElapsedTime(&ms, a, b)); if (rep >= 0) graph_us.push_back(ms * 1e3 / S);
            for (int v = 0; v < NV; v++) {
                if (vfail[v]) continue;
                const Seg* dp = d + (size_t)(k++ % R) * S;
                P.prep(st);
                CHECK(hipEventRecord(a, st)); P.run(st, ds, S, dp, v); CHECK(hipEventRecord(b, st)); CHECK(hipEventSynchronize(b));
                if (report_err(P, VNAME[v])) { vfail[v] = true; continue; }
                CHECK(hipEventElapsedTime(&ms, a, b));
                if (rep < 0) continue;
                v_us[v].push_back(ms * 1e3 / S);
                auto u = P.stage_us(S); for (int i = 1; i < S; i++) v_stage[v].push_back(u[i]);
            }
        }
        char row[1024]; int w = snprintf(row, sizeof row, "pshapes,%s,%s,%d,%d,\"%s\",%.0f,%d,%d,%d,%.1f,%.2f,%.2f", o.fmt.c_str(), o.warm ? "warm" : "cold", sh.N, sh.K, sh.label,
                                         (double)c.bytes, S, P.grid, n, reuse_mb(c, S), median(launch_us), median(graph_us));
        for (int v = 0; v < NV; v++) {
            if (vfail[v]) w += snprintf(row + w, sizeof row - w, ",fail,fail");
            else w += snprintf(row + w, sizeof row - w, ",%.2f,%.2f", median(v_us[v]), median(finite(v_stage[v])));
        }
        csv_row(o, H_PSHAPES, row);
        for (auto& g : graphs) CHECK(hipGraphExecDestroy(g));
        CHECK(hipEventDestroy(a)); CHECK(hipEventDestroy(b)); CHECK(hipFree(d)); CHECK(hipFree(ds)); c.a.release();
    }
    if (P.nonmono) printf("# WARNING: %d stage stamps ran backwards (excluded)\n", P.nonmono);
    return 0;
}

static const char* H_SYNC = "mode,fmt,variant,tiles_per_stage,grid,blocks_per_WGP,stage_us_median,stage_us_p90,event_us_each,status";

// the synchronization alone: S stages of 0 tiles (the barrier variants only: the device-wide sync) and of 1 tile (one
// 16 x 2048 INT4 tile, the smallest unit of work; every variant), in one persistent launch; against --grid-per-cu
static int sync_mode(const Opts& o) {
    const int S = o.stages;
    hipStream_t st; CHECK(hipStreamCreate(&st));
    __half* x; float* y; CHECK(hipMalloc(&x, 4096 * 2)); CHECK(hipMalloc(&y, 248320 * 4));
    hipLaunchKernelGGL(k_fill_half, dim3(16), dim3(256), 0, 0, x, (size_t)4096, 7, -1.f, 1.f);
    Persist P; P.init(S); print_grid(P);
    int dev; CHECK(hipGetDevice(&dev)); hipDeviceProp_t pr; CHECK(hipGetDeviceProperties(&pr, dev));
    Copies c; make_copies(c, Shape{TILE_ROWS, 2048, "one tile"}, FMT_I4, false, S);
    const int n = (int)c.t.size(), R = n / S;
    printf("# one-tile copies: %d (%.1f MB), a copy is re-read after %.1f MB\n", n, n * c.bytes / 1e6, reuse_mb(c, S));
    std::vector<Seg> hs; for (int i = 0; i < R * S; i++) hs.push_back(seg_of(c.t[i], x, y, 0));
    Seg* d; CHECK(hipMalloc(&d, hs.size() * sizeof(Seg))); CHECK(hipMemcpy(d, hs.data(), hs.size() * sizeof(Seg), hipMemcpyHostToDevice));
    hipEvent_t a, b; CHECK(hipEventCreate(&a)); CHECK(hipEventCreate(&b));
    size_t k = 0;
    for (int nt : {0, 1}) {
        if (getenv("ONLY_NT") && atoi(getenv("ONLY_NT")) != nt) continue;   // (debugging)
        const int spf = getenv("PF_OFF") ? 0 : 1;                           // (debugging: the PF kernels with prefetch switched off per stage)
        std::vector<Stage> hst; for (int i = 0; i < S; i++) hst.push_back({i, nt, nt, spf, 1});
        Stage* ds; CHECK(hipMalloc(&ds, S * sizeof(Stage))); CHECK(hipMemcpy(ds, hst.data(), S * sizeof(Stage), hipMemcpyHostToDevice));
        for (int v = 0; v < NV; v++) {
            if (nt == 0 && v >= V_DATAFLOW) continue;   // (0-tile stages: the barrier alone; dataflow needs a tile per stage)
            if (getenv("DF_ONLY") && strcmp(getenv("DF_ONLY"), VNAME[v]) != 0) continue;   // (debugging: one variant)
            std::vector<double> stage_us, each_us;
            P.prep(st); P.run(st, ds, S, d, v, nt);  CHECK(hipStreamSynchronize(st));   // (the dependency check, 1-tile stages)
            bool failed = report_err(P, VNAME[v]);
            for (int rep = -1; rep < std::max(3, o.reps / 40) && !failed; rep++) {
                P.prep(st);
                CHECK(hipEventRecord(a, st)); P.run(st, ds, S, d + (size_t)(k++ % R) * S, v); CHECK(hipEventRecord(b, st)); CHECK(hipEventSynchronize(b));
                if (report_err(P, VNAME[v])) { failed = true; break; }
                if (rep < 0) continue;
                float ms; CHECK(hipEventElapsedTime(&ms, a, b)); each_us.push_back(ms * 1e3 / S);
                auto u = P.stage_us(S); for (int i = 1; i < S; i++) stage_us.push_back(u[i]);
            }
            char row[256];
            if (failed) snprintf(row, sizeof row, "sync,i4,%s,%d,%d,%d,,,,fail", VNAME[v], nt, P.grid, P.grid / pr.multiProcessorCount);
            else snprintf(row, sizeof row, "sync,i4,%s,%d,%d,%d,%.2f,%.2f,%.2f,ok", VNAME[v], nt, P.grid, P.grid / pr.multiProcessorCount, median(finite(stage_us)), pct(finite(stage_us), 0.9), median(each_us));
            csv_row(o, H_SYNC, row);
        }
        CHECK(hipFree(ds));
    }
    if (P.nonmono) printf("# WARNING: %d stage stamps ran backwards (excluded)\n", P.nonmono);
    CHECK(hipFree(d)); c.a.release();
    return 0;
}

// ---------------------------------------------------------------------------------- the Qwen3-30B-A3B decode token
// 48 layers: stage 1 q|k|v (x = h), stage 2 o (x = attention out), stage 3 router, stage 4 the 8 experts' gate|up,
// stage 5 the 8 experts' down (x = the SwiGLU output); then the lm_head. Attention, norms, RoPE and top-k are out of
// scope (PLAN.md): the matvec inputs are fixed vectors, the stage order and syncs are the real ones.
constexpr int HID = 2048, QD = 4096, KVD = 512, NEXP = 128, TOPK = 8, EFF = 768, VOCAB = 151936;

struct TokenModel {
    Arena a; std::vector<Tensor> q, k, v, o, router; std::vector<std::vector<Tensor>> gate, up, down; Tensor head;
    int layers = 48, nexp = NEXP, fmt = FMT_I4;
};
// warm: one layer with 8 experts (INT4 about 31 MB, inside the 96 MB Infinity Cache), reused by every layer of the token
static void build_model(TokenModel& m, int fmt, int layers, bool warm) {
    m.fmt = fmt; m.layers = layers; m.nexp = (fmt == FMT_I4 && !warm) ? NEXP : TOPK;   // f16: only the 8 experts a token uses fit
    size_t need = 0;
    auto add = [&](int r, int K) { need += tensor_alloc_bytes(r, K, fmt); };
    for (int l = 0; l < layers; l++) { add(QD, HID); add(KVD, HID); add(KVD, HID); add(HID, QD); add(NEXP, HID); for (int e = 0; e < m.nexp; e++) { add(EFF, HID); add(EFF, HID); add(HID, EFF); } }
    if (!warm) add(VOCAB, HID);
    m.a.init(need + 65536, "token model");
    uint32_t seed = 1;
    m.gate.resize(layers); m.up.resize(layers); m.down.resize(layers);
    for (int l = 0; l < layers; l++) {
        m.q.push_back(make_tensor(m.a, QD, HID, fmt, seed++)); m.k.push_back(make_tensor(m.a, KVD, HID, fmt, seed++));
        m.v.push_back(make_tensor(m.a, KVD, HID, fmt, seed++)); m.o.push_back(make_tensor(m.a, HID, QD, fmt, seed++));
        m.router.push_back(make_tensor(m.a, NEXP, HID, fmt, seed++));
        for (int e = 0; e < m.nexp; e++) {
            m.gate[l].push_back(make_tensor(m.a, EFF, HID, fmt, seed++)); m.up[l].push_back(make_tensor(m.a, EFF, HID, fmt, seed++));
            m.down[l].push_back(make_tensor(m.a, HID, EFF, fmt, seed++));
        }
    }
    if (!warm) m.head = make_tensor(m.a, VOCAB, HID, fmt, seed++);
    CHECK(hipDeviceSynchronize());
    printf("# token model: %s, %d layers, %d experts per layer allocated, %.2f GB\n", fmt ? "i4" : "f16", layers, m.nexp, m.a.used / 1e9);
}

struct TokenPlan { std::vector<Seg> segs; std::vector<Stage> stages; size_t bytes = 0; };
static TokenPlan plan_token(const TokenModel& m, int layers, bool warm, std::mt19937& rng, const __half* xh, const __half* xq, const __half* xe, float* ybase) {
    TokenPlan p; size_t yoff = 0;
    // pf: whether a stage's weights are known before its dependency finishes. The routed experts' gate|up are not (the
    // router picks them), so no variant prefetches them; their down weights are (same experts), as are all static weights.
    auto add_stage = [&](std::vector<const Tensor*> ts, const __half* x, int pf) {
        Stage s; s.seg0 = (int)p.segs.size(); s.nseg = (int)ts.size(); s.pf = pf; s.chunk = 1; int tile = 0;
        for (const Tensor* t : ts) { p.segs.push_back(seg_of(*t, x, ybase + yoff, tile)); yoff += t->rows; tile += host_ntiles(t->rows); p.bytes += tensor_bytes(t->rows, t->K, t->fmt); }
        s.ntiles = tile; p.stages.push_back(s);
    };
    for (int ll = 0; ll < layers; ll++) {
        const int l = warm ? 0 : ll;                  // (warm: every layer reads layer 0's weights)
        add_stage({&m.q[l], &m.k[l], &m.v[l]}, xh, 1);
        add_stage({&m.o[l]}, xq, 1);
        add_stage({&m.router[l]}, xh, 1);
        std::vector<int> ex(m.nexp); for (int e = 0; e < m.nexp; e++) ex[e] = e;
        if (m.nexp > TOPK) std::shuffle(ex.begin(), ex.end(), rng);
        std::vector<const Tensor*> gu, dn;
        for (int j = 0; j < TOPK; j++) { gu.push_back(&m.gate[l][ex[j]]); gu.push_back(&m.up[l][ex[j]]); dn.push_back(&m.down[l][ex[j]]); }
        add_stage(gu, xh, 0);
        add_stage(dn, xe, 1);
    }
    if (!warm) add_stage({&m.head}, xh, 1);   // (warm: no vocab head; 166 MB would not fit the cache)
    return p;
}

static int token_mode(const Opts& o) {
    const int fmt = fmt_of(o.fmt);
    hipStream_t st; CHECK(hipStreamCreate(&st));
    TokenModel m; build_model(m, fmt, o.warm ? 1 : o.layers, o.warm);
    __half *xh, *xq, *xe; CHECK(hipMalloc(&xh, HID * 2)); CHECK(hipMalloc(&xq, QD * 2)); CHECK(hipMalloc(&xe, EFF * 2));
    hipLaunchKernelGGL(k_fill_half, dim3(8), dim3(256), 0, 0, xh, (size_t)HID, 11, -1.f, 1.f);
    hipLaunchKernelGGL(k_fill_half, dim3(16), dim3(256), 0, 0, xq, (size_t)QD, 12, -1.f, 1.f);
    hipLaunchKernelGGL(k_fill_half, dim3(4), dim3(256), 0, 0, xe, (size_t)EFF, 13, -1.f, 1.f);
    const size_t yfloats = (size_t)o.layers * (QD + 2 * KVD + HID + NEXP + 2 * TOPK * EFF + TOPK * HID) + VOCAB;
    float *y, *y2; CHECK(hipMalloc(&y, yfloats * 4)); CHECK(hipMalloc(&y2, yfloats * 4));
    std::mt19937 rng(1234);
    const int P = std::max(1, o.tokens);
    std::vector<TokenPlan> plans; std::vector<Seg*> dsegs; Stage* dst = nullptr;
    for (int i = 0; i < P; i++) {
        plans.push_back(plan_token(m, o.layers, o.warm, rng, xh, xq, xe, y));
        Seg* d; CHECK(hipMalloc(&d, plans[i].segs.size() * sizeof(Seg)));
        CHECK(hipMemcpy(d, plans[i].segs.data(), plans[i].segs.size() * sizeof(Seg), hipMemcpyHostToDevice)); dsegs.push_back(d);
    }
    std::vector<Stage> S0 = plans[0].stages; const int nst = (int)S0.size(), nseg = (int)plans[0].segs.size();   // (the same stages for every routing sample)
    Persist Pe; Pe.init(nst);
    set_chunks(S0, Pe.grid);
    CHECK(hipMalloc(&dst, nst * sizeof(Stage))); CHECK(hipMemcpy(dst, S0.data(), nst * sizeof(Stage), hipMemcpyHostToDevice));
    printf("# token: %d matvecs in %d stages, %.3f GB of weights per token, %d routing samples\n", nseg, nst, plans[0].bytes / 1e9, P);
    print_grid(Pe);

    auto run_perop = [&](int p) { const auto& sg = plans[p].segs; for (int i = 0; i < nseg; i++) hipLaunchKernelGGL(k_launch, dim3(host_ntiles(sg[i].rows)), dim3(WG), 0, st, dsegs[p] + i, 1, sg[i].tile0); };
    auto run_grouped = [&](int p) { for (const Stage& s : S0) hipLaunchKernelGGL(k_launch, dim3(s.ntiles), dim3(WG), 0, st, dsegs[p] + s.seg0, s.nseg, 0); };
    auto capture = [&](auto fn, int p) { hipGraph_t g; hipGraphExec_t ge; CHECK(hipStreamBeginCapture(st, hipStreamCaptureModeGlobal)); fn(p); CHECK(hipStreamEndCapture(st, &g)); CHECK(hipGraphInstantiate(&ge, g, nullptr, nullptr, 0)); CHECK(hipGraphDestroy(g)); return ge; };
    std::vector<hipGraphExec_t> g_perop, g_grouped;
    for (int p = 0; p < P; p++) { g_perop.push_back(capture(run_perop, p)); g_grouped.push_back(capture(run_grouped, p)); }

    // correctness: every persistent variant's outputs == the per-op outputs, bitwise (the same tile code on the same
    // data; prefetching only moves loads), no wait timed out, and the dependency check (validate) saw every producer.
    // y is zeroed before the reference too, so regions a plan does not write (the warm token's vocab head) compare equal.
    bool vfail[NV] = {false, false, false, false};
    CHECK(hipMemset(y, 0, yfloats * 4)); run_perop(0); CHECK(hipStreamSynchronize(st)); CHECK(hipMemcpy(y2, y, yfloats * 4, hipMemcpyDeviceToDevice));
    {   std::vector<float> ref(yfloats), got(yfloats); CHECK(hipMemcpy(ref.data(), y2, yfloats * 4, hipMemcpyDeviceToHost));
        for (int v = 0; v < NV; v++) {
            CHECK(hipMemset(y, 0, yfloats * 4)); Pe.prep(st); Pe.run(st, dst, nst, dsegs[0], v, 1); CHECK(hipStreamSynchronize(st));
            if (report_err(Pe, VNAME[v])) { vfail[v] = true; printf("FAIL token: %s (wait or dependency check)\n", VNAME[v]); continue; }
            CHECK(hipMemcpy(got.data(), y, yfloats * 4, hipMemcpyDeviceToHost));
            const bool same = memcmp(got.data(), ref.data(), yfloats * 4) == 0;
            printf("%s token: %s outputs %s the per-op outputs bitwise (%zu floats), dependency check ok\n", same ? "PASS" : "FAIL", VNAME[v], same ? "==" : "!=", yfloats);
            if (!same) vfail[v] = true;
        }
    }

    const int NM = 4 + NV;
    const char* names[4 + NV] = {"per-op", "per-op-graph", "grouped", "grouped-graph", "persistent-barrier", "persistent-barrier-pf", "persistent-dataflow", "persistent-dataflow-pf"};
    std::vector<std::vector<double>> ms(NM);
    hipEvent_t a, b; CHECK(hipEventCreate(&a)); CHECK(hipEventCreate(&b));
    const int reps = std::max(5, o.reps / 20);
    for (int rep = -2; rep < reps; rep++) {                       // (2 warmup rounds; the modes interleaved, the order rotated each round)
        for (int j = 0; j < NM; j++) {
            const int mo = (j + rep + 2) % NM, p = (rep + 2 + mo) % P;   // (each mode cycles through the routing samples)
            if (mo >= 4 && vfail[mo - 4]) continue;
            if (mo >= 4) Pe.prep(st);
            CHECK(hipEventRecord(a, st));
            if (mo == 0) run_perop(p); else if (mo == 1) CHECK(hipGraphLaunch(g_perop[p], st));
            else if (mo == 2) run_grouped(p); else if (mo == 3) CHECK(hipGraphLaunch(g_grouped[p], st));
            else Pe.run(st, dst, nst, dsegs[p], mo - 4);
            CHECK(hipEventRecord(b, st)); CHECK(hipEventSynchronize(b));
            if (mo >= 4 && report_err(Pe, names[mo])) { vfail[mo - 4] = true; continue; }
            float t; CHECK(hipEventElapsedTime(&t, a, b)); if (rep >= 0) ms[mo].push_back(t);
        }
    }
    for (int mo = 0; mo < NM; mo++) {
        char row[512];
        if (mo >= 4 && vfail[mo - 4]) snprintf(row, sizeof row, "token,%s,%s,%s,%d,%d,%d,%.0f,,,,,%d,fail", o.fmt.c_str(), o.warm ? "warm" : "cold", names[mo], o.layers, nseg, nst, (double)plans[0].bytes, Pe.grid);
        else {
            const double med = median(ms[mo]);
            snprintf(row, sizeof row, "token,%s,%s,%s,%d,%d,%d,%.0f,%.3f,%.3f,%.3f,%.1f,%d,ok", o.fmt.c_str(), o.warm ? "warm" : "cold", names[mo], o.layers, nseg, nst, (double)plans[0].bytes,
                     med, pct(ms[mo], 0.1), pct(ms[mo], 0.9), plans[0].bytes / (med * 1e-3) / 1e9, mo >= 4 ? Pe.grid : 0);
        }
        csv_row(o, "mode,fmt,cache,strategy,layers,matvecs,stages,weight_bytes,ms_median,ms_p10,ms_p90,effective_GBps,persistent_grid,status", row);
    }
    // each persistent variant's stage times by stage kind (one more run each, routing sample 0)
    const char* kinds[6] = {"q|k|v", "o", "router", "gate|up x8", "down x8", "lm_head"};
    for (int v = 0; v < NV; v++) {
        if (vfail[v]) continue;
        Pe.prep(st); Pe.run(st, dst, nst, dsegs[0], v); CHECK(hipStreamSynchronize(st));
        if (report_err(Pe, VNAME[v])) continue;
        auto u = Pe.stage_us(nst);
        std::vector<std::vector<double>> by(6);
        for (int i = 0; i < nst; i++) by[(!o.warm && i == nst - 1) ? 5 : i % 5].push_back(u[i]);
        printf("# stages %-12s", VNAME[v]);
        for (int k = 0; k < 6; k++) if (!by[k].empty()) printf(" %s %.2f", kinds[k], median(finite(by[k])));
        printf(" (us, median by kind)\n");
    }
    if (Pe.nonmono) printf("# WARNING: %d stage stamps ran backwards (excluded)\n", Pe.nonmono);
    int nf = 0; for (int v = 0; v < NV; v++) nf += vfail[v];
    return nf ? 6 : 0;   // (6: a persistent variant failed; its rows say fail, the others are valid)
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: decode_floor check|shapes|sweep|pshapes|sync|token [--fmt f16|i4] [--warm] [--reps N] [--stages N] [--layers N] [--tokens N] [--csv FILE] [--expect-cus N] [--grid-per-cu N]\n"); return 2; }
    Opts o; o.mode = argv[1];
    for (int i = 2; i < argc; i++) {
        std::string a = argv[i]; auto next = [&]() { if (i + 1 >= argc) { fprintf(stderr, "%s needs a value\n", a.c_str()); exit(2); } return std::string(argv[++i]); };
        if (a == "--fmt") o.fmt = next(); else if (a == "--warm") o.warm = true; else if (a == "--reps") o.reps = std::stoi(next());
        else if (a == "--stages") o.stages = std::stoi(next()); else if (a == "--layers") o.layers = std::stoi(next());
        else if (a == "--tokens") o.tokens = std::stoi(next()); else if (a == "--csv") o.csv = next();
        else if (a == "--expect-cus") o.expect_cus = std::stoi(next()); else if (a == "--grid-per-cu") o.grid_per_cu = std::stoi(next());
        else { fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    if (o.fmt != "f16" && o.fmt != "i4") { fprintf(stderr, "--fmt f16|i4\n"); return 2; }
    int dev; CHECK(hipGetDevice(&dev)); hipDeviceProp_t p; CHECK(hipGetDeviceProperties(&p, dev));
    printf("# device %s (%s), %d CUs, decode_floor %s fmt=%s cache=%s poll=%d sleep=%d\n", p.name, p.gcnArchName, p.multiProcessorCount, o.mode.c_str(), o.fmt.c_str(), o.warm ? "warm" : "cold", POLL, SLEEPN);
    if (o.expect_cus && p.multiProcessorCount != o.expect_cus) { fprintf(stderr, "expected a device with %d CUs, got %d: refused\n", o.expect_cus, p.multiProcessorCount); return 2; }
    g_grid_per_cu = o.grid_per_cu;
    if (o.mode == "check") return check_mode(o);
    if (o.mode == "shapes" || o.mode == "sweep") return shapes_mode(o);
    if (o.mode == "pshapes") return pshapes_mode(o);
    if (o.mode == "sync") return sync_mode(o);
    if (o.mode == "token") return token_mode(o);
    fprintf(stderr, "unknown mode %s\n", o.mode.c_str()); return 2;
}
