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
//   token    --fmt F      the Qwen3-30B-A3B decode token's matvecs in every mode (E3)
// options: --fmt f16|i4  --warm (reuse one weight copy; default cold: rotate copies totalling > 4 x 96 MB)
//          --reps N  --stages N  --layers N (token: fewer layers for a smoke run)  --tokens N (routing samples)
//          --csv FILE (append result rows)
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
struct Stage { int seg0, nseg, ntiles; };
struct Sync { unsigned count, gen; int err, pad; };

__device__ __forceinline__ h2 as_h2(uint32_t u) { return __builtin_bit_cast(h2, u); }

__device__ __forceinline__ float wave_sum(float a) {
    for (int o = 16; o > 0; o >>= 1) a += __shfl_xor(a, o, 32);
    return a;
}

typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));    // (native vectors: HIP's uint4 / __half classes
                                                                   //  have no address-space-qualified copies)
__device__ __forceinline__ float dot_f16_row(const GLOBAL _Float16* w, const GLOBAL _Float16* x, int K, int lane) {
    float acc = 0.f;
    for (int c = lane * 8; c < K; c += 32 * 8) {
        const u32x4 wv = *reinterpret_cast<const GLOBAL u32x4*>(w + c), xv = *reinterpret_cast<const GLOBAL u32x4*>(x + c);
        acc = __builtin_amdgcn_fdot2(as_h2(wv.x), as_h2(xv.x), acc, false);
        acc = __builtin_amdgcn_fdot2(as_h2(wv.y), as_h2(xv.y), acc, false);
        acc = __builtin_amdgcn_fdot2(as_h2(wv.z), as_h2(xv.z), acc, false);
        acc = __builtin_amdgcn_fdot2(as_h2(wv.w), as_h2(xv.w), acc, false);
    }
    return acc;
}

// 32 columns per lane-chunk (16 bytes of nibbles), always inside one group of 128: acc += s * (sum q.x - z * sum x).
// A nibble q becomes the f16 1024 + q (exponent bits 0x6400 | q), minus 1024 in one packed subtract.
__device__ __forceinline__ float dot_i4_row(const GLOBAL uint8_t* w, const GLOBAL _Float16* S, const GLOBAL _Float16* Z,
                                            const GLOBAL _Float16* x, int K, int lane) {
    const h2 off = {(_Float16)1024.0f, (_Float16)1024.0f}, one = {(_Float16)1.0f, (_Float16)1.0f};
    float acc = 0.f;
    for (int c = lane * 32; c < K; c += 32 * 32) {
        const u32x4 wv = *reinterpret_cast<const GLOBAL u32x4*>(w + c / 2);
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
        const int g = c / GROUP;
        acc += (float)S[g] * (aq - (float)Z[g] * ax);
    }
    return acc;
}

__device__ __forceinline__ void do_tile(const Seg& s, int t) {
    const int wave = threadIdx.x >> 5, lane = threadIdx.x & 31;
#pragma unroll
    for (int i = 0; i < RPW; i++) {
        const int r = t * TILE_ROWS + wave * RPW + i;
        if (r >= s.rows) break;                      // (uniform within the wave)
        float a;
        if (s.fmt == FMT_F16) {
            a = dot_f16_row(gp(static_cast<const _Float16*>(s.W)) + (size_t)r * s.K, gp(reinterpret_cast<const _Float16*>(s.x)), s.K, lane);
        } else {
            const int ng = s.K / GROUP;
            a = dot_i4_row(gp(static_cast<const uint8_t*>(s.W)) + (size_t)r * (s.K / 2), gp(reinterpret_cast<const _Float16*>(s.S)) + (size_t)r * ng,
                           gp(reinterpret_cast<const _Float16*>(s.Z)) + (size_t)r * ng, gp(reinterpret_cast<const _Float16*>(s.x)), s.K, lane);
        }
        a = wave_sum(a);
        if (lane == 0) gpw(s.y)[r] = a;
    }
}

__device__ __forceinline__ int ntiles_of(int rows) { return (rows + TILE_ROWS - 1) / TILE_ROWS; }

__device__ __forceinline__ void run_tile(const Seg* segs, int nseg, int g) {
    for (int i = 0; i < nseg; i++) {                 // (a launch / stage holds at most a few dozen segments)
        const Seg& s = segs[i];
        if (g >= s.tile0 && g < s.tile0 + ntiles_of(s.rows)) { do_tile(s, g - s.tile0); return; }
    }
}

// a launch: block b computes tile base + b of the segments [segs, segs + nseg)
__global__ __launch_bounds__(WG) void k_launch(const Seg* segs, int nseg, int base) {
    run_tile(segs, nseg, base + (int)blockIdx.x);
}

// a device-wide sync with a time limit: on timeout the error flag is set and the kernel gives up (never spins forever)
__device__ bool grid_sync(Sync* sy, unsigned nb, uint64_t limit) {
    __threadfence();
    __syncthreads();
    __shared__ int bad;
    if (threadIdx.x == 0) {
        bad = 0;
        const unsigned g = __hip_atomic_load(&sy->gen, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        const unsigned arrived = __hip_atomic_fetch_add(&sy->count, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
        if (arrived == nb - 1) {
            __hip_atomic_store(&sy->count, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            __hip_atomic_fetch_add(&sy->gen, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        } else {
            const uint64_t t0 = wall_clock64();
            while (__hip_atomic_load(&sy->gen, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) == g) {
                if (wall_clock64() - t0 > limit) { bad = 1; __hip_atomic_store(&sy->err, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT); break; }
                __builtin_amdgcn_s_sleep(1);
            }
        }
    }
    __syncthreads();
    return !bad;
}

// the persistent kernel: every stage's tiles round-robin over the resident blocks, then a device-wide sync.
// stamps[i] (block 0): the wall clock after stage i's sync; stamps[nstages]: before the first stage
__global__ __launch_bounds__(WG) void k_persistent(const Stage* stages, int nstages, const Seg* segs, Sync* sy,
                                                   uint64_t limit, uint64_t* stamps) {
    if (blockIdx.x == 0 && threadIdx.x == 0) stamps[nstages] = wall_clock64();
    for (int i = 0; i < nstages; i++) {
        const Stage st = stages[i];
        for (int g = blockIdx.x; g < st.ntiles; g += gridDim.x) run_tile(segs + st.seg0, st.nseg, g);
        if (!grid_sync(sy, gridDim.x, limit)) return;
        if (blockIdx.x == 0 && threadIdx.x == 0) stamps[i] = wall_clock64();
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

struct Opts { std::string mode, fmt = "i4", csv; bool warm = false; int reps = 200, stages = 256, layers = 48, tokens = 8; };
static int fmt_of(const std::string& f) { return f == "f16" ? FMT_F16 : FMT_I4; }

static void csv_row(const Opts& o, const std::string& row) {
    printf("%s\n", row.c_str());
    if (!o.csv.empty()) { FILE* f = fopen(o.csv.c_str(), "a"); if (f) { fprintf(f, "%s\n", row.c_str()); fclose(f); } }
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

// a set of weight copies of one shape: cold = enough copies that their total passes 4 x the Infinity Cache
struct Copies { Arena a; std::vector<Tensor> t; };
static void make_copies(Copies& c, const Shape& sh, int fmt, bool warm) {
    const size_t b = tensor_bytes(sh.N, sh.K, fmt);
    int n = warm ? 1 : (int)std::min<uint64_t>(4096, (4 * CACHE_BYTES + b - 1) / b + 1);
    n = std::max(n, 1);
    c.a.init(tensor_alloc_bytes(sh.N, sh.K, fmt) * (size_t)n + 4096, "weights");
    for (int i = 0; i < n; i++) c.t.push_back(make_tensor(c.a, sh.N, sh.K, fmt, 1000 + i));
    CHECK(hipDeviceSynchronize());
}

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
        double max_rel = 0; for (int r = 0; r < sh.N; r++) max_rel = std::max(max_rel, (double)std::fabs(hy[r] - hr[r]) / (std::fabs(hr[r]) + 1e-2));
        const bool ok = max_rel < 2e-3 && host_err < 2e-3;
        bad += !ok;
        printf("%s check %-4s %7d x %5d %-22s tile vs reference max rel %.2e, reference vs host %.2e\n", ok ? "PASS" : "FAIL", fmt ? "i4" : "f16", sh.N, sh.K, sh.label, max_rel, host_err);
        CHECK(hipFree(d)); a.release();
    }
    printf("RESULT %s\n", bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}

static int shapes_mode(const Opts& o) {
    const int fmt = fmt_of(o.fmt);
    hipStream_t st; CHECK(hipStreamCreate(&st));
    __half* x; float* y; CHECK(hipMalloc(&x, 4096 * 2)); CHECK(hipMalloc(&y, 248320 * 4));
    hipLaunchKernelGGL(k_fill_half, dim3(16), dim3(256), 0, 0, x, (size_t)4096, 7, -1.f, 1.f);
    std::vector<Shape> shapes = TABLE_SHAPES;
    if (o.mode == "sweep") { shapes.clear(); for (int n : {32, 64, 128, 256, 512, 768, 1024, 2048, 4096, 8192, 16384, 32768}) shapes.push_back({n, 2048, "sweep K=2048"}); }
    for (const Shape& sh : shapes) {
        Copies c; make_copies(c, sh, fmt, o.warm);
        std::vector<Seg> hs; for (auto& t : c.t) hs.push_back(seg_of(t, x, y, 0));
        Seg* d; CHECK(hipMalloc(&d, hs.size() * sizeof(Seg))); CHECK(hipMemcpy(d, hs.data(), hs.size() * sizeof(Seg), hipMemcpyHostToDevice));
        const int nt = host_ntiles(sh.N), reps = o.reps;
        for (int i = 0; i < 20; i++) hipLaunchKernelGGL(k_launch, dim3(nt), dim3(WG), 0, st, d + (i % hs.size()), 1, 0);
        std::vector<hipEvent_t> e(2 * reps); for (auto& v : e) CHECK(hipEventCreate(&v));
        for (int i = 0; i < reps; i++) {
            CHECK(hipEventRecord(e[2 * i], st));
            hipLaunchKernelGGL(k_launch, dim3(nt), dim3(WG), 0, st, d + (i % hs.size()), 1, 0);
            CHECK(hipEventRecord(e[2 * i + 1], st));
        }
        CHECK(hipStreamSynchronize(st));
        std::vector<double> us; for (int i = 0; i < reps; i++) { float ms; CHECK(hipEventElapsedTime(&ms, e[2 * i], e[2 * i + 1])); us.push_back(ms * 1e3); }
        const double b = tensor_bytes(sh.N, sh.K, fmt), med = median(us);
        char row[512];
        snprintf(row, sizeof row, "shapes,%s,%s,%d,%d,\"%s\",%.0f,%zu,%.2f,%.2f,%.2f,%.1f", o.fmt.c_str(), o.warm ? "warm" : "cold", sh.N, sh.K, sh.label, b,
                 hs.size(), med, pct(us, 0.1), pct(us, 0.9), b / (med * 1e-6) / 1e9);
        csv_row(o, row);
        for (auto& v : e) CHECK(hipEventDestroy(v));
        CHECK(hipFree(d)); c.a.release();
    }
    printf("# columns: mode,fmt,cache,N,K,label,weight_bytes,copies,event_us_median,p10,p90,GB/s\n");
    return 0;
}

static int persistent_grid(int* per_cu_out = nullptr) {
    int dev; CHECK(hipGetDevice(&dev)); hipDeviceProp_t p; CHECK(hipGetDeviceProperties(&p, dev));
    int per = 0; CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&per, k_persistent, WG, 0));
    if (per_cu_out) *per_cu_out = per;
    return std::max(1, per) * p.multiProcessorCount;
}
static uint64_t wall_rate_hz() { int dev; CHECK(hipGetDevice(&dev)); int khz = 0; CHECK(hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, dev)); return (uint64_t)khz * 1000; }

struct Persist {
    Sync* sy = nullptr; uint64_t* stamps = nullptr; int grid = 0, nstamps = 0; uint64_t limit = 0, rate = 0;
    void init(int max_stages) {
        CHECK(hipMalloc(&sy, sizeof(Sync))); CHECK(hipMemset(sy, 0, sizeof(Sync)));
        CHECK(hipMalloc(&stamps, (max_stages + 1) * sizeof(uint64_t))); nstamps = max_stages + 1;
        grid = persistent_grid(); rate = wall_rate_hz(); limit = rate / 5;     // 200 ms per sync at most
    }
    bool run(hipStream_t st, const Stage* stages, int n, const Seg* segs) {
        hipLaunchKernelGGL(k_persistent, dim3(grid), dim3(WG), 0, st, stages, n, segs, sy, limit, stamps);
        return true;
    }
    bool ok() { Sync h; CHECK(hipMemcpy(&h, sy, sizeof h, hipMemcpyDeviceToHost)); return h.err == 0; }
};

static int pshapes_mode(const Opts& o) {
    const int fmt = fmt_of(o.fmt), S = o.stages;
    hipStream_t st; CHECK(hipStreamCreate(&st));
    __half* x; float* y; CHECK(hipMalloc(&x, 4096 * 2)); CHECK(hipMalloc(&y, 248320 * 4));
    hipLaunchKernelGGL(k_fill_half, dim3(16), dim3(256), 0, 0, x, (size_t)4096, 7, -1.f, 1.f);
    Persist P; P.init(S);
    printf("# persistent grid %d blocks of %d, wall clock %.0f MHz\n", P.grid, WG, P.rate / 1e6);
    for (const Shape& sh : TABLE_SHAPES) {
        if (sh.N > 16384) continue;                    // (the vocab heads: one per token, nothing to amortize)
        Copies c; make_copies(c, sh, fmt, o.warm);
        std::vector<Seg> hs; std::vector<Stage> hst;
        for (int i = 0; i < S; i++) { hs.push_back(seg_of(c.t[i % c.t.size()], x, y, 0)); hst.push_back({i, 1, host_ntiles(sh.N)}); }
        Seg* d; Stage* ds; CHECK(hipMalloc(&d, hs.size() * sizeof(Seg))); CHECK(hipMalloc(&ds, hst.size() * sizeof(Stage)));
        CHECK(hipMemcpy(d, hs.data(), hs.size() * sizeof(Seg), hipMemcpyHostToDevice)); CHECK(hipMemcpy(ds, hst.data(), hst.size() * sizeof(Stage), hipMemcpyHostToDevice));
        std::vector<double> stage_us, launch_us, pers_total_us;
        hipEvent_t a, b; CHECK(hipEventCreate(&a)); CHECK(hipEventCreate(&b));
        for (int rep = 0; rep < std::max(3, o.reps / 40); rep++) {
            // back-to-back launches of the same S matvecs (the amortized per-launch cost, launch gaps included)
            CHECK(hipEventRecord(a, st));
            for (int i = 0; i < S; i++) hipLaunchKernelGGL(k_launch, dim3(host_ntiles(sh.N)), dim3(WG), 0, st, d + i, 1, 0);
            CHECK(hipEventRecord(b, st)); CHECK(hipEventSynchronize(b));
            float ms; CHECK(hipEventElapsedTime(&ms, a, b)); launch_us.push_back(ms * 1e3 / S);
            // the same S matvecs as stages of one persistent kernel
            CHECK(hipEventRecord(a, st)); P.run(st, ds, S, d); CHECK(hipEventRecord(b, st)); CHECK(hipEventSynchronize(b));
            CHECK(hipEventElapsedTime(&ms, a, b)); pers_total_us.push_back(ms * 1e3 / S);
            if (!P.ok()) { fprintf(stderr, "persistent: a device sync timed out\n"); return 4; }
            std::vector<uint64_t> h(S + 1); CHECK(hipMemcpy(h.data(), P.stamps, (S + 1) * sizeof(uint64_t), hipMemcpyDeviceToHost));
            for (int i = 1; i < S; i++) stage_us.push_back((double)(h[i] - h[i - 1]) * 1e6 / P.rate);
        }
        const double bytes = tensor_bytes(sh.N, sh.K, fmt);
        char row[512];
        snprintf(row, sizeof row, "pshapes,%s,%s,%d,%d,\"%s\",%.0f,%d,%.2f,%.2f,%.2f,%.2f", o.fmt.c_str(), o.warm ? "warm" : "cold", sh.N, sh.K, sh.label, bytes, S,
                 median(launch_us), median(pers_total_us), median(stage_us), pct(stage_us, 0.9));
        csv_row(o, row);
        CHECK(hipEventDestroy(a)); CHECK(hipEventDestroy(b)); CHECK(hipFree(d)); CHECK(hipFree(ds)); c.a.release();
    }
    printf("# columns: mode,fmt,cache,N,K,label,weight_bytes,stages,launch_us_each(back-to-back),persistent_us_each(event),persistent_stage_us_median,stage_p90\n");
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
static void build_model(TokenModel& m, int fmt, int layers) {
    m.fmt = fmt; m.layers = layers; m.nexp = fmt == FMT_I4 ? NEXP : TOPK;   // f16: only the 8 experts a token uses fit
    size_t need = 0;
    auto add = [&](int r, int K) { need += tensor_alloc_bytes(r, K, fmt); };
    for (int l = 0; l < layers; l++) { add(QD, HID); add(KVD, HID); add(KVD, HID); add(HID, QD); add(NEXP, HID); for (int e = 0; e < m.nexp; e++) { add(EFF, HID); add(EFF, HID); add(HID, EFF); } }
    add(VOCAB, HID);
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
    m.head = make_tensor(m.a, VOCAB, HID, fmt, seed++);
    CHECK(hipDeviceSynchronize());
    printf("# token model: %s, %d layers, %d experts per layer allocated, %.2f GB\n", fmt ? "i4" : "f16", layers, m.nexp, m.a.used / 1e9);
}

struct TokenPlan { std::vector<Seg> segs; std::vector<Stage> stages; size_t bytes = 0; };
static TokenPlan plan_token(const TokenModel& m, std::mt19937& rng, const __half* xh, const __half* xq, const __half* xe, float* ybase) {
    TokenPlan p; size_t yoff = 0;
    auto add_stage = [&](std::vector<const Tensor*> ts, const __half* x) {
        Stage s; s.seg0 = (int)p.segs.size(); s.nseg = (int)ts.size(); int tile = 0;
        for (const Tensor* t : ts) { p.segs.push_back(seg_of(*t, x, ybase + yoff, tile)); yoff += t->rows; tile += host_ntiles(t->rows); p.bytes += tensor_bytes(t->rows, t->K, t->fmt); }
        s.ntiles = tile; p.stages.push_back(s);
    };
    for (int l = 0; l < m.layers; l++) {
        add_stage({&m.q[l], &m.k[l], &m.v[l]}, xh);
        add_stage({&m.o[l]}, xq);
        add_stage({&m.router[l]}, xh);
        std::vector<int> ex(m.nexp); for (int e = 0; e < m.nexp; e++) ex[e] = e;
        if (m.nexp > TOPK) std::shuffle(ex.begin(), ex.end(), rng);
        std::vector<const Tensor*> gu, dn;
        for (int j = 0; j < TOPK; j++) { gu.push_back(&m.gate[l][ex[j]]); gu.push_back(&m.up[l][ex[j]]); dn.push_back(&m.down[l][ex[j]]); }
        add_stage(gu, xh);
        add_stage(dn, xe);
    }
    add_stage({&m.head}, xh);
    return p;
}

static int token_mode(const Opts& o) {
    const int fmt = fmt_of(o.fmt);
    hipStream_t st; CHECK(hipStreamCreate(&st));
    TokenModel m; build_model(m, fmt, o.layers);
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
        plans.push_back(plan_token(m, rng, xh, xq, xe, y));
        Seg* d; CHECK(hipMalloc(&d, plans[i].segs.size() * sizeof(Seg)));
        CHECK(hipMemcpy(d, plans[i].segs.data(), plans[i].segs.size() * sizeof(Seg), hipMemcpyHostToDevice)); dsegs.push_back(d);
    }
    const auto& S0 = plans[0].stages; const int nst = (int)S0.size(), nseg = (int)plans[0].segs.size();
    CHECK(hipMalloc(&dst, nst * sizeof(Stage))); CHECK(hipMemcpy(dst, S0.data(), nst * sizeof(Stage), hipMemcpyHostToDevice));
    Persist Pe; Pe.init(nst);
    printf("# token: %d matvecs in %d stages, %.3f GB of weights per token, persistent grid %d, %d routing samples\n", nseg, nst, plans[0].bytes / 1e9, Pe.grid, P);

    auto run_perop = [&](int p) { const auto& sg = plans[p].segs; for (int i = 0; i < nseg; i++) hipLaunchKernelGGL(k_launch, dim3(host_ntiles(sg[i].rows)), dim3(WG), 0, st, dsegs[p] + i, 1, sg[i].tile0); };
    auto run_grouped = [&](int p) { for (const Stage& s : S0) hipLaunchKernelGGL(k_launch, dim3(s.ntiles), dim3(WG), 0, st, dsegs[p] + s.seg0, s.nseg, 0); };
    auto capture = [&](auto fn, int p) { hipGraph_t g; hipGraphExec_t ge; CHECK(hipStreamBeginCapture(st, hipStreamCaptureModeGlobal)); fn(p); CHECK(hipStreamEndCapture(st, &g)); CHECK(hipGraphInstantiate(&ge, g, nullptr, nullptr, 0)); CHECK(hipGraphDestroy(g)); return ge; };
    std::vector<hipGraphExec_t> g_perop, g_grouped;
    for (int p = 0; p < P; p++) { g_perop.push_back(capture(run_perop, p)); g_grouped.push_back(capture(run_grouped, p)); }

    // correctness: the persistent outputs == the per-op outputs, bitwise (the same tile function on the same data)
    run_perop(0); CHECK(hipStreamSynchronize(st)); CHECK(hipMemcpy(y2, y, yfloats * 4, hipMemcpyDeviceToDevice));
    CHECK(hipMemset(y, 0, yfloats * 4)); Pe.run(st, dst, nst, dsegs[0]); CHECK(hipStreamSynchronize(st));
    if (!Pe.ok()) { fprintf(stderr, "persistent: a device sync timed out\n"); return 4; }
    { std::vector<float> a(yfloats), b(yfloats); CHECK(hipMemcpy(a.data(), y, yfloats * 4, hipMemcpyDeviceToHost)); CHECK(hipMemcpy(b.data(), y2, yfloats * 4, hipMemcpyDeviceToHost));
      const bool same = memcmp(a.data(), b.data(), yfloats * 4) == 0;
      printf("%s token: persistent outputs %s the per-op outputs bitwise (%zu floats)\n", same ? "PASS" : "FAIL", same ? "==" : "!=", yfloats);
      if (!same) return 5; }

    const char* names[5] = {"per-op", "per-op-graph", "grouped", "grouped-graph", "persistent"};
    std::vector<std::vector<double>> ms(5);
    hipEvent_t a, b; CHECK(hipEventCreate(&a)); CHECK(hipEventCreate(&b));
    const int reps = std::max(5, o.reps / 20);
    for (int rep = -2; rep < reps; rep++) {                       // (2 warmup rounds)
        for (int mo = 0; mo < 5; mo++) {
            const int p = (rep + 2 + mo) % P;
            CHECK(hipEventRecord(a, st));
            if (mo == 0) run_perop(p); else if (mo == 1) CHECK(hipGraphLaunch(g_perop[p], st));
            else if (mo == 2) run_grouped(p); else if (mo == 3) CHECK(hipGraphLaunch(g_grouped[p], st));
            else Pe.run(st, dst, nst, dsegs[p]);
            CHECK(hipEventRecord(b, st)); CHECK(hipEventSynchronize(b));
            if (mo == 4 && !Pe.ok()) { fprintf(stderr, "persistent: a device sync timed out\n"); return 4; }
            float t; CHECK(hipEventElapsedTime(&t, a, b)); if (rep >= 0) ms[mo].push_back(t);
        }
    }
    for (int mo = 0; mo < 5; mo++) {
        char row[512];
        const double med = median(ms[mo]);
        snprintf(row, sizeof row, "token,%s,%s,%d,%d,%d,%.0f,%.3f,%.3f,%.3f,%.1f", o.fmt.c_str(), names[mo], o.layers, nseg, nst, (double)plans[0].bytes,
                 med, pct(ms[mo], 0.1), pct(ms[mo], 0.9), plans[0].bytes / (med * 1e-3) / 1e9);
        csv_row(o, row);
    }
    // the persistent kernel's stage times (the last persistent run), by stage kind
    std::vector<uint64_t> h(nst + 1); CHECK(hipMemcpy(h.data(), Pe.stamps, (nst + 1) * sizeof(uint64_t), hipMemcpyDeviceToHost));
    const char* kinds[5] = {"q|k|v", "o", "router", "gate|up x8", "down x8"};
    std::vector<std::vector<double>> by(6);
    for (int i = 0; i < nst; i++) { const double us = (double)(h[i] - (i ? h[i - 1] : h[nst])) * 1e6 / Pe.rate; by[i == nst - 1 ? 5 : i % 5].push_back(us); }
    for (int k = 0; k < 6; k++) printf("# persistent stage %-11s median %.2f us (n=%zu)\n", k < 5 ? kinds[k] : "lm_head", median(by[k]), by[k].size());
    printf("# columns: mode,fmt,strategy,layers,matvecs,stages,weight_bytes,ms_median,p10,p90,effective_GB/s\n");
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: decode_floor check|shapes|sweep|pshapes|token [--fmt f16|i4] [--warm] [--reps N] [--stages N] [--layers N] [--tokens N] [--csv FILE]\n"); return 2; }
    Opts o; o.mode = argv[1];
    for (int i = 2; i < argc; i++) {
        std::string a = argv[i]; auto next = [&]() { if (i + 1 >= argc) { fprintf(stderr, "%s needs a value\n", a.c_str()); exit(2); } return std::string(argv[++i]); };
        if (a == "--fmt") o.fmt = next(); else if (a == "--warm") o.warm = true; else if (a == "--reps") o.reps = std::stoi(next());
        else if (a == "--stages") o.stages = std::stoi(next()); else if (a == "--layers") o.layers = std::stoi(next());
        else if (a == "--tokens") o.tokens = std::stoi(next()); else if (a == "--csv") o.csv = next();
        else { fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    if (o.fmt != "f16" && o.fmt != "i4") { fprintf(stderr, "--fmt f16|i4\n"); return 2; }
    int dev; CHECK(hipGetDevice(&dev)); hipDeviceProp_t p; CHECK(hipGetDeviceProperties(&p, dev));
    printf("# device %s (%s), %d CUs, decode_floor %s fmt=%s cache=%s\n", p.name, p.gcnArchName, p.multiProcessorCount, o.mode.c_str(), o.fmt.c_str(), o.warm ? "warm" : "cold");
    if (o.mode == "check") return check_mode(o);
    if (o.mode == "shapes" || o.mode == "sweep") return shapes_mode(o);
    if (o.mode == "pshapes") return pshapes_mode(o);
    if (o.mode == "token") return token_mode(o);
    fprintf(stderr, "unknown mode %s\n", o.mode.c_str()); return 2;
}
