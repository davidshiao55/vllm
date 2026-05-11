// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Stage 7-D — BF16 GEMM on natural `(N, K)` row-major weight layout
// (PyTorch's default `nn.Linear` weight storage:
// `(out_features, in_features)` row-major). Sibling to
// `bf16_gemm_transposed.cpp`, which operates on the transposed
// `(K, N)` layout (used by down-proj after Stage 7-C).
//
// Same casting + parallelism strategy as the transposed kernel —
// pipelined BF16→FP32 upcast in the inner FMA loop, no scratch
// buffer, no allocator round-trip per call; intra-op parallelism
// via `at::parallel_for`. The only structural difference is the
// loop nest, which matches the fast access direction of each
// storage layout (outer-K wide-N for transposed; outer-N inner-K
// dot-product for natural).
//
// What this kernel is NOT (same disclaimers as the transposed
// sibling):
//   * Not a oneDNN BRGEMM port — no JIT, no oneDNN packing
//     (`n16c` blocked layout), no post-ops, no quantization
//     scales, no runtime ISA dispatch, no AMX/AVX-512 paths.
//   * Not a general-purpose GEMM — only the M ∈ {1, 4, ...} cases
//     COTS dispatches; other M handled by a per-m fallback that
//     gives up M-fusion (cheap at small M tail).
//
// ─────────────────────────────────────────────────────────────
// Conceptual relationship to oneDNN's f32:bf16 BRGEMM
// ─────────────────────────────────────────────────────────────
//
// oneDNN's BRGEMM inner kernel for f32:bf16 on the natural layout
// (`oneDNN/src/cpu/x64/brgemm/jit_brgemm_kernel.cpp:2880-2888`)
// uses an M_blk × N_blk register tile with inner K and reads the
// weight 8-BF16-at-a-time via:
//     uni_vpmovzxwd(vmm_load, addr);    // BF16 → 32-bit zero-ext
//     uni_vpslld(vmm_load, vmm_load,16) // shift to FP32 mantissa
//     vfmadd231ps(vmm_acc, vmm_x, vmm_load)
//
// oneDNN cannot read 8 consecutive BF16s directly from a
// `(N, K)` storage — those 8 BF16s are stride-K in memory, not
// contiguous. oneDNN therefore does an **internal repack** at
// primitive setup, transforming `(N, K)` into a blocked `n16c`
// layout (16 consecutive N-rows interleaved into a single
// contiguous 16-element strip per K position). The packed buffer
// is cached internally and reused across calls.
//
// We do NOT repack at install (user constraint: keep natural
// layout for QKV/gate/up). So we cannot use oneDNN's M×N
// register-tile structure directly — the wide N-vector load would
// be 16 stride-K reads.
//
// Our adaptation: **swap the loop nest**. Use outer N, inner K
// vectorize over K (the sequential direction in natural layout).
// Each (m, n) pair gets an 8-wide FP32 accumulator; the inner K
// loop walks k 8 at a time. Both x[m, :] and w[n, :] are read
// sequentially. At end of K, horizontal-reduce the 8-wide acc to
// a scalar y[m, n]. M-fusion still works: one w-row load feeds
// M_TILE FMAs in parallel (M_TILE = 4).
//
// The **inner BF16 → FP32 upcast sequence** is identical to
// oneDNN's: `vpmovzxwd + vpslld 16`. That part is just SIMD
// arithmetic; whether the kernel structure around it is
// M×N-register-tile (oneDNN packed) or N-outer-K-inner-dot-product
// (us, on natural layout) doesn't change the per-byte cost of
// the upcast itself.
//
// Loss vs oneDNN's tuned BRGEMM: one horizontal reduce per output
// (~5-10 cycles) instead of zero. Win vs `at::linear` on
// `bf16:bf16`: no src reorder, no ATen+oneDNN dispatch, no extra
// DRAM pass for src (same wins the transposed sibling gets).
//
// Measured performance (i9-14900KF, Qwen2.5-7B QKV/MLP1 shapes):
// 0.41× of oneDNN's at::linear wall at thr=4 across B ∈ {4, 32,
// 256} — 2.4× faster than the production at::linear path.
// Stage 7-D probe data lives in
// `David/Tests/phase1c/results/stage7_natural_probe.json`.

#include <ATen/ATen.h>
#include <ATen/Parallel.h>

#include <algorithm>
#include <cstdint>
#include <cstring>

#if defined(__AVX2__) && defined(__FMA__)
  #include <immintrin.h>
  #define VLLM_COTS_HAS_AVX2_FMA 1
#else
  #define VLLM_COTS_HAS_AVX2_FMA 0
#endif

namespace vllm {
namespace cots {

#if VLLM_COTS_HAS_AVX2_FMA

namespace {

// 8-wide BF16 → FP32 upcast. Same instruction sequence as
// oneDNN's f32:bf16 BRGEMM inner kernel (`vpmovzxwd + vpslld 16`).
inline __m256 load8_bf16_as_f32(const uint16_t* ptr) {
  __m128i bf16x8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr));
  __m256i ext = _mm256_cvtepu16_epi32(bf16x8);
  __m256i shl = _mm256_slli_epi32(ext, 16);
  return _mm256_castsi256_ps(shl);
}

// Horizontal reduce of an 8-wide FP32 vector to a scalar FP32.
// Standard AVX2 reduction tree: pair-add halves, then within
// 128-bit lane. ~5-10 cycles depending on the dispatch port mix.
inline float hreduce_ps(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v);
  __m128 hi = _mm256_extractf128_ps(v, 1);
  __m128 s = _mm_add_ps(lo, hi);   // 4 partial sums
  __m128 sh = _mm_movehdup_ps(s);  // duplicate odd lanes
  __m128 s2 = _mm_add_ps(s, sh);
  sh = _mm_movehl_ps(sh, s2);
  __m128 s3 = _mm_add_ss(s2, sh);
  return _mm_cvtss_f32(s3);
}

// Scalar BF16 → FP32 — reinterpret BF16 bits as the high 16 of
// FP32, low 16 zero. Used by the K-tail loop where K % 8 trailing
// elements are summed in scalar FP32.
inline float bf16_to_f32_scalar(uint16_t b) {
  uint32_t u = static_cast<uint32_t>(b) << 16;
  float f;
  std::memcpy(&f, &u, sizeof(f));
  return f;
}

// Scalar FP32 → BF16 RNE (round to nearest, ties to even).
// Same RNE formula as f32_to_bf16_rne in the transposed sibling —
// bias by `0x7FFF + LSB_of_target_mantissa` then shift right 16.
inline uint16_t f32_to_bf16_rne_scalar(float f) {
  uint32_t u;
  std::memcpy(&u, &f, sizeof(u));
  const uint32_t lsb = (u >> 16) & 1u;
  u += 0x7FFFu + lsb;
  return static_cast<uint16_t>(u >> 16);
}

// Inner kernel: compute `M_TILE` dot products at column `n` of
// the output. y[m_start + i, n] = sum_k x[m_start+i, k] * w[n, k]
// for i in 0..M_TILE-1. Handles K of any value via a 8-wide main
// loop + scalar K-tail (K % 8 trailing elements).
template <int M_TILE>
inline void dot_tile(const uint16_t* x, const uint16_t* w, uint16_t* y,
                     int64_t K, int64_t N, int64_t m_start, int64_t n) {
  __m256 acc[M_TILE];
  for (int i = 0; i < M_TILE; ++i) {
    acc[i] = _mm256_setzero_ps();
  }

  const uint16_t* w_row = w + n * K;  // sequential along k
  const int64_t K_main = (K / 8) * 8;
  for (int64_t k = 0; k < K_main; k += 8) {
    // Load 8 BF16 weights from w[n, k..k+7] — SEQUENTIAL.
    // Same inner upcast as oneDNN's f32:bf16 BRGEMM.
    const __m256 w_vec = load8_bf16_as_f32(w_row + k);

    // M-fusion: this single w-load feeds M_TILE FMAs in parallel.
    // Each m reads its own x row (also sequential along k).
    for (int i = 0; i < M_TILE; ++i) {
      const __m256 x_vec = load8_bf16_as_f32(x + (m_start + i) * K + k);
      acc[i] = _mm256_fmadd_ps(x_vec, w_vec, acc[i]);
    }
  }

  // Reduce vector accumulators to scalars first, then add the
  // K-tail (K % 8 trailing elements) in scalar FP32. Production
  // shapes have K ∈ {hidden=3584, intermediate=18944, qkv_dim
  // slices, ...} — all multiples of 8 — so the tail is normally
  // a no-op. Kept for correctness on arbitrary K.
  float scalar_acc[M_TILE];
  for (int i = 0; i < M_TILE; ++i) {
    scalar_acc[i] = hreduce_ps(acc[i]);
  }
  for (int64_t k = K_main; k < K; ++k) {
    const float w_s = bf16_to_f32_scalar(w_row[k]);
    for (int i = 0; i < M_TILE; ++i) {
      const float x_s = bf16_to_f32_scalar(x[(m_start + i) * K + k]);
      scalar_acc[i] += x_s * w_s;
    }
  }
  for (int i = 0; i < M_TILE; ++i) {
    y[(m_start + i) * N + n] = f32_to_bf16_rne_scalar(scalar_acc[i]);
  }
}

}  // namespace

// Public entry. `y = x @ w^T` where w is `(N, K)` row-major BF16
// (PyTorch-natural Linear weight layout) and x is `(M, K)` row-major
// BF16. Output y is `(M, N)` row-major BF16.
//
// Handles any (M, N, K): M_TILE=4 register-tile path for groups
// of 4 m's + per-m fallback for the M % 4 tail; 8-wide K main loop
// + scalar K-tail for arbitrary K. No N tail because we walk N
// one column at a time (output is a scalar per (m, n)).
void bf16_gemm_natural(const uint16_t* x, const uint16_t* w, uint16_t* y,
                       int64_t M, int64_t N, int64_t K) {
  // Parallelize on N. Each n-iteration computes M dot products
  // independently; threads write disjoint output columns of y,
  // so no contention. Grain=1 means "one N per task" — at::parallel_for
  // dynamically merges if work is small enough.
  at::parallel_for(0, N, /*grain=*/1,
                   [x, w, y, M, N, K](int64_t n_begin, int64_t n_end) {
                     for (int64_t n = n_begin; n < n_end; ++n) {
                       int64_t m = 0;
                       // M_TILE=4 fast path while we have at least 4 m's left.
                       for (; m + 4 <= M; m += 4) {
                         dot_tile<4>(x, w, y, K, N, m, n);
                       }
                       // M_TILE=1 fallback for the remainder.
                       for (; m < M; ++m) {
                         dot_tile<1>(x, w, y, K, N, m, n);
                       }
                     }
                   });
}

#else  // !VLLM_COTS_HAS_AVX2_FMA

// Scalar fallback for non-AVX2 builds. Correctness only.
void bf16_gemm_natural(const uint16_t* x, const uint16_t* w, uint16_t* y,
                       int64_t M, int64_t N, int64_t K) {
  auto bf16_to_f32 = [](uint16_t b) -> float {
    uint32_t u = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
  };
  auto f32_to_bf16 = [](float f) -> uint16_t {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    const uint32_t lsb = (u >> 16) & 1u;
    u += 0x7FFFu + lsb;
    return static_cast<uint16_t>(u >> 16);
  };
  for (int64_t m = 0; m < M; ++m) {
    for (int64_t n = 0; n < N; ++n) {
      float acc = 0.0f;
      for (int64_t k = 0; k < K; ++k) {
        acc += bf16_to_f32(x[m * K + k]) * bf16_to_f32(w[n * K + k]);
      }
      y[m * N + n] = f32_to_bf16(acc);
    }
  }
}

#endif

// at::Tensor entry. Validates dtype, dimensions, contiguity.
// Caller-managed tensors.
void bf16_gemm_natural_at(const at::Tensor& x, const at::Tensor& w,
                          at::Tensor& y_out) {
  TORCH_CHECK(x.dtype() == at::kBFloat16, "x bf16, got ", x.dtype());
  TORCH_CHECK(w.dtype() == at::kBFloat16, "w bf16, got ", w.dtype());
  TORCH_CHECK(y_out.dtype() == at::kBFloat16, "y bf16, got ", y_out.dtype());
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && y_out.dim() == 2);
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && y_out.is_contiguous(),
              "all tensors must be row-major contiguous (natural layout: "
              "w shape is (N, K))");
  const int64_t M = x.size(0);
  const int64_t K = x.size(1);
  TORCH_CHECK(w.size(1) == K, "w.size(1) must equal x.size(1) (K)");
  const int64_t N = w.size(0);
  TORCH_CHECK(y_out.size(0) == M && y_out.size(1) == N, "y_out shape mismatch");

  bf16_gemm_natural(reinterpret_cast<const uint16_t*>(x.data_ptr()),
                    reinterpret_cast<const uint16_t*>(w.data_ptr()),
                    reinterpret_cast<uint16_t*>(y_out.data_ptr()), M, N, K);
}

}  // namespace cots
}  // namespace vllm
