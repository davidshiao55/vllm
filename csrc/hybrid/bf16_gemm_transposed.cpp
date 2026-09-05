// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Cache-blocked transposed BF16 GEMM: x[M,K] @ w[K,N], contiguous storage.
// BF16 expansion stays in the FMA loop. AVX2 retains its register-sized
// microtiles; the K/N cache traversal is shared with AVX-512.

#include "bf16_kernel_utils.h"
#include "bf16_kernels.h"
#include "bf16_kernels_internal.h"
#include "bf16_transposed_blocking.h"

namespace vllm {
namespace hybrid {
namespace {
template <int MT, int NV>
__attribute__((noinline)) void tile(const uint16_t* x, const uint16_t* w,
                                    float* s, int64_t K, int64_t N, int64_t m0,
                                    int64_t n0, int64_t k0, int64_t k_end,
                                    int pf) {
  __m256 a[MT][NV];
  for (int m = 0; m < MT; ++m)
    for (int j = 0; j < NV; ++j)
      a[m][j] = k0 ? _mm256_loadu_ps(s + (m0 + m) * N + n0 + j * 8)
                   : _mm256_setzero_ps();
  for (int64_t k = k0; k < k_end; ++k) {
    if (pf && k + pf < k_end)
      for (int j = 0; j < NV; j += 4)
        _mm_prefetch((const char*)(w + (k + pf) * N + n0 + j * 8), _MM_HINT_T0);
    __m256 weights[NV];
    for (int j = 0; j < NV; ++j)
      weights[j] = detail::load_bf16x8(w + k * N + n0 + j * 8);
    for (int m = 0; m < MT; ++m) {
      __m256 xv = _mm256_set1_ps(detail::bf16_to_float(x[(m0 + m) * K + k]));
      for (int j = 0; j < NV; ++j)
        a[m][j] = _mm256_fmadd_ps(xv, weights[j], a[m][j]);
    }
  }
  for (int m = 0; m < MT; ++m)
    for (int j = 0; j < NV; ++j)
      _mm256_storeu_ps(s + (m0 + m) * N + n0 + j * 8, a[m][j]);
}
// Explicit registers prevent GCC from emitting redundant per-K stack stores
// for the accumulator array. Arithmetic and reduction order are unchanged.
template <>
__attribute__((noinline)) void tile<4, 2>(const uint16_t* x, const uint16_t* w,
                                          float* s, int64_t K, int64_t N,
                                          int64_t m0, int64_t n0, int64_t k0,
                                          int64_t k_end, int pf) {
  float* s0 = s + m0 * N + n0;
  float* s1 = s0 + N;
  float* s2 = s1 + N;
  float* s3 = s2 + N;
  __m256 a0 = k0 ? _mm256_loadu_ps(s0) : _mm256_setzero_ps();
  __m256 a1 = k0 ? _mm256_loadu_ps(s0 + 8) : _mm256_setzero_ps();
  __m256 a2 = k0 ? _mm256_loadu_ps(s1) : _mm256_setzero_ps();
  __m256 a3 = k0 ? _mm256_loadu_ps(s1 + 8) : _mm256_setzero_ps();
  __m256 a4 = k0 ? _mm256_loadu_ps(s2) : _mm256_setzero_ps();
  __m256 a5 = k0 ? _mm256_loadu_ps(s2 + 8) : _mm256_setzero_ps();
  __m256 a6 = k0 ? _mm256_loadu_ps(s3) : _mm256_setzero_ps();
  __m256 a7 = k0 ? _mm256_loadu_ps(s3 + 8) : _mm256_setzero_ps();
  const uint16_t* x0 = x + m0 * K;
  for (int64_t k = k0; k < k_end; ++k) {
    if (pf && k + pf < k_end)
      _mm_prefetch((const char*)(w + (k + pf) * N + n0), _MM_HINT_T0);
    const __m256 w0 = detail::load_bf16x8(w + k * N + n0);
    const __m256 w1 = detail::load_bf16x8(w + k * N + n0 + 8);
    __m256 xv = _mm256_set1_ps(detail::bf16_to_float(x0[k]));
    a0 = _mm256_fmadd_ps(xv, w0, a0);
    a1 = _mm256_fmadd_ps(xv, w1, a1);
    xv = _mm256_set1_ps(detail::bf16_to_float(x0[K + k]));
    a2 = _mm256_fmadd_ps(xv, w0, a2);
    a3 = _mm256_fmadd_ps(xv, w1, a3);
    xv = _mm256_set1_ps(detail::bf16_to_float(x0[2 * K + k]));
    a4 = _mm256_fmadd_ps(xv, w0, a4);
    a5 = _mm256_fmadd_ps(xv, w1, a5);
    xv = _mm256_set1_ps(detail::bf16_to_float(x0[3 * K + k]));
    a6 = _mm256_fmadd_ps(xv, w0, a6);
    a7 = _mm256_fmadd_ps(xv, w1, a7);
  }
  _mm256_storeu_ps(s0, a0);
  _mm256_storeu_ps(s0 + 8, a1);
  _mm256_storeu_ps(s1, a2);
  _mm256_storeu_ps(s1 + 8, a3);
  _mm256_storeu_ps(s2, a4);
  _mm256_storeu_ps(s2 + 8, a5);
  _mm256_storeu_ps(s3, a6);
  _mm256_storeu_ps(s3 + 8, a7);
}
template <int MT, int NV>
void panel(const uint16_t* x, const uint16_t* w, float* s, int64_t K, int64_t N,
           int64_t m, int64_t begin, int64_t end, int64_t k0, int64_t k_end,
           int pf) {
  int64_t n = begin;
  for (; n + NV * 8 <= end; n += NV * 8)
    tile<MT, NV>(x, w, s, K, N, m, n, k0, k_end, pf);
  for (; n < end; ++n)
    for (int r = 0; r < MT; ++r) {
      float a = k0 ? s[(m + r) * N + n] : 0;
      for (int64_t k = k0; k < k_end; ++k)
        a = __builtin_fmaf(detail::bf16_to_float(x[(m + r) * K + k]),
                           detail::bf16_to_float(w[k * N + n]), a);
      s[(m + r) * N + n] = a;
    }
}

}  // namespace

void bf16_gemm_transposed_avx2(const uint16_t* x, const uint16_t* w,
                               uint16_t* y, int64_t M, int64_t N, int64_t K) {
  detail::run_transposed_blocked(
      y, M, N, K,
      [=](float* s, int64_t n0, int64_t n_end, int64_t k0, int64_t k_end) {
        constexpr int pf = detail::kTransposedPrefetch;
        int64_t m = 0;
        for (; m + 4 <= M; m += 4)
          panel<4, 2>(x, w, s, K, N, m, n0, n_end, k0, k_end, pf);
        switch (M - m) {
          case 3:
            panel<3, 4>(x, w, s, K, N, m, n0, n_end, k0, k_end, pf);
            break;
          case 2:
            panel<2, 4>(x, w, s, K, N, m, n0, n_end, k0, k_end, pf);
            break;
          case 1:
            if (M == 1)
              panel<1, 8>(x, w, s, K, N, m, n0, n_end, k0, k_end, pf);
            else
              panel<1, 2>(x, w, s, K, N, m, n0, n_end, k0, k_end, pf);
        }
      },
      [=](const float* s, int64_t n0, int64_t n_end) {
        for (int64_t m = 0; m < M; ++m) {
          int64_t n = n0;
          for (; n + 8 <= n_end; n += 8)
            _mm_storeu_si128(
                reinterpret_cast<__m128i*>(y + m * N + n),
                detail::float8_to_bf16_rne(_mm256_loadu_ps(s + m * N + n)));
          for (; n < n_end; ++n)
            y[m * N + n] = detail::float_to_bf16_rne(s[m * N + n]);
        }
      });
}

// at::Tensor entry. Validates dtype + contiguity then dispatches
// to the raw-pointer kernel. Designed to be called from
// HybridWeightTaskRunner's MLP-block worker for the transposed-storage path
// (Stage 7-C). Caller passes:
//   * x: (M, K) BF16, row-major contiguous.
//   * w: (K, N) BF16, row-major contiguous (this is the
//     transposed-storage layout — `w_cpu_t` from Stage 7's
//     storage unification proposal).
//   * y_out: (M, N) BF16, row-major contiguous, pre-allocated.
//
// Output is written in-place into y_out.
void bf16_gemm_transposed_at(const at::Tensor& x, const at::Tensor& w,
                             at::Tensor& y_out) {
  TORCH_CHECK(x.dtype() == at::kBFloat16, "x must be bfloat16, got ",
              x.dtype());
  TORCH_CHECK(w.dtype() == at::kBFloat16, "w must be bfloat16, got ",
              w.dtype());
  TORCH_CHECK(y_out.dtype() == at::kBFloat16, "y_out must be bfloat16, got ",
              y_out.dtype());
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && y_out.dim() == 2,
              "all tensors must be 2D");
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && y_out.is_contiguous(),
              "all tensors must be row-major contiguous (transposed-storage "
              "layout: w shape is (K, N), not (N, K))");
  const int64_t M = x.size(0);
  const int64_t K = x.size(1);
  TORCH_CHECK(w.size(0) == K, "w.size(0)=", w.size(0),
              " must equal x.size(1)=", K);
  const int64_t N = w.size(1);
  TORCH_CHECK(y_out.size(0) == M && y_out.size(1) == N,
              "y_out shape mismatch: expected (", M, ",", N, ") got (",
              y_out.size(0), ",", y_out.size(1), ")");

  bf16_gemm_transposed(reinterpret_cast<const uint16_t*>(x.data_ptr()),
                       reinterpret_cast<const uint16_t*>(w.data_ptr()),
                       reinterpret_cast<uint16_t*>(y_out.data_ptr()), M, N, K);
}

}  // namespace hybrid
}  // namespace vllm
