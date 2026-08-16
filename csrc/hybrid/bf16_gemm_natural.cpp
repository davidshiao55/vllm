// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Natural-layout BF16 GEMM for Hybrid small-token CPU slices.
//
// x is (M, K), w is PyTorch-natural (N, K), and y is (M, N). The kernel
// vectorizes along contiguous K, accumulates in FP32, horizontally reduces,
// and emits BF16 RNE. Adjacent output rows share activation loads; groups of
// four tokens share weight loads. No packing, bias, or post-ops are used.

#include "bf16_kernel_utils.h"
#include "bf16_kernels.h"
#include "bf16_kernels_internal.h"

#include <ATen/Parallel.h>

#include <cstdint>

namespace vllm {
namespace hybrid {

namespace {

// Single-row fallback for an odd output count.
template <int M_TILE>
inline void dot_single(const uint16_t* x, const uint16_t* w, uint16_t* y,
                       int64_t K, int64_t N, int64_t m_start, int64_t n) {
  __m256 acc[M_TILE];
  for (int i = 0; i < M_TILE; ++i) {
    acc[i] = _mm256_setzero_ps();
  }

  const uint16_t* w_row = w + n * K;
  const int64_t K_main = K / 8 * 8;
  for (int64_t k = 0; k < K_main; k += 8) {
    const __m256 w_vec = detail::load_bf16x8(w_row + k);
    for (int i = 0; i < M_TILE; ++i) {
      const __m256 x_vec = detail::load_bf16x8(x + (m_start + i) * K + k);
      acc[i] = _mm256_fmadd_ps(x_vec, w_vec, acc[i]);
    }
  }

  float scalar_acc[M_TILE];
  for (int i = 0; i < M_TILE; ++i) {
    scalar_acc[i] = detail::horizontal_sum(acc[i]);
  }
  for (int64_t k = K_main; k < K; ++k) {
    const float w_s = detail::bf16_to_float(w_row[k]);
    for (int i = 0; i < M_TILE; ++i) {
      const float x_s = detail::bf16_to_float(x[(m_start + i) * K + k]);
      scalar_acc[i] += x_s * w_s;
    }
  }
  for (int i = 0; i < M_TILE; ++i) {
    y[(m_start + i) * N + n] = detail::float_to_bf16_rne(scalar_acc[i]);
  }
}

// Compute two adjacent output channels for up to four tokens. Each activation
// vector feeds both weight rows, and each weight vector feeds every token in
// the tile. At M_TILE=4 this needs eight accumulators plus two weights and one
// transient activation vector, staying below AVX2's 16-register budget.
template <int M_TILE>
inline void dot_pair(const uint16_t* x, const uint16_t* w, uint16_t* y,
                     int64_t K, int64_t N, int64_t m_start, int64_t n) {
  const uint16_t* w0 = w + n * K;
  const uint16_t* w1 = w0 + K;
  float sum0[M_TILE];
  float sum1[M_TILE];
  detail::dot_two_rows<M_TILE>(x, w0, w1, K, m_start, sum0, sum1);
  for (int i = 0; i < M_TILE; ++i) {
    y[(m_start + i) * N + n] = detail::float_to_bf16_rne(sum0[i]);
    y[(m_start + i) * N + n + 1] = detail::float_to_bf16_rne(sum1[i]);
  }
}

}  // namespace

// Public entry. `y = x @ w^T` where w is `(N, K)` row-major BF16
// (PyTorch-natural Linear weight layout) and x is `(M, K)` row-major
// BF16. Output y is `(M, N)` row-major BF16.
//
// Handles any (M, N, K): adjacent output pairs use M_TILE=4 plus one fused
// M_TILE=1/2/3 remainder. An odd final output uses the single-row fallback.
// Both paths retain the scalar K tail for arbitrary K.
void bf16_gemm_natural_avx2(const uint16_t* x, const uint16_t* w, uint16_t* y,
                            int64_t M, int64_t N, int64_t K) {
  const int64_t n_pairs = N / 2;
  at::parallel_for(0, n_pairs, /*grain=*/1,
                   [x, w, y, M, N, K](int64_t n_begin, int64_t n_end) {
                     for (int64_t pair = n_begin; pair < n_end; ++pair) {
                       const int64_t n = pair * 2;
                       int64_t m = 0;
                       for (; m + 4 <= M; m += 4) {
                         dot_pair<4>(x, w, y, K, N, m, n);
                       }
                       switch (M - m) {
                         case 3:
                           dot_pair<3>(x, w, y, K, N, m, n);
                           break;
                         case 2:
                           dot_pair<2>(x, w, y, K, N, m, n);
                           break;
                         case 1:
                           dot_pair<1>(x, w, y, K, N, m, n);
                           break;
                       }
                     }
                   });

  if (N % 2 != 0) {
    const int64_t n = N - 1;
    int64_t m = 0;
    for (; m + 4 <= M; m += 4) {
      dot_single<4>(x, w, y, K, N, m, n);
    }
    switch (M - m) {
      case 3:
        dot_single<3>(x, w, y, K, N, m, n);
        break;
      case 2:
        dot_single<2>(x, w, y, K, N, m, n);
        break;
      case 1:
        dot_single<1>(x, w, y, K, N, m, n);
        break;
    }
  }
}

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

}  // namespace hybrid
}  // namespace vllm
