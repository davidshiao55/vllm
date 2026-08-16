// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Hybrid CPU MLP block for decode/small-batch CPU compute.
//
// Computes:
//
//   z = silu(x @ w_gate^T) * (x @ w_up^T)
//   y = z @ w_down
//
// with BF16 x/weights/output, natural row-major gate/up weights `(I, H)`,
// transposed row-major down weights `(I, O)`, and worker-owned BF16 scratch for
// z. The gate/up/SwiGLU part is fused into one pass over x and the two input
// matrices; the down projection intentionally reuses the existing optimized
// BF16 transposed GEMM.

#include "bf16_kernel_utils.h"
#include "bf16_kernels.h"
#include "bf16_kernels_internal.h"

#include <ATen/Parallel.h>

#include <cmath>
#include <cstdint>

namespace vllm {
namespace hybrid {

namespace {

inline float silu(float x) { return x / (1.0f + std::exp(-x)); }

template <int M_TILE>
inline void gate_up_silu_tile(const uint16_t* x, const uint16_t* w_gate,
                              const uint16_t* w_up, uint16_t* z_scratch,
                              int64_t H, int64_t I, int64_t m_start,
                              int64_t i) {
  const uint16_t* wg = w_gate + i * H;
  const uint16_t* wu = w_up + i * H;
  float gate_s[M_TILE];
  float up_s[M_TILE];
  detail::dot_two_rows<M_TILE>(x, wg, wu, H, m_start, gate_s, up_s);

  for (int m = 0; m < M_TILE; ++m) {
    const uint16_t gate_b = detail::float_to_bf16_rne(gate_s[m]);
    const uint16_t up_b = detail::float_to_bf16_rne(up_s[m]);
    const float gate_r = detail::bf16_to_float(gate_b);
    const float up_r = detail::bf16_to_float(up_b);
    z_scratch[(m_start + m) * I + i] =
        detail::float_to_bf16_rne(silu(gate_r) * up_r);
  }
}

void gate_up_silu_bf16_scratch(const uint16_t* x, const uint16_t* w_gate,
                               const uint16_t* w_up, uint16_t* z_scratch,
                               int64_t M, int64_t H, int64_t I) {
  at::parallel_for(
      0, I, /*grain=*/1,
      [x, w_gate, w_up, z_scratch, M, H, I](int64_t i_begin, int64_t i_end) {
        for (int64_t i = i_begin; i < i_end; ++i) {
          int64_t m = 0;
          for (; m + 4 <= M; m += 4) {
            gate_up_silu_tile<4>(x, w_gate, w_up, z_scratch, H, I, m, i);
          }
          switch (M - m) {
            case 3:
              gate_up_silu_tile<3>(x, w_gate, w_up, z_scratch, H, I, m, i);
              break;
            case 2:
              gate_up_silu_tile<2>(x, w_gate, w_up, z_scratch, H, I, m, i);
              break;
            case 1:
              gate_up_silu_tile<1>(x, w_gate, w_up, z_scratch, H, I, m, i);
              break;
          }
        }
      });
}

}  // namespace

void bf16_mlp_gate_up_silu_down_avx2(const uint16_t* x, const uint16_t* w_gate,
                                     const uint16_t* w_up,
                                     const uint16_t* w_down, uint16_t* y,
                                     uint16_t* z_scratch, int64_t M, int64_t H,
                                     int64_t I, int64_t O) {
  if (M <= 0 || H <= 0 || I <= 0 || O <= 0) {
    return;
  }
  gate_up_silu_bf16_scratch(x, w_gate, w_up, z_scratch, M, H, I);
  bf16_gemm_transposed_avx2(z_scratch, w_down, y, M, O, I);
}

}  // namespace hybrid
}  // namespace vllm
