// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#pragma once

#include <ATen/ATen.h>

#include <cstdint>

namespace vllm {
namespace hybrid {

// y[M, N] = x[M, K] @ w[N, K]^T.
void bf16_gemm_natural(const uint16_t* x, const uint16_t* w, uint16_t* y,
                       int64_t M, int64_t N, int64_t K);

// y[M, N] = x[M, K] @ w[K, N].
void bf16_gemm_transposed(const uint16_t* x, const uint16_t* w, uint16_t* y,
                          int64_t M, int64_t N, int64_t K);

void bf16_gemm_natural_at(const at::Tensor& x, const at::Tensor& w,
                          at::Tensor& y_out);
void bf16_gemm_transposed_at(const at::Tensor& x, const at::Tensor& w,
                             at::Tensor& y_out);

// gate/up use natural (I, H) weights; down uses transposed (I, O) storage.
void bf16_mlp_gate_up_silu_down(const uint16_t* x, const uint16_t* w_gate,
                                const uint16_t* w_up, const uint16_t* w_down,
                                uint16_t* y, uint16_t* z_scratch, int64_t M,
                                int64_t H, int64_t I, int64_t O);

// Runtime-selected implementation for the complete hybrid BF16 kernel table.
const char* bf16_kernel_isa();

}  // namespace hybrid
}  // namespace vllm
