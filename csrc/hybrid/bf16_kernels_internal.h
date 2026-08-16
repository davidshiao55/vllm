// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#pragma once

#include <cstdint>

namespace vllm {
namespace hybrid {

void bf16_gemm_natural_avx2(const uint16_t* x, const uint16_t* w, uint16_t* y,
                            int64_t M, int64_t N, int64_t K);
void bf16_gemm_transposed_avx2(const uint16_t* x, const uint16_t* w,
                               uint16_t* y, int64_t M, int64_t N, int64_t K);
void bf16_mlp_gate_up_silu_down_avx2(const uint16_t* x, const uint16_t* w_gate,
                                     const uint16_t* w_up,
                                     const uint16_t* w_down, uint16_t* y,
                                     uint16_t* z_scratch, int64_t M, int64_t H,
                                     int64_t I, int64_t O);

void bf16_gemm_natural_avx512(const uint16_t* x, const uint16_t* w, uint16_t* y,
                              int64_t M, int64_t N, int64_t K);
void bf16_gemm_transposed_avx512(const uint16_t* x, const uint16_t* w,
                                 uint16_t* y, int64_t M, int64_t N, int64_t K);
void bf16_mlp_gate_up_silu_down_avx512(const uint16_t* x,
                                       const uint16_t* w_gate,
                                       const uint16_t* w_up,
                                       const uint16_t* w_down, uint16_t* y,
                                       uint16_t* z_scratch, int64_t M,
                                       int64_t H, int64_t I, int64_t O);

}  // namespace hybrid
}  // namespace vllm
