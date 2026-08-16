// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include "bf16_kernels.h"

#include "bf16_kernels_internal.h"

namespace vllm {
namespace hybrid {

namespace {

using GemmKernel = void (*)(const uint16_t*, const uint16_t*, uint16_t*,
                            int64_t, int64_t, int64_t);
using MlpKernel = void (*)(const uint16_t*, const uint16_t*, const uint16_t*,
                           const uint16_t*, uint16_t*, uint16_t*, int64_t,
                           int64_t, int64_t, int64_t);

struct KernelTable {
  GemmKernel natural;
  GemmKernel transposed;
  MlpKernel mlp;
  const char* isa;
};

bool cpu_supports_avx512() {
#if (defined(__x86_64__) || defined(_M_X64)) && \
    (defined(__GNUC__) || defined(__clang__))
  __builtin_cpu_init();
  return __builtin_cpu_supports("avx512f") &&
         __builtin_cpu_supports("avx512bw");
#else
  return false;
#endif
}

const KernelTable& kernels() {
  static const KernelTable selected = []() {
    if (cpu_supports_avx512()) {
      return KernelTable{bf16_gemm_natural_avx512, bf16_gemm_transposed_avx512,
                         bf16_mlp_gate_up_silu_down_avx512, "avx512"};
    }
    return KernelTable{bf16_gemm_natural_avx2, bf16_gemm_transposed_avx2,
                       bf16_mlp_gate_up_silu_down_avx2, "avx2"};
  }();
  return selected;
}

}  // namespace

void bf16_gemm_natural(const uint16_t* x, const uint16_t* w, uint16_t* y,
                       int64_t M, int64_t N, int64_t K) {
  kernels().natural(x, w, y, M, N, K);
}

void bf16_gemm_transposed(const uint16_t* x, const uint16_t* w, uint16_t* y,
                          int64_t M, int64_t N, int64_t K) {
  kernels().transposed(x, w, y, M, N, K);
}

void bf16_mlp_gate_up_silu_down(const uint16_t* x, const uint16_t* w_gate,
                                const uint16_t* w_up, const uint16_t* w_down,
                                uint16_t* y, uint16_t* z_scratch, int64_t M,
                                int64_t H, int64_t I, int64_t O) {
  kernels().mlp(x, w_gate, w_up, w_down, y, z_scratch, M, H, I, O);
}

const char* bf16_kernel_isa() { return kernels().isa; }

}  // namespace hybrid
}  // namespace vllm
