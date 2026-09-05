// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#pragma once

#include <ATen/Parallel.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

namespace vllm::hybrid::detail {

constexpr int64_t kTransposedKPanel = 64;
constexpr int64_t kTransposedNPanel = 512;
constexpr int64_t kTransposedPrefetch = 8;

// AVX-512 cache traversal. Keep weight storage and GPU transfers
// contiguous and unchanged; only CPU traversal and FP32 scratch are different.
// Each output visits K in increasing order, with a single final BF16 rounding.
// Scratch belongs to the calling thread (the native task worker in production),
// is reused up to its high-water mark, and is released when that thread exits.
// OpenMP workers write disjoint N panels; simultaneous callers do not share it.
template <typename ComputePanel, typename StorePanel>
void run_transposed_blocked(uint16_t* y, int64_t M, int64_t N, int64_t K,
                            ComputePanel compute, StorePanel store) {
  TORCH_CHECK(M >= 0 && N >= 0 && K >= 0,
              "Hybrid transposed GEMM dimensions must be non-negative");
  if (M == 0 || N == 0) return;
  TORCH_CHECK(M <= std::numeric_limits<int64_t>::max() / N,
              "Hybrid transposed GEMM scratch size overflow");
  if (K == 0) {
    std::fill_n(y, M * N, uint16_t{0});
    return;
  }
  static thread_local std::vector<float> scratch;
  const auto elements = static_cast<size_t>(M * N);
  if (scratch.size() < elements) scratch.resize(elements);
  float* accum = scratch.data();
  const int64_t panels = N / kTransposedNPanel + (N % kTransposedNPanel != 0);
  at::parallel_for(0, panels, 1, [=](int64_t begin, int64_t end) {
    for (int64_t panel = begin; panel < end; ++panel) {
      const int64_t n0 = panel * kTransposedNPanel;
      const int64_t n_end = std::min(N, n0 + kTransposedNPanel);
      for (int64_t k0 = 0; k0 < K; k0 += kTransposedKPanel) {
        compute(accum, n0, n_end, k0, std::min(K, k0 + kTransposedKPanel));
      }
      store(accum, n0, n_end);
    }
  });
}

}  // namespace vllm::hybrid::detail
