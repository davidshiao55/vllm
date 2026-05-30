// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Shared host-side helpers for COTS native task runners.

#ifndef VLLM_COTS_COMMON_H_
#define VLLM_COTS_COMMON_H_

#include <c10/util/Exception.h>
#include <cuda_runtime_api.h>
#include <nvtx3/nvToolsExt.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>

namespace vllm {
namespace cots {

inline bool env_flag(const char* name) {
  const char* v = std::getenv(name);
  return v != nullptr && v[0] == '1' && v[1] == '\0';
}

inline bool cots_nvtx_enabled() {
  static const bool enabled = []() {
    return env_flag("VLLM_COTS_DIAG") || env_flag("VLLM_COTS_NVTX");
  }();
  return enabled;
}

struct NvtxScope {
  explicit NvtxScope(const char* name) : active_(cots_nvtx_enabled()) {
    if (active_) nvtxRangePushA(name);
  }
  ~NvtxScope() {
    if (active_) nvtxRangePop();
  }
  NvtxScope(const NvtxScope&) = delete;
  NvtxScope& operator=(const NvtxScope&) = delete;

 private:
  bool active_;
};

inline int64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

inline void ensure_mapped_i64_cell(int64_t** host_ptr, int64_t** dev_ptr,
                                   const char* context, const char* name) {
  if (*host_ptr != nullptr) return;
  void* hp = nullptr;
  cudaError_t e = cudaHostAlloc(&hp, sizeof(int64_t), cudaHostAllocMapped);
  TORCH_CHECK(e == cudaSuccess, context, ": cudaHostAlloc(", name,
              ") failed: ", cudaGetErrorString(e));
  *static_cast<int64_t*>(hp) = 0;

  void* dp = nullptr;
  cudaError_t e2 = cudaHostGetDevicePointer(&dp, hp, 0);
  if (e2 != cudaSuccess) {
    cudaFreeHost(hp);
    TORCH_CHECK(false, context, ": cudaHostGetDevicePointer(", name,
                ") failed: ", cudaGetErrorString(e2));
  }
  *host_ptr = static_cast<int64_t*>(hp);
  *dev_ptr = static_cast<int64_t*>(dp);
}

extern "C" void launch_cots_wait_done_kernel_production(uint32_t* dev_req_slot,
                                                        uint32_t* dev_done_slot,
                                                        cudaStream_t stream);
extern "C" void launch_cots_wait_done_kernel_diag(
    uint32_t* dev_req_slot, uint32_t* dev_done_slot, int64_t* spin_iters_total,
    int64_t* immediate_count, int64_t* lagging_count, cudaStream_t stream);

inline void launch_wait_done_kernel(uint32_t* dev_req_slot,
                                    uint32_t* dev_done_slot, bool diag_enabled,
                                    int64_t* spin_iters_dev,
                                    int64_t* immediate_resume_dev,
                                    int64_t* lagging_wait_dev,
                                    cudaStream_t stream, const char* context) {
  if (diag_enabled) {
    TORCH_CHECK(spin_iters_dev != nullptr && immediate_resume_dev != nullptr &&
                    lagging_wait_dev != nullptr,
                context,
                ": diag mode active but wait-kernel counter cells are not "
                "allocated");
    launch_cots_wait_done_kernel_diag(dev_req_slot, dev_done_slot,
                                      spin_iters_dev, immediate_resume_dev,
                                      lagging_wait_dev, stream);
  } else {
    launch_cots_wait_done_kernel_production(dev_req_slot, dev_done_slot,
                                            stream);
  }

  cudaError_t le = cudaGetLastError();
  TORCH_CHECK(le == cudaSuccess, context,
              ": kernel launch failed: ", cudaGetErrorString(le));
}

}  // namespace cots
}  // namespace vllm

#endif  // VLLM_COTS_COMMON_H_
