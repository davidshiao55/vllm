// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// §1c.29 wait-kernel sync — captured wait kernel that replaces the
// `cudaLaunchHostFunc(sync_cb)` node in the COTS captured graph
// when `cots_capture_sync_mode="wait_kernel"`. Compiled as CUDA so nvcc
// handles `__global__` and PTX inline asm; rest of `_cots_C`
// stays as plain C++ host code under the same target language.
//
// Two kernels — production and diag — to keep the production
// hot path branch-free. The diag kernel is launched only when
// VLLM_COTS_DIAG=1 (decision made on the host side, captured
// into the graph at trace time).

#include <cuda_runtime.h>

#include <cstdint>

namespace vllm {
namespace cots {

namespace {

// Production wait kernel. Reads the captured-stream's expected
// seq from req_slot, then spins on done_slot until the worker
// publishes >= expected. Uses PTX `nanosleep` (sm_70+) to avoid
// burning SM cycles on the spin loop.
//
// The volatile reads are critical for host-mapped pinned
// visibility — the worker writes done_slot from the CPU side and
// the GPU read must not be cached.
__global__ void cots_wait_done_kernel(volatile uint32_t* req_slot,
                                      volatile uint32_t* done_slot) {
  uint32_t expected = *req_slot;
  uint32_t done = *done_slot;
  while (done < expected) {
    asm volatile("nanosleep.u32 100;" ::: "memory");
    done = *done_slot;
  }
}

// Diag wait kernel. Identical control flow plus three counter
// increments. `immediate_resume_count` increments when the GPU
// window covered the CPU work (done >= req on first read);
// `lagging_wait_count` increments when the kernel had to spin
// at all; `spin_iters_total` accumulates the spin-loop iteration
// count for the lagging waits. Together they tell us how often
// the wait actually serializes (the §1c.29 design's canary).
//
// Atomics used to be safe under any future multi-block / multi-
// stream variant. With the current single-thread-block launch,
// regular adds would suffice — keeping atomic to make the
// counters correct under any later expansion.
__global__ void cots_wait_done_kernel_diag(volatile uint32_t* req_slot,
                                           volatile uint32_t* done_slot,
                                           int64_t* spin_iters_total,
                                           int64_t* immediate_count,
                                           int64_t* lagging_count) {
  uint32_t expected = *req_slot;
  uint32_t done = *done_slot;
  if (done >= expected) {
    atomicAdd(reinterpret_cast<unsigned long long*>(immediate_count), 1ull);
    return;
  }
  atomicAdd(reinterpret_cast<unsigned long long*>(lagging_count), 1ull);
  int64_t iters = 0;
  do {
    asm volatile("nanosleep.u32 100;" ::: "memory");
    done = *done_slot;
    ++iters;
  } while (done < expected);
  atomicAdd(reinterpret_cast<unsigned long long*>(spin_iters_total),
            static_cast<unsigned long long>(iters));
}

__device__ __forceinline__ int64_t
wait_for_done_slot(volatile uint32_t* req_slot, volatile uint32_t* done_slot,
                   bool record_diag, int64_t* spin_iters_total,
                   int64_t* immediate_count, int64_t* lagging_count) {
  uint32_t expected = *req_slot;
  uint32_t done = *done_slot;
  if (done >= expected) {
    if (record_diag) {
      atomicAdd(reinterpret_cast<unsigned long long*>(immediate_count), 1ull);
    }
    return 0;
  }
  if (record_diag) {
    atomicAdd(reinterpret_cast<unsigned long long*>(lagging_count), 1ull);
  }
  int64_t iters = 0;
  do {
    asm volatile("nanosleep.u32 100;" ::: "memory");
    done = *done_slot;
    ++iters;
  } while (done < expected);
  if (record_diag) {
    atomicAdd(reinterpret_cast<unsigned long long*>(spin_iters_total),
              static_cast<unsigned long long>(iters));
  }
  return iters;
}

__global__ void cots_wait_uva_copy_kernel(volatile uint32_t* req_slot,
                                          volatile uint32_t* done_slot,
                                          const uint16_t* src, uint16_t* dst,
                                          int64_t n_elements) {
  if (threadIdx.x == 0) {
    wait_for_done_slot(req_slot, done_slot, false, nullptr, nullptr, nullptr);
  }
  __syncthreads();

  int64_t offset = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t i = offset; i < n_elements; i += stride) {
    dst[i] = src[i];
  }
}

__global__ void cots_wait_uva_copy_kernel_diag(
    volatile uint32_t* req_slot, volatile uint32_t* done_slot,
    const uint16_t* src, uint16_t* dst, int64_t n_elements,
    int64_t* spin_iters_total, int64_t* immediate_count,
    int64_t* lagging_count) {
  if (threadIdx.x == 0) {
    wait_for_done_slot(req_slot, done_slot, blockIdx.x == 0, spin_iters_total,
                       immediate_count, lagging_count);
  }
  __syncthreads();

  int64_t offset = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t i = offset; i < n_elements; i += stride) {
    dst[i] = src[i];
  }
}

}  // namespace

// C-linkage launcher entry points — called from cots_cpu_infer.cpp
// (compiled as C++ host code under the same target). Hide the
// `<<<>>>` syntax + kernel symbols behind plain functions so the
// .cpp file doesn't need to be CUDA.

extern "C" void launch_cots_wait_done_kernel_production(uint32_t* dev_req_slot,
                                                        uint32_t* dev_done_slot,
                                                        cudaStream_t stream) {
  cots_wait_done_kernel<<<1, 1, 0, stream>>>(dev_req_slot, dev_done_slot);
}

extern "C" void launch_cots_wait_done_kernel_diag(
    uint32_t* dev_req_slot, uint32_t* dev_done_slot, int64_t* spin_iters_total,
    int64_t* immediate_count, int64_t* lagging_count, cudaStream_t stream) {
  cots_wait_done_kernel_diag<<<1, 1, 0, stream>>>(
      dev_req_slot, dev_done_slot, spin_iters_total, immediate_count,
      lagging_count);
}

extern "C" void launch_cots_wait_uva_copy_kernel_production(
    uint32_t* dev_req_slot, uint32_t* dev_done_slot, const void* src_host_uva,
    void* dst_device, int64_t n_elements, cudaStream_t stream) {
  if (n_elements <= 0) {
    return;
  }
  constexpr int kBlock = 256;
  int64_t blocks64 = (n_elements + kBlock - 1) / kBlock;
  int blocks = static_cast<int>(blocks64 > 65535 ? 65535 : blocks64);
  cots_wait_uva_copy_kernel<<<blocks, kBlock, 0, stream>>>(
      dev_req_slot, dev_done_slot, static_cast<const uint16_t*>(src_host_uva),
      static_cast<uint16_t*>(dst_device), n_elements);
}

extern "C" void launch_cots_wait_uva_copy_kernel_diag(
    uint32_t* dev_req_slot, uint32_t* dev_done_slot, const void* src_host_uva,
    void* dst_device, int64_t n_elements, int64_t* spin_iters_total,
    int64_t* immediate_count, int64_t* lagging_count, cudaStream_t stream) {
  if (n_elements <= 0) {
    return;
  }
  constexpr int kBlock = 256;
  int64_t blocks64 = (n_elements + kBlock - 1) / kBlock;
  int blocks = static_cast<int>(blocks64 > 65535 ? 65535 : blocks64);
  cots_wait_uva_copy_kernel_diag<<<blocks, kBlock, 0, stream>>>(
      dev_req_slot, dev_done_slot, static_cast<const uint16_t*>(src_host_uva),
      static_cast<uint16_t*>(dst_device), n_elements, spin_iters_total,
      immediate_count, lagging_count);
}

}  // namespace cots
}  // namespace vllm
