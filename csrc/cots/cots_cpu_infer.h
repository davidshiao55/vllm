// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Phase 1c native CPU runner for COTS. Adapted from KTransformers
// `kt-kernel/cpu_backend/cpuinfer.h` with WorkerPool removed (oneDNN/ATen
// own intra-op threading) and a slab-based task dispatch added.
//
// See David/Docs/implementation_roadmap.md Phase 1c, the approved plan
// at /root/.claude/plans/pleaes-implement-phase1c-in-quizzical-mist.md,
// and David/Docs/phase1a_findings.md §1.14 for the substrate motivation.

#ifndef VLLM_COTS_CPU_INFER_H_
#define VLLM_COTS_CPU_INFER_H_

#include <ATen/ATen.h>
#include <c10/core/ScalarType.h>
#include <cuda_runtime_api.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "task_queue.h"

namespace vllm {
namespace cots {

// Per-(layer, bucket, op_kind) task slab. Address-stable for the lifetime
// of CotsCpuInfer (slabs_ is reserve()'d once at install time and never
// resized) so that captured CUDA graphs can record &slabs_[id] as their
// host-callback userData.
//
// Layout / population rules:
//   * Every w_*_ptr is a POST-narrow data_ptr() — never base + manual offset.
//     `at::from_blob` has no storage-offset parameter, so the offset must be
//     baked into the pointer.
//   * For column-narrowed views (down-proj), w_down_stride_row /
//     w_down_stride_col mirror the source tensor's stride() so the C++ side
//     can reconstruct the strided view via `at::from_blob(ptr, sizes,
//     strides, opts)`.
//   * num_tokens is the only field written per-call (by the submit-callback
//     before TaskQueue::enqueue). All other fields are constant after
//     populate_slab() and remain valid across CUDA graph re-replays.
struct alignas(64) TaskSlab {
  enum OpKind : int32_t { kQkv = 0, kMlpBlock = 1, kDryrunNoop = 2 };

  // Self-pointer pattern preserved from kt-kernel (the host callback writes
  // it before calling into the dispatcher, which lets the static dispatcher
  // recover the owning CotsCpuInfer*). For our reserve-once design the
  // pointer is in fact set once at install time and never rewritten, but we
  // keep the field at the standard cpuinfer offset for parity with debug
  // tooling.
  void* self = nullptr;

  int32_t op_kind = kDryrunNoop;
  int32_t n_threads = 1;

  // Only field updated per submit. atomic<int32_t> with relaxed load is OK:
  // the submit-callback's host-function dispatch happens-before TaskQueue
  // worker dequeue (CUDA stream ordering + cv_ notify in TaskQueue::enqueue).
  std::atomic<int32_t> num_tokens{0};

  void* x_pinned_ptr = nullptr;
  int32_t in_dim = 0;
  void* y_pinned_ptr = nullptr;
  int32_t cpu_out_dim = 0;
  // dtype is hard-coded BFloat16 by the .cpp view builders (the
  // `CotsOffloadConfig.cpu_dtype` literal locks this for Phase 1c).

  // QKV (op_kind == kQkv): contiguous row-major weight slice.
  void* w_cpu_ptr = nullptr;
  int32_t w_cpu_rows = 0;

  // MLP block (op_kind == kMlpBlock).
  void* w_gate_ptr = nullptr;
  int32_t w_gate_rows = 0;
  void* w_up_ptr = nullptr;
  int32_t w_up_rows = 0;
  // Down-proj is a column-narrow on (out_dim, n_cpu) row-major storage.
  // w_down_ptr is the post-narrow data_ptr() (already offset by
  // dn_n_pref * elem_size); strides mirror the source tensor.
  void* w_down_ptr = nullptr;
  int32_t w_down_rows = 0;        // = out_dim
  int32_t w_down_cols = 0;        // = dn_n_cpu
  int64_t w_down_stride_row = 0;  // = original n_cpu (in elements)
  int64_t w_down_stride_col = 1;
  int32_t intermediate_per_half = 0;  // for silu*up shape
};

// Static sync-callback userData — owned as a stable member of CotsCpuInfer
// so its address is valid across CUDA graph replays.
struct SyncArgs {
  void* infer = nullptr;
  size_t allow_n_pending = 0;
};

class CotsCpuInfer {
 public:
  CotsCpuInfer();
  ~CotsCpuInfer();

  CotsCpuInfer(const CotsCpuInfer&) = delete;
  CotsCpuInfer& operator=(const CotsCpuInfer&) = delete;

  // Reserves N slabs, sized once. Allocates the worker-local MLP scratch
  // (`scratch_silu_up_`) at the worst-case shape across all MLP slabs.
  // After install(), slabs_.size() == n_slabs is invariant for the
  // lifetime of this CotsCpuInfer. Subsequent populate_slab calls only
  // mutate the pre-existing slab entries.
  void install(int64_t n_slabs, int64_t scratch_max_tokens,
               int64_t scratch_max_intermediate_per_half);

  // Populate a previously-reserved slab. All pointers must be POST-narrow
  // data_ptr()s (see TaskSlab doc above). Idempotent. dtype is hard-coded
  // to bfloat16 — `CotsOffloadConfig.cpu_dtype` is `Literal["bfloat16"]`
  // (per phase0 §0.3.2 oneDNN BF16 is the only fast CPU GEMM path on
  // AVX2 hardware), so passing `torch.dtype` over pybind would just be
  // a brittle int-enum dance. If we ever support fp16/fp32, take dtype
  // back as a parameter at that point.
  void populate_slab_qkv(int64_t task_id, int32_t n_threads,
                         uintptr_t x_pinned_ptr, int32_t in_dim,
                         uintptr_t y_pinned_ptr, int32_t cpu_out_dim,
                         uintptr_t w_cpu_ptr, int32_t w_cpu_rows);

  void populate_slab_mlp(int64_t task_id, int32_t n_threads,
                         uintptr_t x_pinned_ptr, int32_t in_dim,
                         uintptr_t y_pinned_ptr, int32_t cpu_out_dim,
                         uintptr_t w_gate_ptr, int32_t w_gate_rows,
                         uintptr_t w_up_ptr, int32_t w_up_rows,
                         uintptr_t w_down_ptr, int32_t w_down_rows,
                         int32_t w_down_cols, int64_t w_down_stride_row,
                         int64_t w_down_stride_col,
                         int32_t intermediate_per_half);

  void populate_slab_dryrun(int64_t task_id);

  // Submit a task on the *current* CUDA stream. Writes num_tokens into the
  // slab and queues a cudaLaunchHostFunc node onto the supplied stream;
  // when that host callback fires (after prior stream work completes), it
  // enqueues the actual task body onto TaskQueue.
  //
  // The Python custom-op `vllm.cots_submit_gemm` (registered in
  // vllm/model_executor/offloader/cots_ops.py) translates to a call here
  // with `stream = c10::cuda::getCurrentCUDAStream()`.
  void submit_on_stream(int64_t task_id, int32_t num_tokens,
                        uintptr_t cuda_stream);

  // Likewise for sync. Schedules `&sync_args_` (a stable member) as a host
  // callback on the supplied stream; the callback blocks the CUDA driver
  // thread on `task_queue_->sync(0)`.
  void sync_on_stream(uintptr_t cuda_stream);

  // Test-only / Python-side helpers (not in the captured-graph hot path).
  // Submit N dryrun_noop tasks directly via TaskQueue::enqueue (no CUDA
  // stream / host callback involved). Used by test_taskqueue_stress.
  void submit_dryrun_burst(int64_t n);
  // Block calling thread until TaskQueue drains. Same as sync_on_stream but
  // without the cudaLaunchHostFunc indirection.
  void sync_blocking();

  // Bucket-aware thread policy hook. Sets the worker thread's CPU affinity
  // to `cpu_set` (a bitmask packed into int64). cpu_set == 0 means clear
  // affinity. Intersect with sched_getaffinity at call time to avoid
  // EINVAL when running under a restrictive cgroup.
  void set_worker_affinity(int64_t cpu_set);

  // Test helper: read whether the worker has set num_threads to the value
  // the slab requested, after a sync. Used by test_bucket_thread_policy.
  int32_t last_observed_num_threads() const {
    return last_observed_num_threads_.load(std::memory_order_acquire);
  }

  // Worker exception surfacing. Each Python-side `submit*` / `sync*` call
  // checks has_error_ and re-raises last_error_msg_ as a Python
  // RuntimeError. This mirrors the Python runner's future.result()
  // re-raise semantics.
  //
  // The guard is invoked at the START of every entry point (submit*, sync*,
  // submit_dryrun_burst, populate_slab*) so a previously-failed task
  // surfaces as a Python RuntimeError on the next call rather than getting
  // silently swallowed. The first call after a worker error consumes the
  // error (clears has_error_); subsequent calls succeed as normal.
  bool has_error() const { return has_error_.load(std::memory_order_acquire); }
  std::string take_error();
  // Throws std::runtime_error (mapped by pybind11 to RuntimeError) if a
  // worker task failed since the last error consumption. Public so a
  // direct C++ caller (e.g., a future C++ binding) can opt in.
  void check_error();

  // Stage-1 microbench helper: run `at::linear(x, w)` on the calling
  // thread (NOT through TaskQueue) and write into `y`. Used by
  // test_at_linear_microbench to compare C++ at::linear vs Python F.linear
  // perf BEFORE wiring through host callbacks. Argument tensors are
  // user-managed; this just calls into ATen.
  void run_at_linear_inline(at::Tensor x, at::Tensor w, at::Tensor y_out);

 private:
  // Static dispatchers used by cudaLaunchHostFunc. Both must be
  // `void(*)(void*)`.
  static void DispatchCallback(void* user_data);
  static void SyncCallback(void* user_data);

  // Worker-thread task body; runs whatever op_kind says.
  void RunSlabOnWorker(TaskSlab* slab);

  std::unique_ptr<TaskQueue> task_queue_;

  // Heap-allocated, sized-once raw array. Address-stable for the lifetime of
  // *this — captured CUDA graphs record &slabs_[id] as host-callback userData
  // and re-replay must see the same address. We deliberately do NOT use
  // std::vector<TaskSlab> because std::atomic<int32_t> num_tokens makes
  // TaskSlab non-MoveConstructible, which std::vector's template machinery
  // (eagerly instantiated even for reserve-only flows) rejects.
  std::unique_ptr<TaskSlab[]> slabs_;
  int64_t slab_count_ = 0;

  // Worker-local scratch for MLP intermediates (silu(gate) * up). One
  // max-sized tensor, sized at install. NOT one-per-slab.
  at::Tensor scratch_silu_up_;
  int64_t scratch_max_tokens_ = 0;
  int64_t scratch_max_intermediate_per_half_ = 0;

  // Stable userData for sync_on_stream's cudaLaunchHostFunc — must be
  // a member, NOT a stack/heap-per-call alloc. CUDA graph capture freezes
  // the userData pointer at capture time.
  SyncArgs sync_args_{};

  // Cached at::set_num_threads value to avoid redundant rebuilds of the
  // at-thread-pool on every task. Accessed only on the worker thread.
  int32_t worker_current_n_threads_ = 0;

  std::atomic<int32_t> last_observed_num_threads_{0};

  // Worker exception surfacing (see has_error / take_error above).
  std::atomic<bool> has_error_{false};
  std::mutex error_mtx_;
  std::string last_error_msg_;
};

}  // namespace cots
}  // namespace vllm

#endif  // VLLM_COTS_CPU_INFER_H_
