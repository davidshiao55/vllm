// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Phase 1c — native CPU runner implementation.

#include "cots_cpu_infer.h"

#include <ATen/ops/linear.h>
#include <ATen/ops/silu.h>
#include <c10/core/InferenceMode.h>
#include <pthread.h>
#include <sched.h>
#include <torch/torch.h>

#include <cstring>
#include <exception>
#include <stdexcept>
#include <utility>

namespace vllm {
namespace cots {

namespace {

constexpr int kMaxCpus = 64;

// Phase 1c is bfloat16-locked (see header comment + cpu_dtype literal in
// `vllm/config/offload.py`). Centralizing the dtype here keeps the slab
// struct lean and avoids any int-enum dance over pybind.
constexpr auto kCpuDtype = at::kBFloat16;

// Build an ATen tensor view over a pinned-CPU pointer with a contiguous
// row-major layout. Shape is {rows, cols}.
at::Tensor ContigCpuViewFromBlob(void* ptr, int64_t rows, int64_t cols) {
  auto opts = at::TensorOptions().dtype(kCpuDtype).device(at::kCPU);
  return at::from_blob(ptr, {rows, cols}, opts);
}

// Build an ATen tensor view over a column-narrowed pinned-CPU pointer.
// `ptr` is the post-narrow data pointer (offset already baked in).
// strides {row_stride, col_stride} reflect the source tensor.
at::Tensor StridedCpuViewFromBlob(void* ptr, int64_t rows, int64_t cols,
                                  int64_t row_stride, int64_t col_stride) {
  auto opts = at::TensorOptions().dtype(kCpuDtype).device(at::kCPU);
  return at::from_blob(ptr, {rows, cols}, {row_stride, col_stride}, opts);
}

}  // namespace

CotsCpuInfer::CotsCpuInfer() : task_queue_(std::make_unique<TaskQueue>()) {
  sync_args_.infer = static_cast<void*>(this);
  sync_args_.allow_n_pending = 0;
}

CotsCpuInfer::~CotsCpuInfer() {
  // Drain before destructing the queue so any in-flight task completes.
  if (task_queue_) {
    task_queue_->sync(0);
  }
}

void CotsCpuInfer::install(int64_t n_slabs, int64_t scratch_max_tokens,
                           int64_t scratch_max_intermediate_per_half) {
  TORCH_CHECK(n_slabs >= 0, "install: n_slabs must be >= 0, got ", n_slabs);
  TORCH_CHECK(!slabs_,
              "install: CotsCpuInfer already installed; call once per "
              "CotsCpuInfer instance");

  // Sized-once heap array. Address-stable for the lifetime of *this so
  // captured CUDA graphs can record &slabs_[id] as host-callback userData.
  if (n_slabs > 0) {
    slabs_ = std::unique_ptr<TaskSlab[]>(new TaskSlab[n_slabs]);
    for (int64_t i = 0; i < n_slabs; ++i) {
      slabs_[i].self = static_cast<void*>(this);
      slabs_[i].op_kind = TaskSlab::kDryrunNoop;
    }
  }
  slab_count_ = n_slabs;

  scratch_max_tokens_ = scratch_max_tokens;
  scratch_max_intermediate_per_half_ = scratch_max_intermediate_per_half;
  if (scratch_max_tokens > 0 && scratch_max_intermediate_per_half > 0) {
    auto opts = at::TensorOptions().dtype(at::kBFloat16).device(at::kCPU);
    // Worker-local scratch for silu(gate)*up. One max-sized tensor.
    scratch_silu_up_ = at::empty(
        {scratch_max_tokens, scratch_max_intermediate_per_half}, opts);
  }
}

void CotsCpuInfer::check_error() {
  if (!has_error_.load(std::memory_order_acquire)) return;
  std::lock_guard<std::mutex> lock(error_mtx_);
  std::string msg = std::move(last_error_msg_);
  last_error_msg_.clear();
  has_error_.store(false, std::memory_order_release);
  // Throwing std::runtime_error: pybind11 maps it to Python RuntimeError
  // automatically. Mirrors the Python runner's `future.result()` re-raise.
  throw std::runtime_error(msg);
}

void CotsCpuInfer::populate_slab_qkv(int64_t task_id, int32_t n_threads,
                                     uintptr_t x_pinned_ptr, int32_t in_dim,
                                     uintptr_t y_pinned_ptr,
                                     int32_t cpu_out_dim, uintptr_t w_cpu_ptr,
                                     int32_t w_cpu_rows) {
  check_error();
  TORCH_CHECK(task_id >= 0 && task_id < slab_count_,
              "populate_slab_qkv: task_id ", task_id, " out of range");
  TaskSlab& s = slabs_[task_id];
  s.op_kind = TaskSlab::kQkv;
  s.n_threads = n_threads;
  s.x_pinned_ptr = reinterpret_cast<void*>(x_pinned_ptr);
  s.in_dim = in_dim;
  s.y_pinned_ptr = reinterpret_cast<void*>(y_pinned_ptr);
  s.cpu_out_dim = cpu_out_dim;
  s.w_cpu_ptr = reinterpret_cast<void*>(w_cpu_ptr);
  s.w_cpu_rows = w_cpu_rows;
}

void CotsCpuInfer::populate_slab_mlp(
    int64_t task_id, int32_t n_threads, uintptr_t x_pinned_ptr, int32_t in_dim,
    uintptr_t y_pinned_ptr, int32_t cpu_out_dim, uintptr_t w_gate_ptr,
    int32_t w_gate_rows, uintptr_t w_up_ptr, int32_t w_up_rows,
    uintptr_t w_down_ptr, int32_t w_down_rows, int32_t w_down_cols,
    int64_t w_down_stride_row, int64_t w_down_stride_col,
    int32_t intermediate_per_half) {
  check_error();
  TORCH_CHECK(task_id >= 0 && task_id < slab_count_,
              "populate_slab_mlp: task_id ", task_id, " out of range");
  TaskSlab& s = slabs_[task_id];
  s.op_kind = TaskSlab::kMlpBlock;
  s.n_threads = n_threads;
  s.x_pinned_ptr = reinterpret_cast<void*>(x_pinned_ptr);
  s.in_dim = in_dim;
  s.y_pinned_ptr = reinterpret_cast<void*>(y_pinned_ptr);
  s.cpu_out_dim = cpu_out_dim;
  s.w_gate_ptr = reinterpret_cast<void*>(w_gate_ptr);
  s.w_gate_rows = w_gate_rows;
  s.w_up_ptr = reinterpret_cast<void*>(w_up_ptr);
  s.w_up_rows = w_up_rows;
  s.w_down_ptr = reinterpret_cast<void*>(w_down_ptr);
  s.w_down_rows = w_down_rows;
  s.w_down_cols = w_down_cols;
  s.w_down_stride_row = w_down_stride_row;
  s.w_down_stride_col = w_down_stride_col;
  s.intermediate_per_half = intermediate_per_half;
}

void CotsCpuInfer::populate_slab_dryrun(int64_t task_id, uintptr_t x_pinned_ptr,
                                        int32_t in_dim, uintptr_t y_pinned_ptr,
                                        int32_t cpu_out_dim) {
  check_error();
  TORCH_CHECK(task_id >= 0 && task_id < slab_count_,
              "populate_slab_dryrun: task_id ", task_id, " out of range");
  TaskSlab& s = slabs_[task_id];
  s.op_kind = TaskSlab::kDryrunNoop;
  // §1c.20: dryrun must carry the x_pinned + y_pinned pointers so
  // submit_on_stream's D2H and sync's y_pinned_view both resolve.
  s.x_pinned_ptr = reinterpret_cast<void*>(x_pinned_ptr);
  s.in_dim = in_dim;
  s.y_pinned_ptr = reinterpret_cast<void*>(y_pinned_ptr);
  s.cpu_out_dim = cpu_out_dim;
}

void CotsCpuInfer::submit_on_stream(int64_t task_id, int32_t num_tokens,
                                    uintptr_t x_gpu_ptr, int64_t x_cols,
                                    int64_t x_stride0, int64_t x_stride1,
                                    uintptr_t cuda_stream) {
  // Surface any prior worker error BEFORE queueing more work.
  check_error();
  TORCH_CHECK(task_id >= 0 && task_id < slab_count_,
              "submit_on_stream: task_id ", task_id, " out of range");
  TaskSlab* slab = &slabs_[task_id];
  TORCH_CHECK(num_tokens >= 0, "submit_on_stream: num_tokens=", num_tokens,
              " < 0");
  // §1c.20: bound check against scratch_max_tokens_ catches Planner
  // mistakes where an oversized batch reaches a slab. Skipped when
  // scratch_max_tokens_==0 (test fixtures that don't allocate scratch
  // / don't run real tokens through the slab — diagnostic-only path).
  TORCH_CHECK(scratch_max_tokens_ == 0 || num_tokens <= scratch_max_tokens_,
              "submit_on_stream: num_tokens=", num_tokens,
              " exceeds scratch_max_tokens=", scratch_max_tokens_,
              " (would write past the pinned buffer's tail)");
  slab->num_tokens.store(num_tokens, std::memory_order_release);
  // §1c.21 counters: bump submit_count + num_tokens histogram for
  // this op kind. Bin index = position of msb of num_tokens, clamped
  // to [0, 7]. nt<=1 → 0, nt<=2 → 1, ..., nt<=64 → 6, nt>64 → 7.
  {
    int hist_bin;
    if (num_tokens <= 1)
      hist_bin = 0;
    else if (num_tokens <= 2)
      hist_bin = 1;
    else if (num_tokens <= 4)
      hist_bin = 2;
    else if (num_tokens <= 8)
      hist_bin = 3;
    else if (num_tokens <= 16)
      hist_bin = 4;
    else if (num_tokens <= 32)
      hist_bin = 5;
    else if (num_tokens <= 64)
      hist_bin = 6;
    else
      hist_bin = 7;
    switch (slab->op_kind) {
      case TaskSlab::kQkv:
        submit_count_qkv_.fetch_add(1, std::memory_order_relaxed);
        nt_hist_qkv_[hist_bin].fetch_add(1, std::memory_order_relaxed);
        break;
      case TaskSlab::kMlpBlock:
        submit_count_mlp_.fetch_add(1, std::memory_order_relaxed);
        nt_hist_mlp_[hist_bin].fetch_add(1, std::memory_order_relaxed);
        break;
      case TaskSlab::kDryrunNoop:
      default:
        submit_count_dryrun_.fetch_add(1, std::memory_order_relaxed);
        nt_hist_dryrun_[hist_bin].fetch_add(1, std::memory_order_relaxed);
        break;
    }
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  // §1c.20: D2H from x_gpu to slab's pinned input buffer.
  // Stride-aware: contiguous → 1D cudaMemcpyAsync, row-strided →
  // 2D cudaMemcpy2DAsync. Both graph-capturable. Skipped when
  // `x_gpu_ptr == 0` (test fixtures that exercise dispatch only).
  if (x_gpu_ptr != 0) {
    TORCH_CHECK(slab->x_pinned_ptr != nullptr,
                "submit_on_stream: slab.x_pinned_ptr is null at task_id=",
                task_id, " (slab not populated?)");
    TORCH_CHECK(slab->in_dim > 0, "submit_on_stream: slab.in_dim is ",
                slab->in_dim, " at task_id=", task_id);
    TORCH_CHECK(
        x_cols == slab->in_dim, "submit_on_stream: x_gpu.shape[1]=", x_cols,
        " disagrees with slab.in_dim=", slab->in_dim, " at task_id=", task_id);
    TORCH_CHECK(x_stride1 == 1, "submit_on_stream: x_gpu.stride(1)=", x_stride1,
                " (must be 1 — no transposed-stride layouts; only "
                "row-strided contiguous-along-feature inputs are "
                "supported)");
    TORCH_CHECK(x_stride0 >= x_cols,
                "submit_on_stream: x_gpu.stride(0)=", x_stride0,
                " < x_cols=", x_cols, " (rows would overlap; reject)");
    const size_t width_bytes =
        static_cast<size_t>(slab->in_dim) * sizeof(at::BFloat16);
    cudaError_t copy_err;
    if (x_stride0 == x_cols) {
      // Contiguous row layout — single 1D copy is fastest.
      const size_t bytes = static_cast<size_t>(num_tokens) * width_bytes;
      d2h_1d_count_.fetch_add(1, std::memory_order_relaxed);
      d2h_1d_bytes_.fetch_add(static_cast<int64_t>(bytes),
                              std::memory_order_relaxed);
      copy_err = cudaMemcpyAsync(slab->x_pinned_ptr,
                                 reinterpret_cast<void*>(x_gpu_ptr), bytes,
                                 cudaMemcpyDeviceToHost, stream);
    } else {
      // Row-strided (e.g., a `[:, :hidden_dim]` slice over a wider
      // base, or a hidden-state view that the model produced with
      // padding). 2D copy walks rows at the source stride and writes
      // tightly-packed rows at the destination.
      const size_t src_pitch =
          static_cast<size_t>(x_stride0) * sizeof(at::BFloat16);
      const size_t bytes_2d = width_bytes * static_cast<size_t>(num_tokens);
      d2h_2d_count_.fetch_add(1, std::memory_order_relaxed);
      d2h_2d_bytes_.fetch_add(static_cast<int64_t>(bytes_2d),
                              std::memory_order_relaxed);
      copy_err = cudaMemcpy2DAsync(
          slab->x_pinned_ptr, /*dpitch=*/width_bytes,
          reinterpret_cast<void*>(x_gpu_ptr), /*spitch=*/src_pitch,
          /*width=*/width_bytes, /*height=*/static_cast<size_t>(num_tokens),
          cudaMemcpyDeviceToHost, stream);
    }
    TORCH_CHECK(copy_err == cudaSuccess, "submit_on_stream: D2H copy failed (",
                (x_stride0 == x_cols ? "1D" : "2D"),
                "): ", cudaGetErrorString(copy_err));
  }
  cudaError_t err = cudaLaunchHostFunc(stream, &CotsCpuInfer::DispatchCallback,
                                       static_cast<void*>(slab));
  TORCH_CHECK(
      err == cudaSuccess,
      "cudaLaunchHostFunc(DispatchCallback) failed: ", cudaGetErrorString(err));
}

void CotsCpuInfer::sync_on_stream(uintptr_t cuda_stream) {
  check_error();
  // sync_args_ is a stable member of *this; safe to take its address as
  // userData for cudaLaunchHostFunc, including across CUDA graph replays.
  cudaError_t err = cudaLaunchHostFunc(
      reinterpret_cast<cudaStream_t>(cuda_stream), &CotsCpuInfer::SyncCallback,
      static_cast<void*>(&sync_args_));
  TORCH_CHECK(err == cudaSuccess, "cudaLaunchHostFunc(SyncCallback) failed: ",
              cudaGetErrorString(err));
}

void CotsCpuInfer::submit_dryrun_burst(int64_t n) {
  check_error();
  for (int64_t i = 0; i < n; ++i) {
    task_queue_->enqueue([] {
      // Pure no-op: no slab read, no scratch use. Used by
      // test_taskqueue_stress to validate FIFO + drain semantics.
    });
  }
}

void CotsCpuInfer::sync_blocking() {
  task_queue_->sync(0);
  // Surface any worker error that fired while we were waiting.
  check_error();
}

void CotsCpuInfer::set_worker_affinity(uint64_t cpu_set) {
  if (cpu_set == 0) {
    return;  // No-op; leave kernel default. (Stage 4 fills this out.)
  }
  // Intersect with the process's existing affinity to avoid EINVAL under
  // a restrictive cgroup mask. (Phase 1c plan §risk-3 / Stage 4.)
  cpu_set_t requested;
  CPU_ZERO(&requested);
  // Iterate up to min(kMaxCpus, 64) — uint64 mask has 64 bits, no more.
  // uint64_t shift up to bit 63 is well-defined (unlike signed int64).
  for (int i = 0; i < kMaxCpus && i < 64; ++i) {
    if (cpu_set & (uint64_t{1} << i)) CPU_SET(i, &requested);
  }
  cpu_set_t process_mask;
  CPU_ZERO(&process_mask);
  if (sched_getaffinity(0, sizeof(process_mask), &process_mask) != 0) {
    return;
  }
  cpu_set_t effective;
  CPU_ZERO(&effective);
  for (int i = 0; i < kMaxCpus; ++i) {
    if (CPU_ISSET(i, &requested) && CPU_ISSET(i, &process_mask)) {
      CPU_SET(i, &effective);
    }
  }
  if (CPU_COUNT(&effective) == 0) {
    return;  // empty intersection; warn-and-skip per plan §Stage 4.
  }
  // Run the actual pthread_setaffinity_np on the worker thread itself by
  // submitting a tiny task. Stage 4 expands this; Stage 1 stores a pending
  // mask only via this side channel.
  task_queue_->enqueue([effective] {
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &effective);
  });
}

std::string CotsCpuInfer::take_error() {
  std::lock_guard<std::mutex> lock(error_mtx_);
  std::string msg = std::move(last_error_msg_);
  last_error_msg_.clear();
  has_error_.store(false, std::memory_order_release);
  return msg;
}

void CotsCpuInfer::run_at_linear_inline(at::Tensor x, at::Tensor w,
                                        at::Tensor y_out) {
  // Stage 1 microbench helper: directly drive `at::linear` from C++.
  // No TaskQueue, no host callback. Lets the test compare wall-clock
  // against Python `F.linear` on bit-identical tensors and catch the
  // catastrophic-scalar-fallback path documented in
  // `David/Docs/phase0_findings.md §0.3.2`.
  c10::InferenceMode g;
  // c10::AutoDispatchBelowAutograd is implied by InferenceMode.
  y_out.copy_(at::linear(x, w));
}

at::Tensor CotsCpuInfer::y_pinned_view(int64_t task_id,
                                       int32_t num_tokens) const {
  // §1c.20: build an `at::from_blob` CPU tensor view over the slab's
  // pinned output buffer. Used by `cots_sync_then_uva`'s impl on the
  // captured-graph hot path so the CPU output tensor is NEVER a
  // custom-op argument visible to Inductor.
  if (task_id < 0 || task_id >= slab_count_) {
    throw std::out_of_range(
        "CotsCpuInfer::y_pinned_view: task_id=" + std::to_string(task_id) +
        " out of range [0, " + std::to_string(slab_count_) + ")");
  }
  const TaskSlab& slab = slabs_[task_id];
  if (slab.y_pinned_ptr == nullptr) {
    throw std::runtime_error(
        "CotsCpuInfer::y_pinned_view: slab.y_pinned_ptr is null at task_id=" +
        std::to_string(task_id) + " (slab not populated?)");
  }
  if (num_tokens < 0) {
    throw std::invalid_argument("CotsCpuInfer::y_pinned_view: num_tokens=" +
                                std::to_string(num_tokens) + " < 0");
  }
  if (scratch_max_tokens_ != 0 && num_tokens > scratch_max_tokens_) {
    throw std::invalid_argument(
        "CotsCpuInfer::y_pinned_view: num_tokens=" +
        std::to_string(num_tokens) +
        " exceeds scratch_max_tokens=" + std::to_string(scratch_max_tokens_) +
        " (would read past the pinned buffer's tail)");
  }
  // dtype is hard-coded bfloat16 (matches populate_slab_*'s contract).
  // The pointer's underlying allocation came from `_y_pinned` (a
  // `torch.empty(..., pin_memory=True)`) and is page-locked; we trust
  // the install-time invariant rather than re-validating here.
  auto options = at::TensorOptions().dtype(at::kBFloat16).device(at::kCPU);
  auto sizes = std::array<int64_t, 2>{static_cast<int64_t>(num_tokens),
                                      static_cast<int64_t>(slab.cpu_out_dim)};
  return at::from_blob(slab.y_pinned_ptr, sizes, options);
}

// --- static cudaLaunchHostFunc callbacks -----------------------------------

// Submit-side host callback. Runs on the CUDA driver thread; must NOT
// block (CUDA stream is paused while we run). Just enqueues to the
// TaskQueue worker.
void CotsCpuInfer::DispatchCallback(void* user_data) {
  TaskSlab* slab = static_cast<TaskSlab*>(user_data);
  CotsCpuInfer* self = static_cast<CotsCpuInfer*>(slab->self);
  // Copy the pointer into the lambda capture; this is safe because
  // `slab` is a member of `self->slabs_` which is reserve-once.
  self->task_queue_->enqueue([self, slab] { self->RunSlabOnWorker(slab); });
}

// Sync-side host callback. Blocks the CUDA driver thread until the
// TaskQueue drains. The user_data is `&self->sync_args_` (stable member).
void CotsCpuInfer::SyncCallback(void* user_data) {
  SyncArgs* args = static_cast<SyncArgs*>(user_data);
  CotsCpuInfer* self = static_cast<CotsCpuInfer*>(args->infer);
  self->task_queue_->sync(args->allow_n_pending);
}

// --- worker-thread task body ----------------------------------------------

void CotsCpuInfer::RunSlabOnWorker(TaskSlab* slab) {
  // Worker exception policy: every body wrapped in try/catch so that
  // pending-decrement / cv_ notify in TaskQueue::Worker still happens
  // (we return normally from this function), and the next Python-side
  // submit/sync call surfaces the error as a Python RuntimeError.
  try {
    c10::InferenceMode g;

    // Bucket-aware thread policy: only call set_num_threads when it would
    // change. Stage 4 populates slab->n_threads from the per-bucket map;
    // Stage 1 leaves n_threads = 1, which means we call set_num_threads(1)
    // on the very first task and then stay at 1. (No regression vs the
    // Phase 1a Python runner default, which used scalar `cpu_num_threads`
    // via `torch.set_num_threads` once at offloader init.)
    if (slab->n_threads > 0 && slab->n_threads != worker_current_n_threads_) {
      at::set_num_threads(slab->n_threads);
      worker_current_n_threads_ = slab->n_threads;
    }
    last_observed_num_threads_.store(at::get_num_threads(),
                                     std::memory_order_release);

    // §1c.21: prefer runtime_num_tokens_ (live unpadded count, set
    // OUT OF GRAPH by `set_runtime_num_tokens` before each captured
    // replay) over `slab->num_tokens` (captured bucket capacity).
    // Sentinel 0 → fall back to slab capacity; bounded by
    // effective_n <= slab->num_tokens so the worker never reads past
    // the slab's pinned buffer.
    const int32_t slab_cap = slab->num_tokens.load(std::memory_order_acquire);
    const int32_t override_n =
        runtime_num_tokens_.load(std::memory_order_acquire);
    const int32_t n = (override_n > 0) ? override_n : slab_cap;
    TORCH_CHECK(n <= slab_cap, "RunSlabOnWorker: runtime_num_tokens=", n,
                " exceeds slab capacity (slab.num_tokens=", slab_cap,
                ") at task_id=", static_cast<int64_t>(slab - slabs_.get()),
                " — would read past the pinned buffer's tail");

    // §1c.21 fix-validation: bin the effective_n the worker
    // actually used. If submit-time num_tokens histogram is mostly
    // `nt_gt_64` but this is mostly `nt_le_1`, the override is
    // working as intended.
    {
      int hist_bin;
      if (n <= 1)
        hist_bin = 0;
      else if (n <= 2)
        hist_bin = 1;
      else if (n <= 4)
        hist_bin = 2;
      else if (n <= 8)
        hist_bin = 3;
      else if (n <= 16)
        hist_bin = 4;
      else if (n <= 32)
        hist_bin = 5;
      else if (n <= 64)
        hist_bin = 6;
      else
        hist_bin = 7;
      worker_effective_n_hist_[hist_bin].fetch_add(1,
                                                   std::memory_order_relaxed);
    }

    switch (slab->op_kind) {
      case TaskSlab::kDryrunNoop: {
        // Stage 2 substrate gate: install all wrappers but skip real
        // CPU work. Mirrors `_cpu_dryrun_noop` (cots.py:1161).
        break;
      }
      case TaskSlab::kQkv: {
        // y_view <- at::linear(x_view, w_view).
        auto x_view =
            ContigCpuViewFromBlob(slab->x_pinned_ptr, n, slab->in_dim);
        auto w_view = ContigCpuViewFromBlob(slab->w_cpu_ptr, slab->w_cpu_rows,
                                            slab->in_dim);
        auto y_view =
            ContigCpuViewFromBlob(slab->y_pinned_ptr, n, slab->cpu_out_dim);
        y_view.copy_(at::linear(x_view, w_view));
        break;
      }
      case TaskSlab::kMlpBlock: {
        // gate / up are contiguous prefix views of their respective halves
        // of the gate_up CPU buffer (Phase 1b populates them this way).
        auto x_view =
            ContigCpuViewFromBlob(slab->x_pinned_ptr, n, slab->in_dim);
        auto w_gate = ContigCpuViewFromBlob(slab->w_gate_ptr, slab->w_gate_rows,
                                            slab->in_dim);
        auto w_up = ContigCpuViewFromBlob(slab->w_up_ptr, slab->w_up_rows,
                                          slab->in_dim);
        auto gate_out =
            at::linear(x_view, w_gate);  // (n, intermediate_per_half)
        auto up_out = at::linear(x_view, w_up);

        // Worker-local scratch — single max-sized tensor.
        TORCH_CHECK(scratch_silu_up_.defined(),
                    "MLP slab dispatched but scratch_silu_up_ not "
                    "allocated; install() must be called with non-zero "
                    "scratch sizes.");
        TORCH_CHECK(n <= scratch_max_tokens_, "MLP slab num_tokens (", n,
                    ") exceeds scratch_max_tokens_ (", scratch_max_tokens_,
                    ")");
        TORCH_CHECK(
            slab->intermediate_per_half <= scratch_max_intermediate_per_half_,
            "MLP slab intermediate_per_half (", slab->intermediate_per_half,
            ") exceeds scratch_max_intermediate_per_half_ (",
            scratch_max_intermediate_per_half_, ")");
        auto z = scratch_silu_up_.narrow(0, 0, n).narrow(
            1, 0, slab->intermediate_per_half);
        z.copy_(at::silu(gate_out) * up_out);

        // Down-proj: column-strided view.
        auto w_down = StridedCpuViewFromBlob(
            slab->w_down_ptr, slab->w_down_rows, slab->w_down_cols,
            slab->w_down_stride_row, slab->w_down_stride_col);
        auto y_view =
            ContigCpuViewFromBlob(slab->y_pinned_ptr, n, slab->cpu_out_dim);
        y_view.copy_(at::linear(z, w_down));
        break;
      }
    }
  } catch (const std::exception& e) {
    std::lock_guard<std::mutex> lock(error_mtx_);
    last_error_msg_ = std::string("[cots worker] ") + e.what();
    has_error_.store(true, std::memory_order_release);
  } catch (...) {
    std::lock_guard<std::mutex> lock(error_mtx_);
    last_error_msg_ = "[cots worker] unknown exception";
    has_error_.store(true, std::memory_order_release);
  }
}

// --- §1c.21 live-token override ---------------------------------------

void CotsCpuInfer::set_runtime_num_tokens(int32_t n) {
  TORCH_CHECK(n >= 0, "set_runtime_num_tokens: n=", n,
              " < 0; pass 0 to clear the override.");
  // Release store: the worker's acquire load in RunSlabOnWorker pairs
  // with this. The caller's responsibility is to set this BEFORE the
  // captured graph replay begins (i.e., from
  // CotsOffloader.prepare_before_forward, called by
  // cudagraph_utils.py:267 outside the captured region).
  runtime_num_tokens_.store(n, std::memory_order_release);
  runtime_set_calls_.fetch_add(1, std::memory_order_relaxed);
  runtime_last_value_.store(n, std::memory_order_relaxed);
}

// --- §1c.21 perf-investigation counters --------------------------------

std::vector<std::pair<std::string, int64_t>> CotsCpuInfer::get_counters()
    const {
  std::vector<std::pair<std::string, int64_t>> out;
  out.reserve(40);
  auto load = [](const std::atomic<int64_t>& a) {
    return a.load(std::memory_order_relaxed);
  };
  out.emplace_back("submit_count_qkv", load(submit_count_qkv_));
  out.emplace_back("submit_count_mlp", load(submit_count_mlp_));
  out.emplace_back("submit_count_dryrun", load(submit_count_dryrun_));
  static const char* kBinNames[8] = {"nt_le_1",  "nt_le_2",  "nt_le_4",
                                     "nt_le_8",  "nt_le_16", "nt_le_32",
                                     "nt_le_64", "nt_gt_64"};
  for (int i = 0; i < 8; ++i) {
    out.emplace_back(std::string("nt_qkv_") + kBinNames[i],
                     load(nt_hist_qkv_[i]));
  }
  for (int i = 0; i < 8; ++i) {
    out.emplace_back(std::string("nt_mlp_") + kBinNames[i],
                     load(nt_hist_mlp_[i]));
  }
  for (int i = 0; i < 8; ++i) {
    out.emplace_back(std::string("nt_dryrun_") + kBinNames[i],
                     load(nt_hist_dryrun_[i]));
  }
  out.emplace_back("d2h_1d_count", load(d2h_1d_count_));
  out.emplace_back("d2h_2d_count", load(d2h_2d_count_));
  out.emplace_back("d2h_1d_bytes", load(d2h_1d_bytes_));
  out.emplace_back("d2h_2d_bytes", load(d2h_2d_bytes_));
  out.emplace_back("runtime_set_calls", load(runtime_set_calls_));
  out.emplace_back("runtime_last_value", load(runtime_last_value_));
  for (int i = 0; i < 8; ++i) {
    out.emplace_back(std::string("worker_eff_n_") + kBinNames[i],
                     load(worker_effective_n_hist_[i]));
  }
  return out;
}

void CotsCpuInfer::reset_counters() {
  submit_count_qkv_.store(0, std::memory_order_relaxed);
  submit_count_mlp_.store(0, std::memory_order_relaxed);
  submit_count_dryrun_.store(0, std::memory_order_relaxed);
  for (int i = 0; i < 8; ++i) {
    nt_hist_qkv_[i].store(0, std::memory_order_relaxed);
    nt_hist_mlp_[i].store(0, std::memory_order_relaxed);
    nt_hist_dryrun_[i].store(0, std::memory_order_relaxed);
  }
  d2h_1d_count_.store(0, std::memory_order_relaxed);
  d2h_2d_count_.store(0, std::memory_order_relaxed);
  d2h_1d_bytes_.store(0, std::memory_order_relaxed);
  d2h_2d_bytes_.store(0, std::memory_order_relaxed);
  runtime_set_calls_.store(0, std::memory_order_relaxed);
  runtime_last_value_.store(0, std::memory_order_relaxed);
  for (int i = 0; i < 8; ++i) {
    worker_effective_n_hist_[i].store(0, std::memory_order_relaxed);
  }
}

}  // namespace cots
}  // namespace vllm
