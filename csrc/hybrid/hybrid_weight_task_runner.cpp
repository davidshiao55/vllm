// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Phase 1c — native weight task runner implementation.

#include "hybrid_weight_task_runner.h"

#include "bf16_kernels.h"
#include "hybrid_common.h"

#include <ATen/ops/linear.h>
#include <c10/core/InferenceMode.h>
#include <pthread.h>
#include <sched.h>
#include <torch/torch.h>

#include <algorithm>
#include <exception>
#include <stdexcept>
#include <utility>

namespace vllm {
namespace hybrid {

namespace {

constexpr int kMaxCpus = 64;

// Instrumentation gates. Counters and NVTX are controlled independently so
// benchmark runs can collect cheap counters without emitting NVTX ranges.
// NVTX is shared across Hybrid runners via NvtxScope in hybrid_common.h.
namespace hybrid_diag {
inline bool counters_enabled() {
  static const bool enabled = []() {
    return env_flag("VLLM_HYBRID_COUNTERS");
  }();
  return enabled;
}

}  // namespace hybrid_diag

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

}  // namespace

HybridWeightTaskRunner::HybridWeightTaskRunner()
    : task_queue_(std::make_unique<TaskQueue>()) {
  sync_args_.runner = static_cast<void*>(this);
  sync_args_.allow_n_pending = 0;
}

HybridWeightTaskRunner::~HybridWeightTaskRunner() {
  // Drain before destructing the queue so any in-flight task completes.
  if (task_queue_) {
    task_queue_->sync(0);
  }
}

void HybridWeightTaskRunner::install(int64_t n_slabs, int64_t max_num_tokens) {
  TORCH_CHECK(n_slabs >= 0, "install: n_slabs must be >= 0, got ", n_slabs);
  TORCH_CHECK(
      !slabs_,
      "install: HybridWeightTaskRunner already installed; call once per "
      "HybridWeightTaskRunner instance");

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
  max_num_tokens_ = max_num_tokens;
}

void HybridWeightTaskRunner::check_error() {
  if (!has_error_.load(std::memory_order_acquire)) return;
  std::lock_guard<std::mutex> lock(error_mtx_);
  std::string msg = std::move(last_error_msg_);
  last_error_msg_.clear();
  has_error_.store(false, std::memory_order_release);
  // Throwing std::runtime_error: pybind11 maps it to Python RuntimeError
  // automatically. The next Python entry point re-raises the stored worker
  // error instead of letting a failed task hang the engine.
  throw std::runtime_error(msg);
}

int32_t HybridWeightTaskRunner::slab_bucket_capacity_tokens(
    int64_t task_id) const {
  TORCH_CHECK(task_id >= 0 && task_id < slab_count_,
              "slab_bucket_capacity_tokens: task_id ", task_id,
              " out of range");
  return slabs_[task_id].bucket_capacity_tokens;
}

int32_t HybridWeightTaskRunner::slab_num_tokens(int64_t task_id) const {
  TORCH_CHECK(task_id >= 0 && task_id < slab_count_,
              "slab_num_tokens: task_id ", task_id, " out of range");
  return slabs_[task_id].num_tokens.load(std::memory_order_acquire);
}

void HybridWeightTaskRunner::populate_slab_qkv(
    int64_t task_id, int32_t n_threads, int32_t bucket_capacity_tokens,
    uintptr_t x_pinned_ptr, int32_t in_dim, uintptr_t y_pinned_ptr,
    int32_t cpu_out_dim, uintptr_t w_cpu_ptr, int32_t w_cpu_rows) {
  check_error();
  TORCH_CHECK(task_id >= 0 && task_id < slab_count_,
              "populate_slab_qkv: task_id ", task_id, " out of range");
  TORCH_CHECK(bucket_capacity_tokens >= 0,
              "populate_slab_qkv: bucket_capacity_tokens=",
              bucket_capacity_tokens, " < 0");
  TaskSlab& s = slabs_[task_id];
  s.op_kind = TaskSlab::kQkv;
  s.n_threads = n_threads;
  s.bucket_capacity_tokens = bucket_capacity_tokens;
  s.x_pinned_ptr = reinterpret_cast<void*>(x_pinned_ptr);
  s.in_dim = in_dim;
  s.y_pinned_ptr = reinterpret_cast<void*>(y_pinned_ptr);
  s.cpu_out_dim = cpu_out_dim;
  s.w_cpu_ptr = reinterpret_cast<void*>(w_cpu_ptr);
  s.w_cpu_rows = w_cpu_rows;
}

void HybridWeightTaskRunner::populate_slab_mlp(
    int64_t task_id, int32_t n_threads, int32_t bucket_capacity_tokens,
    uintptr_t x_pinned_ptr, int32_t in_dim, uintptr_t y_pinned_ptr,
    int32_t cpu_out_dim, uintptr_t w_gate_ptr, int32_t w_gate_rows,
    uintptr_t w_up_ptr, int32_t w_up_rows, uintptr_t w_down_ptr,
    int32_t w_down_rows, int32_t w_down_cols) {
  check_error();
  TORCH_CHECK(task_id >= 0 && task_id < slab_count_,
              "populate_slab_mlp: task_id ", task_id, " out of range");
  TORCH_CHECK(bucket_capacity_tokens >= 0,
              "populate_slab_mlp: bucket_capacity_tokens=",
              bucket_capacity_tokens, " < 0");
  TaskSlab& s = slabs_[task_id];
  s.op_kind = TaskSlab::kMlpBlock;
  s.n_threads = n_threads;
  s.bucket_capacity_tokens = bucket_capacity_tokens;
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
}

void HybridWeightTaskRunner::populate_slab_dryrun(
    int64_t task_id, int32_t bucket_capacity_tokens, uintptr_t x_pinned_ptr,
    int32_t in_dim, uintptr_t y_pinned_ptr, int32_t cpu_out_dim) {
  check_error();
  TORCH_CHECK(task_id >= 0 && task_id < slab_count_,
              "populate_slab_dryrun: task_id ", task_id, " out of range");
  TORCH_CHECK(bucket_capacity_tokens >= 0,
              "populate_slab_dryrun: bucket_capacity_tokens=",
              bucket_capacity_tokens, " < 0");
  TaskSlab& s = slabs_[task_id];
  s.op_kind = TaskSlab::kDryrunNoop;
  s.bucket_capacity_tokens = bucket_capacity_tokens;
  // §1c.20: dryrun must carry the x_pinned + y_pinned pointers so
  // submit_on_stream's D2H and sync's y_pinned_view both resolve.
  s.x_pinned_ptr = reinterpret_cast<void*>(x_pinned_ptr);
  s.in_dim = in_dim;
  s.y_pinned_ptr = reinterpret_cast<void*>(y_pinned_ptr);
  s.cpu_out_dim = cpu_out_dim;
}

void HybridWeightTaskRunner::submit_on_stream(
    int64_t task_id, int32_t num_tokens, uintptr_t x_gpu_ptr, int64_t x_cols,
    int64_t x_stride0, int64_t x_stride1, uintptr_t cuda_stream) {
  NvtxScope nvtx_scope("hybrid:submit_on_stream");
  // Surface any prior worker error BEFORE queueing more work.
  check_error();
  TORCH_CHECK(task_id >= 0 && task_id < slab_count_,
              "submit_on_stream: task_id ", task_id, " out of range");
  TaskSlab* slab = &slabs_[task_id];
  TORCH_CHECK(num_tokens >= 0, "submit_on_stream: num_tokens=", num_tokens,
              " < 0");
  // §1c.20: bound check against max_num_tokens_ catches Planner
  // mistakes where an oversized batch reaches a slab. Skipped when
  // max_num_tokens_==0 (test fixtures that don't run real tokens
  // through the slab — diagnostic-only path).
  TORCH_CHECK(max_num_tokens_ == 0 || num_tokens <= max_num_tokens_,
              "submit_on_stream: num_tokens=", num_tokens,
              " exceeds max_num_tokens=", max_num_tokens_,
              " (would write past the pinned buffer's tail)");
  // §1c.35 commit-2: clamp the Python-passed num_tokens at the
  // slab's bucket capacity. The passed value comes from
  // `int(x_gpu.shape[0])` which under vLLM's PIECEWISE compile +
  // BACKED dynamic shapes resolves to the SymInt hint
  // (typically max_num_batched_tokens or another large constant),
  // baked into the captured custom-op argument and shared across
  // all per-bucket FULL captures. `slab->bucket_capacity_tokens`
  // is IMMUTABLE (set at install time per (layer, bucket,
  // op_kind) from `op_descriptor[1]`) and IS the dispatched
  // bucket. Clamping here makes the captured D2H byte count and
  // `slab.num_tokens` tight per-bucket — matching what
  // §1c.21's runtime override clamps the worker GEMM to (live).
  // For B=1 decode at bucket=1: effective_n=1 even without the
  // override.
  num_tokens = std::min(num_tokens, slab->bucket_capacity_tokens);
  slab->num_tokens.store(num_tokens, std::memory_order_release);
  // §1c.21 counters: bump submit_count + num_tokens histogram for
  // this op kind. Diag-gated (§1c.34 cleanup C) — production-default
  // path skips the atomic adds entirely.
  if (hybrid_diag::counters_enabled()) {
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
    NvtxScope d2h_scope("hybrid:d2h_record");
    // §1c.34 cleanup C: D2H byte/count counters are diagnostic only;
    // diag-gate them so the hot path skips the atomic adds.
    const bool d2h_diag = hybrid_diag::counters_enabled();
    if (x_stride0 == x_cols) {
      // Contiguous row layout — single 1D copy is fastest.
      const size_t bytes = static_cast<size_t>(num_tokens) * width_bytes;
      if (d2h_diag) {
        d2h_1d_count_.fetch_add(1, std::memory_order_relaxed);
        d2h_record_bytes_1d_.fetch_add(static_cast<int64_t>(bytes),
                                       std::memory_order_relaxed);
      }
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
      if (d2h_diag) {
        d2h_2d_count_.fetch_add(1, std::memory_order_relaxed);
        d2h_record_bytes_2d_.fetch_add(static_cast<int64_t>(bytes_2d),
                                       std::memory_order_relaxed);
      }
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
  NvtxScope launch_scope("hybrid:launch_dispatch_cb");
  cudaError_t err =
      cudaLaunchHostFunc(stream, &HybridWeightTaskRunner::DispatchCallback,
                         static_cast<void*>(slab));
  TORCH_CHECK(
      err == cudaSuccess,
      "cudaLaunchHostFunc(DispatchCallback) failed: ", cudaGetErrorString(err));
}

void HybridWeightTaskRunner::sync_on_stream(uintptr_t cuda_stream) {
  NvtxScope nvtx_scope("hybrid:sync_on_stream");
  check_error();
  // sync_args_ is a stable member of *this; safe to take its address as
  // userData for cudaLaunchHostFunc, including across CUDA graph replays.
  cudaError_t err = cudaLaunchHostFunc(
      reinterpret_cast<cudaStream_t>(cuda_stream),
      &HybridWeightTaskRunner::SyncCallback, static_cast<void*>(&sync_args_));
  TORCH_CHECK(err == cudaSuccess, "cudaLaunchHostFunc(SyncCallback) failed: ",
              cudaGetErrorString(err));
}

void HybridWeightTaskRunner::publish_live_num_tokens_on_stream(
    int32_t n, uintptr_t cuda_stream) {
  check_error();
  TORCH_CHECK(n > 0, "publish_live_num_tokens_on_stream: n=", n,
              " must be positive");

  // The Python thread can enqueue multiple forwards before the CUDA stream
  // reaches any of them. Give every publication immutable callback data;
  // reusing one runner-global argument record would recreate the same race.
  auto args = std::make_unique<LiveTokenPublishArgs>();
  args->runner = static_cast<void*>(this);
  args->live_num_tokens = n;
  cudaError_t err =
      cudaLaunchHostFunc(reinterpret_cast<cudaStream_t>(cuda_stream),
                         &HybridWeightTaskRunner::PublishLiveTokenCallback,
                         static_cast<void*>(args.get()));
  TORCH_CHECK(err == cudaSuccess,
              "cudaLaunchHostFunc(PublishLiveTokenCallback) failed: ",
              cudaGetErrorString(err));
  // PublishLiveTokenCallback owns and deletes the record after the CUDA stream
  // reaches it.
  args.release();
}

void HybridWeightTaskRunner::sync_blocking() {
  task_queue_->sync(0);
  // Surface any worker error that fired while we were waiting.
  check_error();
}

void HybridWeightTaskRunner::set_worker_affinity(uint64_t cpu_set) {
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

std::string HybridWeightTaskRunner::take_error() {
  std::lock_guard<std::mutex> lock(error_mtx_);
  std::string msg = std::move(last_error_msg_);
  last_error_msg_.clear();
  has_error_.store(false, std::memory_order_release);
  return msg;
}

void HybridWeightTaskRunner::run_at_linear_inline(at::Tensor x, at::Tensor w,
                                                  at::Tensor y_out) {
  // Stage 1 microbench helper: directly drive `at::linear` from C++.
  // No TaskQueue, no host callback. Lets the test compare wall-clock
  // against Python `F.linear` on bit-identical tensors and catch the
  // catastrophic-scalar-fallback path documented in
  // `docs/phase0_findings.md §0.3.2`.
  c10::InferenceMode g;
  // c10::AutoDispatchBelowAutograd is implied by InferenceMode.
  y_out.copy_(at::linear(x, w));
}

void HybridWeightTaskRunner::run_bf16_gemm_transposed_inline(at::Tensor x,
                                                             at::Tensor w,
                                                             at::Tensor y_out) {
  c10::InferenceMode g;
  bf16_gemm_transposed_at(x, w, y_out);
}

void HybridWeightTaskRunner::run_bf16_gemm_natural_inline(at::Tensor x,
                                                          at::Tensor w,
                                                          at::Tensor y_out) {
  c10::InferenceMode g;
  bf16_gemm_natural_at(x, w, y_out);
}

void HybridWeightTaskRunner::run_bf16_mlp_inline(at::Tensor x,
                                                 at::Tensor w_gate,
                                                 at::Tensor w_up,
                                                 at::Tensor w_down,
                                                 at::Tensor y_out) {
  c10::InferenceMode g;
  TORCH_CHECK(x.device().is_cpu() && w_gate.device().is_cpu() &&
                  w_up.device().is_cpu() && w_down.device().is_cpu() &&
                  y_out.device().is_cpu(),
              "run_bf16_mlp_inline: all tensors must be CPU tensors");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 &&
                  w_gate.scalar_type() == at::kBFloat16 &&
                  w_up.scalar_type() == at::kBFloat16 &&
                  w_down.scalar_type() == at::kBFloat16 &&
                  y_out.scalar_type() == at::kBFloat16,
              "run_bf16_mlp_inline: all tensors must be bfloat16");
  TORCH_CHECK(x.is_contiguous() && w_gate.is_contiguous() &&
                  w_up.is_contiguous() && w_down.is_contiguous() &&
                  y_out.is_contiguous(),
              "run_bf16_mlp_inline: all tensors must be contiguous");
  TORCH_CHECK(x.dim() == 2 && w_gate.dim() == 2 && w_up.dim() == 2 &&
                  w_down.dim() == 2 && y_out.dim() == 2,
              "run_bf16_mlp_inline: all tensors must be rank-2");

  const int64_t M = x.size(0);
  const int64_t H = x.size(1);
  const int64_t I = w_gate.size(0);
  const int64_t O = w_down.size(1);
  TORCH_CHECK(w_gate.size(1) == H,
              "run_bf16_mlp_inline: w_gate shape must be (I, H)");
  TORCH_CHECK(w_up.size(0) == I && w_up.size(1) == H,
              "run_bf16_mlp_inline: w_up shape must match w_gate");
  TORCH_CHECK(w_down.size(0) == I,
              "run_bf16_mlp_inline: w_down shape must be (I, O)");
  TORCH_CHECK(y_out.size(0) == M && y_out.size(1) == O,
              "run_bf16_mlp_inline: y_out shape must be (M, O)");

  const int64_t z_elems = M * I;
  mlp_scratch_bf16_.resize(static_cast<size_t>(std::max<int64_t>(z_elems, 0)));
  bf16_mlp_gate_up_silu_down(
      reinterpret_cast<const uint16_t*>(x.data_ptr()),
      reinterpret_cast<const uint16_t*>(w_gate.data_ptr()),
      reinterpret_cast<const uint16_t*>(w_up.data_ptr()),
      reinterpret_cast<const uint16_t*>(w_down.data_ptr()),
      reinterpret_cast<uint16_t*>(y_out.data_ptr()), mlp_scratch_bf16_.data(),
      M, H, I, O);
}

at::Tensor HybridWeightTaskRunner::y_pinned_view(int64_t task_id,
                                                 int32_t num_tokens) const {
  // §1c.20: build an `at::from_blob` CPU tensor view over the slab's
  // pinned output buffer. Used by `hybrid_sync_then_uva`'s impl on the
  // captured-graph hot path so the CPU output tensor is NEVER a
  // custom-op argument visible to Inductor.
  if (task_id < 0 || task_id >= slab_count_) {
    throw std::out_of_range("HybridWeightTaskRunner::y_pinned_view: task_id=" +
                            std::to_string(task_id) + " out of range [0, " +
                            std::to_string(slab_count_) + ")");
  }
  const TaskSlab& slab = slabs_[task_id];
  if (slab.y_pinned_ptr == nullptr) {
    throw std::runtime_error(
        "HybridWeightTaskRunner::y_pinned_view: slab.y_pinned_ptr is null at "
        "task_id=" +
        std::to_string(task_id) + " (slab not populated?)");
  }
  if (num_tokens < 0) {
    throw std::invalid_argument(
        "HybridWeightTaskRunner::y_pinned_view: num_tokens=" +
        std::to_string(num_tokens) + " < 0");
  }
  if (max_num_tokens_ != 0 && num_tokens > max_num_tokens_) {
    throw std::invalid_argument(
        "HybridWeightTaskRunner::y_pinned_view: num_tokens=" +
        std::to_string(num_tokens) +
        " exceeds max_num_tokens=" + std::to_string(max_num_tokens_) +
        " (would read past the pinned buffer's tail)");
  }
  // §1c.35 commit-2: clamp at the slab's bucket capacity (same
  // rationale as submit_on_stream above). Makes the returned view
  // bucket-sized — so the captured UVA Triton kernel's grid is
  // sized to bucket * cpu_out_dim, not max * cpu_out_dim. The
  // operator's destination view (sized to the
  // Inductor-baked int(y_gpu.shape[0])) may be larger; the
  // Triton kernel writes src.numel() elements into the prefix
  // of dst, leaving the tail rows untouched. Downstream consumes
  // only the first `live_count <= bucket` rows (see also
  // _uva_copy_trusted_host_into_gpu's relaxed shape assertion).
  num_tokens = std::min(num_tokens, slab.bucket_capacity_tokens);
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

// Establish the live-row count in CUDA-stream order. on_dispatch can run far
// ahead on the Python thread, but these callbacks remain ordered with the
// eager forward / graph replay and its captured DispatchCallbacks.
void HybridWeightTaskRunner::PublishLiveTokenCallback(void* user_data) {
  std::unique_ptr<LiveTokenPublishArgs> args(
      static_cast<LiveTokenPublishArgs*>(user_data));
  auto* self = static_cast<HybridWeightTaskRunner*>(args->runner);
  self->stream_live_num_tokens_.store(args->live_num_tokens,
                                      std::memory_order_release);
  if (hybrid_diag::counters_enabled()) {
    self->live_set_calls_.fetch_add(1, std::memory_order_relaxed);
    self->live_last_value_.store(args->live_num_tokens,
                                 std::memory_order_relaxed);
  }
}

// Submit-side host callback. Runs on the CUDA driver thread; must NOT
// block (CUDA stream is paused while we run). Just enqueues to the
// TaskQueue worker.
void HybridWeightTaskRunner::DispatchCallback(void* user_data) {
  NvtxScope nvtx_scope("hybrid:dispatch_cb");
  TaskSlab* slab = static_cast<TaskSlab*>(user_data);
  HybridWeightTaskRunner* self =
      static_cast<HybridWeightTaskRunner*>(slab->self);
  const int32_t live_num_tokens =
      self->stream_live_num_tokens_.load(std::memory_order_acquire);
  const int32_t slab_cap = slab->bucket_capacity_tokens;
  // A zero live value occurs only in direct native fixtures / graph capture
  // setup that did not publish a runtime dispatch. Preserve their historical
  // bucket-sized behavior. Production on_dispatch always publishes > 0.
  const int32_t effective_num_tokens =
      live_num_tokens > 0 ? std::min(live_num_tokens, slab_cap) : slab_cap;
  // §1c.24 attribution: stamp enqueue time so the worker can later
  // compute queue_wait = worker_start - enqueue_time. Gated by
  // VLLM_HYBRID_COUNTERS=1. Carry the timestamp by value alongside the live
  // rows so slab reuse cannot corrupt diagnostics either.
  const bool diag = hybrid_diag::counters_enabled();
  const int64_t enqueue_time_ns = diag ? now_ns() : 0;
  if (diag) {
    self->dispatch_cb_count_.fetch_add(1, std::memory_order_relaxed);
    if (live_num_tokens > slab_cap) {
      self->worker_clamp_override_count_.fetch_add(1,
                                                   std::memory_order_relaxed);
    }
  }
  self->task_queue_->enqueue(
      [self, slab, effective_num_tokens, enqueue_time_ns] {
        self->RunSlabOnWorker(slab, effective_num_tokens, enqueue_time_ns);
      });
}

// Sync-side host callback. Blocks the CUDA driver thread until the
// TaskQueue drains. The user_data is `&self->sync_args_` (stable member).
void HybridWeightTaskRunner::SyncCallback(void* user_data) {
  NvtxScope nvtx_scope("hybrid:sync_cb_wait");
  SyncArgs* args = static_cast<SyncArgs*>(user_data);
  HybridWeightTaskRunner* self =
      static_cast<HybridWeightTaskRunner*>(args->runner);
  // §1c.24 attribution: time the sync wait — distinguishes "driver
  // blocked waiting for the worker" from "driver doing other work
  // then unblocking immediately". Same VLLM_HYBRID_COUNTERS gate as the
  // dispatch counter; in production-default mode the timestamps
  // and atomic adds are skipped.
  if (hybrid_diag::counters_enabled()) {
    const int64_t t0 = now_ns();
    self->task_queue_->sync(args->allow_n_pending);
    const int64_t t1 = now_ns();
    self->sync_cb_wait_total_ns_.fetch_add(t1 - t0, std::memory_order_relaxed);
    self->sync_cb_count_.fetch_add(1, std::memory_order_relaxed);
  } else {
    self->task_queue_->sync(args->allow_n_pending);
  }
}

// --- worker-thread task body ----------------------------------------------

void HybridWeightTaskRunner::RunSlabOnWorker(TaskSlab* slab,
                                             int32_t effective_num_tokens,
                                             int64_t enqueue_time_ns) {
  // §1c.24 attribution: stamp worker start + queue wait, gated by
  // VLLM_HYBRID_COUNTERS. Production-default leaves worker_t0 at 0 (the
  // worker_busy_total_ns add at the end is also gated). NVTX scope
  // is independently gated inside NvtxScope's ctor.
  const bool diag = hybrid_diag::counters_enabled();
  const int64_t worker_t0 = diag ? now_ns() : 0;
  if (diag && enqueue_time_ns > 0) {
    worker_queue_wait_total_ns_.fetch_add(worker_t0 - enqueue_time_ns,
                                          std::memory_order_relaxed);
  }
  const char* nvtx_name = "hybrid:worker";
  switch (slab->op_kind) {
    case TaskSlab::kQkv:
      nvtx_name = "hybrid:worker_qkv";
      break;
    case TaskSlab::kMlpBlock:
      nvtx_name = "hybrid:worker_mlp";
      break;
    case TaskSlab::kDryrunNoop:
      nvtx_name = "hybrid:worker_dryrun";
      break;
  }
  NvtxScope nvtx_scope(nvtx_name);
  // Worker exception policy: every body wrapped in try/catch so that
  // pending-decrement / cv_ notify in TaskQueue::Worker still happens
  // (we return normally from this function), and the next Python-side
  // submit/sync call surfaces the error as a Python RuntimeError.
  try {
    c10::InferenceMode g;

    // Bucket-aware thread policy: only call set_num_threads when it would
    // change. Stage 4 populates slab->n_threads from the per-bucket map;
    // Stage 1 leaves n_threads = 1, which means we call set_num_threads(1)
    // on the very first task and then stay at 1. This preserves the Phase 1a
    // scalar `cpu_num_threads` default while allowing per-bucket overrides.
    if (slab->n_threads > 0 && slab->n_threads != worker_current_n_threads_) {
      at::set_num_threads(slab->n_threads);
      worker_current_n_threads_ = slab->n_threads;
    }
    last_observed_num_threads_.store(at::get_num_threads(),
                                     std::memory_order_release);

    // DispatchCallback already clamped and copied this value into the queue
    // closure. Never reread mutable runner or slab state here: later forwards
    // and graph replays may be published while this task waits.
    const int32_t n = effective_num_tokens;

    // §1c.21 fix-validation + §1c.22 byte accounting: bin the
    // effective_n the worker actually used, plus per-replay bucket
    // and live byte counters. All diag-gated (§1c.34 cleanup C);
    // production-default skips the atomic adds entirely.
    if (diag) {
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
      // §1c.22 byte accounting. Read the IMMUTABLE
      // `bucket_capacity_tokens` populated at install time (NOT
      // mutable `slab->num_tokens` which can be overwritten by
      // later submit_on_stream calls during graph capture /
      // PIECEWISE Python re-execution).
      const int64_t bucket_n =
          static_cast<int64_t>(slab->bucket_capacity_tokens);
      if (slab->in_dim > 0) {
        const int64_t row_bytes_in = static_cast<int64_t>(slab->in_dim) *
                                     static_cast<int64_t>(sizeof(at::BFloat16));
        d2h_replay_bucket_bytes_.fetch_add(bucket_n * row_bytes_in,
                                           std::memory_order_relaxed);
        if (n > 0) {
          worker_input_live_bytes_.fetch_add(
              static_cast<int64_t>(n) * row_bytes_in,
              std::memory_order_relaxed);
        }
      }
      if (slab->cpu_out_dim > 0) {
        const int64_t row_bytes_out =
            static_cast<int64_t>(slab->cpu_out_dim) *
            static_cast<int64_t>(sizeof(at::BFloat16));
        uva_replay_bucket_bytes_.fetch_add(bucket_n * row_bytes_out,
                                           std::memory_order_relaxed);
        if (n > 0) {
          worker_output_live_bytes_.fetch_add(
              static_cast<int64_t>(n) * row_bytes_out,
              std::memory_order_relaxed);
        }
      }
    }

    switch (slab->op_kind) {
      case TaskSlab::kDryrunNoop: {
        // Stage 2 substrate gate: install all wrappers but skip real
        // CPU work.
        break;
      }
      case TaskSlab::kQkv: {
        // Stage 7-C: replace at::linear (oneDNN bf16:bf16 emulation
        // path, ~30 µs dispatch + scratch reorder per call) with our
        // custom natural-layout BF16 GEMM. Same (N, K) row-major
        // weight layout the slab already carries; output is written
        // in-place into y_view's pinned-memory backing.
        auto x_view =
            ContigCpuViewFromBlob(slab->x_pinned_ptr, n, slab->in_dim);
        auto w_view = ContigCpuViewFromBlob(slab->w_cpu_ptr, slab->w_cpu_rows,
                                            slab->in_dim);
        auto y_view =
            ContigCpuViewFromBlob(slab->y_pinned_ptr, n, slab->cpu_out_dim);
        bf16_gemm_natural_at(x_view, w_view, y_view);
        break;
      }
      case TaskSlab::kMlpBlock: {
        TORCH_CHECK(slab->w_gate_rows == slab->w_up_rows,
                    "MLP CPU path requires gate/up row counts to match, got ",
                    slab->w_gate_rows, " and ", slab->w_up_rows);
        TORCH_CHECK(slab->w_down_rows == slab->w_gate_rows,
                    "MLP CPU path requires down K to match intermediate rows, "
                    "got down_rows=",
                    slab->w_down_rows, " gate_rows=", slab->w_gate_rows);
        TORCH_CHECK(slab->w_down_cols == slab->cpu_out_dim,
                    "MLP CPU path output dim mismatch: down_cols=",
                    slab->w_down_cols, " cpu_out_dim=", slab->cpu_out_dim);
        const int64_t z_elems =
            static_cast<int64_t>(n) * static_cast<int64_t>(slab->w_down_rows);
        mlp_scratch_bf16_.resize(static_cast<size_t>(
            std::max<int64_t>(z_elems, static_cast<int64_t>(0))));
        bf16_mlp_gate_up_silu_down(
            reinterpret_cast<const uint16_t*>(slab->x_pinned_ptr),
            reinterpret_cast<const uint16_t*>(slab->w_gate_ptr),
            reinterpret_cast<const uint16_t*>(slab->w_up_ptr),
            reinterpret_cast<const uint16_t*>(slab->w_down_ptr),
            reinterpret_cast<uint16_t*>(slab->y_pinned_ptr),
            mlp_scratch_bf16_.data(), n, slab->in_dim, slab->w_down_rows,
            slab->w_down_cols);
        break;
      }
    }
  } catch (const std::exception& e) {
    std::lock_guard<std::mutex> lock(error_mtx_);
    last_error_msg_ = std::string("[hybrid worker] ") + e.what();
    has_error_.store(true, std::memory_order_release);
  } catch (...) {
    std::lock_guard<std::mutex> lock(error_mtx_);
    last_error_msg_ = "[hybrid worker] unknown exception";
    has_error_.store(true, std::memory_order_release);
  }
  // §1c.24: worker compute duration (NVTX scope ends when the
  // function returns; this counter sums it for the bench summary).
  // Gated together with the start-of-function timestamp.
  if (diag) {
    worker_busy_total_ns_.fetch_add(now_ns() - worker_t0,
                                    std::memory_order_relaxed);
    worker_run_count_.fetch_add(1, std::memory_order_relaxed);
  }
}

// --- §1c.21 live-token accounting -------------------------------------

void HybridWeightTaskRunner::note_uva_request(int32_t num_tokens,
                                              int32_t cpu_out_dim) {
  // §1c.22 bookkeeping. Diag-gated (§1c.34 cleanup C) —
  // measurement-only path; no functional dependency.
  if (num_tokens <= 0 || cpu_out_dim <= 0) return;
  if (!hybrid_diag::counters_enabled()) return;
  const int64_t bytes = static_cast<int64_t>(num_tokens) *
                        static_cast<int64_t>(cpu_out_dim) *
                        static_cast<int64_t>(sizeof(at::BFloat16));
  uva_record_bytes_.fetch_add(bytes, std::memory_order_relaxed);
  uva_record_count_.fetch_add(1, std::memory_order_relaxed);
}

// --- §1c.21 perf-investigation counters --------------------------------

std::vector<std::pair<std::string, int64_t>>
HybridWeightTaskRunner::get_counters() const {
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
  // §1c.22 — record-time counters (capture/warmup only).
  out.emplace_back("d2h_1d_count", load(d2h_1d_count_));
  out.emplace_back("d2h_2d_count", load(d2h_2d_count_));
  out.emplace_back("d2h_record_bytes_1d", load(d2h_record_bytes_1d_));
  out.emplace_back("d2h_record_bytes_2d", load(d2h_record_bytes_2d_));
  // §1c.22 — replay-time counters (incremented in RunSlabOnWorker
  // which fires per replay because the captured host callback
  // re-executes). Apples-to-apples with worker_*_live_bytes.
  out.emplace_back("d2h_replay_bucket_bytes", load(d2h_replay_bucket_bytes_));
  out.emplace_back("uva_replay_bucket_bytes", load(uva_replay_bucket_bytes_));
  out.emplace_back("live_set_calls", load(live_set_calls_));
  out.emplace_back("live_last_value", load(live_last_value_));
  for (int i = 0; i < 8; ++i) {
    out.emplace_back(std::string("worker_eff_n_") + kBinNames[i],
                     load(worker_effective_n_hist_[i]));
  }
  out.emplace_back("worker_input_live_bytes", load(worker_input_live_bytes_));
  out.emplace_back("worker_output_live_bytes", load(worker_output_live_bytes_));
  // §1c.22 — record-time UVA accounting (captured Triton grid
  // size, set once during graph capture by Python-side
  // note_uva_request).
  out.emplace_back("uva_record_bytes", load(uva_record_bytes_));
  out.emplace_back("uva_record_count", load(uva_record_count_));
  // §1c.24 attribution counters. ns totals + invocation counts.
  out.emplace_back("dispatch_cb_count", load(dispatch_cb_count_));
  out.emplace_back("sync_cb_count", load(sync_cb_count_));
  out.emplace_back("sync_cb_wait_total_ns", load(sync_cb_wait_total_ns_));
  out.emplace_back("worker_run_count", load(worker_run_count_));
  out.emplace_back("worker_busy_total_ns", load(worker_busy_total_ns_));
  out.emplace_back("worker_queue_wait_total_ns",
                   load(worker_queue_wait_total_ns_));
  out.emplace_back("worker_clamp_override_count",
                   load(worker_clamp_override_count_));
  return out;
}

void HybridWeightTaskRunner::reset_counters() {
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
  d2h_record_bytes_1d_.store(0, std::memory_order_relaxed);
  d2h_record_bytes_2d_.store(0, std::memory_order_relaxed);
  live_set_calls_.store(0, std::memory_order_relaxed);
  live_last_value_.store(0, std::memory_order_relaxed);
  for (int i = 0; i < 8; ++i) {
    worker_effective_n_hist_[i].store(0, std::memory_order_relaxed);
  }
  worker_input_live_bytes_.store(0, std::memory_order_relaxed);
  worker_output_live_bytes_.store(0, std::memory_order_relaxed);
  uva_record_bytes_.store(0, std::memory_order_relaxed);
  uva_record_count_.store(0, std::memory_order_relaxed);
  d2h_replay_bucket_bytes_.store(0, std::memory_order_relaxed);
  uva_replay_bucket_bytes_.store(0, std::memory_order_relaxed);
  // §1c.24 attribution counters.
  dispatch_cb_count_.store(0, std::memory_order_relaxed);
  sync_cb_count_.store(0, std::memory_order_relaxed);
  sync_cb_wait_total_ns_.store(0, std::memory_order_relaxed);
  worker_run_count_.store(0, std::memory_order_relaxed);
  worker_busy_total_ns_.store(0, std::memory_order_relaxed);
  worker_queue_wait_total_ns_.store(0, std::memory_order_relaxed);
  worker_clamp_override_count_.store(0, std::memory_order_relaxed);
}

}  // namespace hybrid
}  // namespace vllm
