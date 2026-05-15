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

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
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
//   * Stage 7-C: down-proj is row-narrow on transposed `(n_cpu_total,
//     out_dim)` storage → contiguous (dn_n_cpu, out_dim) view, no strides
//     needed (kernel takes ContigCpuViewFromBlob, not StridedCpuViewFromBlob).
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

  // §1c.24 diagnostic-only: written in DispatchCallback when
  // VLLM_COTS_DIAG=1 so the worker can attribute (worker_start −
  // enqueue_time) as TaskQueue wait. 0 = unset (production-default
  // mode never writes this).
  std::atomic<int64_t> enqueue_time_ns{0};

  // §1c.29 wait-kernel sync (sync host_fn replacement). One req/done slot pair
  // per slab, allocated lazily by `install_wait_kernel_sync_for_task` only when
  // the wait-kernel sync flag is on. host_*_ptr is the CPU-visible
  // address (worker reads/writes); dev_*_ptr is the GPU-visible
  // address (cots_wait_done_kernel reads via host-mapped pinned).
  // `next_seq` is incremented per dispatch_cb fire from the
  // driver thread; `uint32_t` is documented as "wraps at 2^32"
  // — at 56 ops × billions of replays this is a non-issue for
  // any realistic workload, but a future robust version could
  // promote to uint64_t. wait_kernel_sync_installed is set by
  // install_wait_kernel_sync_for_task and gates the dispatch/wait paths.
  void* host_req_slot{nullptr};
  void* dev_req_slot{nullptr};
  void* host_done_slot{nullptr};
  void* dev_done_slot{nullptr};
  std::atomic<uint32_t> next_seq{0};
  std::atomic<bool> wait_kernel_sync_installed{false};

  // §1c.22 review-fix: immutable bucket capacity, populated once at
  // install time from the (layer, bucket, op_kind) descriptor and
  // never written again. Replay-time byte accounting MUST read this
  // (not `num_tokens`, which is captured/recorded state that can be
  // overwritten by later submit_on_stream calls during graph replay
  // or PIECEWISE Python re-execution). Two distinct concepts:
  //   * `num_tokens` — submit-time/captured token count (what the
  //     captured cudaMemcpyAsync was sized for at record time, but
  //     can be touched by later writes).
  //   * `bucket_capacity_tokens` — descriptor-bucket size, the value
  //     vLLM rounded num_tokens UP to in order to pick this slab.
  //     Stable for the lifetime of the slab. This is the
  //     authoritative byte-budget the captured graph reserved.
  // Note: this still ESTIMATES the captured-graph node byte count;
  // the only authoritative value is the cudaMemcpyAsync param baked
  // into the cuGraphNode at capture time. Without parsing the graph
  // we can't be exact; bucket_capacity_tokens is the closest stable
  // proxy.
  int32_t bucket_capacity_tokens = 0;

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
  // Stage 7-C: down-proj weight is stored as transposed `(n_cpu_total,
  // out_dim)` row-major. The slab carries a `narrow(0, dn_n_pref,
  // dn_n_cpu)` view — contiguous `(dn_n_cpu, out_dim) = (K, N)` —
  // feedable to the custom `bf16_gemm_transposed` kernel.
  void* w_down_ptr = nullptr;
  int32_t w_down_rows = 0;  // = K (dn_n_cpu)
  int32_t w_down_cols = 0;  // = N (out_dim)
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

  // Reserves N slabs, sized once. After install(), slabs_.size() ==
  // n_slabs is invariant for the lifetime of this CotsCpuInfer.
  // `max_num_tokens` is the upper bound on per-call num_tokens; it
  // gates submit/run-side bounds checks against the pinned x/y
  // buffers. Subsequent populate_slab calls only mutate the
  // pre-existing slab entries.
  //
  // The MLP worker owns a lazily-resized gate/up/SwiGLU BF16 scratch
  // buffer sized from the active slab shape.
  void install(int64_t n_slabs, int64_t max_num_tokens);

  // Populate a previously-reserved slab. All pointers must be POST-narrow
  // data_ptr()s (see TaskSlab doc above). Idempotent. dtype is hard-coded
  // to bfloat16 — `CotsOffloadConfig.cpu_dtype` is `Literal["bfloat16"]`
  // (per phase0 §0.3.2 oneDNN BF16 is the only fast CPU GEMM path on
  // AVX2 hardware), so passing `torch.dtype` over pybind would just be
  // a brittle int-enum dance. If we ever support fp16/fp32, take dtype
  // back as a parameter at that point.
  // §1c.22 review-fix: populate_slab_* now takes `bucket_capacity_tokens`
  // — the stable descriptor-bucket size — and stores it as immutable on
  // the slab. Replay-time byte counters read this, NOT mutable num_tokens.
  void populate_slab_qkv(int64_t task_id, int32_t n_threads,
                         int32_t bucket_capacity_tokens, uintptr_t x_pinned_ptr,
                         int32_t in_dim, uintptr_t y_pinned_ptr,
                         int32_t cpu_out_dim, uintptr_t w_cpu_ptr,
                         int32_t w_cpu_rows);

  void populate_slab_mlp(int64_t task_id, int32_t n_threads,
                         int32_t bucket_capacity_tokens, uintptr_t x_pinned_ptr,
                         int32_t in_dim, uintptr_t y_pinned_ptr,
                         int32_t cpu_out_dim, uintptr_t w_gate_ptr,
                         int32_t w_gate_rows, uintptr_t w_up_ptr,
                         int32_t w_up_rows, uintptr_t w_down_ptr,
                         int32_t w_down_rows, int32_t w_down_cols);

  // Populate a slab as a dryrun-noop. §1c.20: dryrun must still
  // carry x_pinned_ptr + in_dim (so submit_on_stream's D2H copy
  // resolves) AND y_pinned_ptr + cpu_out_dim (so the captured-graph
  // sync op's y_pinned_view call resolves) even though the worker
  // skips the GEMM. The y_pinned values are uninitialized after
  // sync but that's exactly what dryrun means — the bench measures
  // orchestration overhead, not output correctness.
  void populate_slab_dryrun(int64_t task_id, int32_t bucket_capacity_tokens,
                            uintptr_t x_pinned_ptr, int32_t in_dim,
                            uintptr_t y_pinned_ptr, int32_t cpu_out_dim);

  // Submit a task on the *current* CUDA stream. §1c.20: bundles the
  // x_gpu → x_pinned D2H copy WITH the host-callback enqueue so
  // neither x_pinned NOR x_gpu's view need to be Python tensors
  // visible to Inductor in the captured graph. The D2H is
  // stride-aware: when `x_stride0 == x_cols` (contiguous row layout)
  // we issue a single 1D `cudaMemcpyAsync`; otherwise — the
  // row-strided case real-model hidden-states paths can produce —
  // we use `cudaMemcpy2DAsync` to walk rows correctly. Both are
  // graph-capturable; the host callback we queue immediately after
  // is strictly ordered behind the copy. `x_stride1 == 1` is
  // required (no transposed-stride layouts in production decode).
  //
  // The Python custom-op `vllm.cots_submit_gemm` (registered in
  // vllm/model_executor/offloader/cots_ops.py) translates to a call
  // here with `stream = c10::cuda::getCurrentCUDAStream()`,
  // `x_gpu_ptr = x_gpu.data_ptr()`, and the shape/stride metadata
  // pulled off the tensor. The torch.ops schema stays
  // `(x_gpu, runner_id, task_id, num_tokens)` — Inductor only sees
  // CUDA tensors + scalar ids.
  void submit_on_stream(int64_t task_id, int32_t num_tokens,
                        uintptr_t x_gpu_ptr, int64_t x_cols, int64_t x_stride0,
                        int64_t x_stride1, uintptr_t cuda_stream);

  // Likewise for sync. Schedules `&sync_args_` (a stable member) as a host
  // callback on the supplied stream; the callback blocks the CUDA driver
  // thread on `task_queue_->sync(0)`.
  void sync_on_stream(uintptr_t cuda_stream);

  // §1c.29 commit 2 — unified sync/wait dispatch. Per-slab choice
  // based on `slab.wait_kernel_sync_installed`: when wait-kernel sync is
  // installed for this task (offloader called
  // `install_wait_kernel_sync_for_task` in post_init under
  // `cots_capture_sync_mode="wait_kernel"`), the captured graph node is the
  // wait-kernel sync reading the worker-published `done_slot=seq`. Otherwise
  // the captured node is the legacy `cudaLaunchHostFunc(SyncCallback)` blocking
  // the driver thread on `TaskQueue::sync(0)`. The Python side
  // (`cots_sync_then_uva`) ALWAYS calls this entry; the A/B is
  // controlled exclusively by whether the offloader installed wait-kernel sync
  // for this task at startup.
  void sync_or_wait_on_stream(int64_t task_id, uintptr_t cuda_stream);

  // Python-side helper (not in the captured-graph hot path).
  // Block calling thread until TaskQueue drains. Same as sync_on_stream but
  // without the cudaLaunchHostFunc indirection.
  void sync_blocking();

  // Bucket-aware thread policy hook. Sets the worker thread's CPU affinity
  // to `cpu_set` (a bitmask of CPU IDs packed into uint64; bit i set =>
  // CPU i is allowed). cpu_set == 0 means clear affinity. Intersect with
  // sched_getaffinity at call time to avoid EINVAL when running under a
  // restrictive cgroup. uint64 (not int64) so cpu_id 63 is representable
  // — `int64_t{1} << 63` is signed-shift UB.
  void set_worker_affinity(uint64_t cpu_set);

  // Test helper: read whether the worker has set num_threads to the value
  // the slab requested, after a sync. Used by test_bucket_thread_policy.
  int32_t last_observed_num_threads() const {
    return last_observed_num_threads_.load(std::memory_order_acquire);
  }

  // §1c.29 wait-kernel sync — install per-slab host-mapped pinned req/done
  // slots. Idempotent; calling twice for the same task_id raises.
  // Hard-fails on cudaHostAlloc(cudaHostAllocMapped) or
  // cudaHostGetDevicePointer error (per §1c.29 safety gate (c)/(d)
  // — silent fallback under graph capture would put different
  // slabs on different mechanisms, refuse to install). After
  // success, `slab.wait_kernel_sync_installed = true`.
  void install_wait_kernel_sync_for_task(int64_t task_id);

  // Launch the captured-graph wait kernel for `task_id` on
  // `cuda_stream`. Selects production vs diag kernel based on
  // VLLM_COTS_DIAG. Hard-fails if wait-kernel sync was not installed for this
  // task. This entry point CALLS check_error() first; it's the
  // public/test-helper variant. Production captured ops should
  // use `sync_or_wait_on_stream` (which routes through the
  // no-check launcher to avoid wedging the stream when a prior
  // worker task has set has_error_; see §1c.29 commit 2
  // review-fix).
  void wait_kernel_sync_on_stream(int64_t task_id, uintptr_t cuda_stream);

  // §1c.29 test helpers. Direct slot access from CPU — used by
  // the unit smoke (test_cots_wait_done_kernel_smoke.py) to drive the
  // wait kernel in isolation. NOT used on the captured-graph
  // hot path.
  // Test helper: query whether `install_wait_kernel_sync_for_task` was called
  // for `task_id`. Used by safety-gate tests to assert the
  // post_init wiring matches the config flag.
  bool wait_kernel_sync_installed_for_task(int64_t task_id) const;

  uint32_t wait_kernel_get_req_slot(int64_t task_id) const;
  uint32_t wait_kernel_get_done_slot(int64_t task_id) const;
  void wait_kernel_set_req_slot(int64_t task_id, uint32_t value);
  void wait_kernel_set_done_slot(int64_t task_id, uint32_t value);

  // §1c.22 review-fix test helpers: read the immutable
  // `bucket_capacity_tokens` and the mutable `num_tokens` for a
  // given slab. The pair lets test_bucket_capacity_immutable assert
  // that `bucket_capacity_tokens` does NOT change when
  // `runtime_num_tokens` or `slab.num_tokens` is mutated.
  int32_t slab_bucket_capacity_tokens(int64_t task_id) const;
  int32_t slab_num_tokens(int64_t task_id) const;

  // Worker exception surfacing. Each Python-side `submit*` / `sync*` call
  // checks has_error_ and re-raises last_error_msg_ as a Python
  // RuntimeError. This mirrors the Python runner's future.result()
  // re-raise semantics.
  //
  // The guard is invoked at the START of every entry point (submit*, sync*,
  // populate_slab*) so a previously-failed task
  // surfaces as a Python RuntimeError on the next call rather than getting
  // silently swallowed. The first call after a worker error consumes the
  // error (clears has_error_); subsequent calls succeed as normal.
  bool has_error() const { return has_error_.load(std::memory_order_acquire); }
  std::string take_error();
  // Throws std::runtime_error (mapped by pybind11 to RuntimeError) if a
  // worker task failed since the last error consumption. Public so a
  // direct C++ caller (e.g., a future C++ binding) can opt in.
  void check_error();

  // Lightweight load/correctness helper: run `at::linear(x, w)` on the
  // calling thread (NOT through TaskQueue) and write into `y`. Retained for
  // native-extension sanity tests; argument tensors are user-managed.
  void run_at_linear_inline(at::Tensor x, at::Tensor w, at::Tensor y_out);

  // Inline call to the custom BF16 row-major-weight GEMM kernel
  // (csrc/cots/bf16_gemm_transposed.cpp). All tensors BF16 row-major
  // contiguous. Retained for correctness tests of the production kernel
  // layout; caller-managed tensors, no TaskQueue involvement.
  void run_bf16_gemm_transposed_inline(at::Tensor x, at::Tensor w,
                                       at::Tensor y_out);

  // Stage 7-D — natural (N, K) row-major BF16 GEMM kernel.
  // Sibling to run_bf16_gemm_transposed_inline; same casting
  // and parallelism strategy but with loop nest swapped for the
  // natural-layout fast access direction. Used by the production
  // worker paths for QKV/gate/up after Stage 7-C storage
  // unification. Caller-managed tensors; no TaskQueue involvement.
  void run_bf16_gemm_natural_inline(at::Tensor x, at::Tensor w,
                                    at::Tensor y_out);

  // §1c.20: produce an `at::from_blob` CPU tensor view over the slab's
  // pinned output buffer for the given task. Trusted: the y_pinned_ptr
  // came from `_y_pinned`, which was allocated with `pin_memory=True`
  // and validated at install time — re-checking `is_pinned()` here
  // would duplicate work. Used by `cots_sync_then_uva`'s impl to
  // reach the worker's output WITHOUT exposing the CPU tensor as a
  // graph input that Inductor would materialize via a CPU↔GPU
  // shuffle. Shape: {num_tokens, slab.cpu_out_dim}, dtype bfloat16,
  // contiguous. Caller must NOT outlive the slab (i.e., this
  // CotsCpuInfer instance).
  at::Tensor y_pinned_view(int64_t task_id, int32_t num_tokens) const;

  // §1c.21 fix — live unpadded token count plumbed through OUT OF
  // GRAPH. Under vLLM full CUDA graph capture, `slab->num_tokens` is
  // frozen at the captured bucket size (e.g., 256). Replays at B=1
  // decode would otherwise run CPU GEMMs for the bucket size, costing
  // ~17 ms/GEMM × 7000 calls = ~120 s/generate wasted (see
  // phase1c_findings.md §1c.21 counter-driven diagnosis).
  //
  // Caller (CotsOffloader.prepare_before_forward, called by
  // cudagraph_utils.py before each captured replay) sets this to the
  // live unpadded token count from
  // `scheduler_output.total_num_scheduled_tokens`. The worker reads it via the
  // loaded value and uses `effective_n` for all row-count arithmetic —
  // at::from_blob shapes, GEMM input rows, scratch slicing — instead of
  // `slab->num_tokens`. Always bounded by `effective_n <= slab->num_tokens`
  // (the slab capacity) so no buffer overrun is possible.
  //
  // Sentinel 0 = unset → fall back to slab->num_tokens. Any positive
  // value overrides. Validated `n >= 0` at the entry point.
  //
  // The captured cudaMemcpyAsync byte count and the captured Triton
  // UVA grid are still sized to the bucket — those nodes are baked
  // into the graph at capture time. Only the worker's CPU-side
  // arithmetic shrinks to the live count. Rows beyond `effective_n`
  // in y_pinned hold stale data; vLLM's full-graph contract only
  // consumes the first `effective_n` rows of any output anyway.
  // Wasted PCIe / Triton bandwidth is a §1c.22 follow-up; the
  // dominant cost the counter data showed was CPU GEMM work, which
  // this fix collapses.
  void set_runtime_num_tokens(int32_t n);

  // §1c.22: invoked from `cots_sync_then_uva`'s Python impl to
  // record the captured Triton kernel's bucket-sized H2D request
  // (bytes = num_tokens × cpu_out_dim × bf16_size). Pure
  // bookkeeping — no side effect on the kernel.
  void note_uva_request(int32_t num_tokens, int32_t cpu_out_dim);

  // §1c.21 perf-investigation counters (focused). Tracks submit_count
  // and a num_tokens histogram by op kind, plus D2H 1D/2D split.
  // Reset via `reset_counters()` at the start of a measurement window;
  // read via `get_counters()` after. All increments are
  // `memory_order_relaxed` — observational, not load-bearing for
  // correctness.
  //
  // Hypothesis being tested: under vLLM full CUDA graph replay, COTS
  // sees `x_gpu.shape[0]` == captured graph-bucket size (e.g., 256)
  // even at B=1 decode, so the worker does CPU GEMMs for ~256 tokens
  // every step. The histogram pins this immediately — if
  // capture-mode runs are dominated by `nt_gt_64` while eager runs
  // sit in `nt_le_1`, the diagnosis is confirmed.
  //
  // Histogram bins are powers-of-2: nt_le_1, nt_le_2, nt_le_4,
  // nt_le_8, nt_le_16, nt_le_32, nt_le_64, nt_gt_64.
  std::vector<std::pair<std::string, int64_t>> get_counters() const;
  void reset_counters();

 private:
  // Static dispatchers used by cudaLaunchHostFunc. Both must be
  // `void(*)(void*)`.
  static void DispatchCallback(void* user_data);
  static void SyncCallback(void* user_data);

  // §1c.29 commit 2 review-fix — internal launcher that does NOT
  // call check_error(). Used by `sync_or_wait_on_stream` so a
  // prior worker error cannot prevent the captured wait kernel
  // from being recorded/launched (which would leave the stream
  // wedged with no done_slot consumer). Errors are still surfaced
  // by check_error() at the next Python-side entry point that is
  // safe to short-circuit (submit_on_stream, sync_blocking, etc.).
  void wait_kernel_sync_on_stream_no_check(int64_t task_id,
                                           uintptr_t cuda_stream);

  // Worker-thread task body; runs whatever op_kind says. The `seq`
  // parameter (0 == no wait-kernel sync, > 0 == wait-kernel sync enabled for
  // this dispatch) controls whether the worker publishes `done_slot = seq`
  // after the task completes — see §1c.29 commit 2. The publish is in a
  // finally-style block so a worker exception still releases the
  // captured wait kernel (otherwise `cots_wait_done_kernel` would spin
  // forever on `done < req` and the GPU stream would deadlock,
  // hiding the error from Python).
  void RunSlabOnWorker(TaskSlab* slab, uint32_t seq);

  std::unique_ptr<TaskQueue> task_queue_;

  // Heap-allocated, sized-once raw array. Address-stable for the lifetime of
  // *this — captured CUDA graphs record &slabs_[id] as host-callback userData
  // and re-replay must see the same address. We deliberately do NOT use
  // std::vector<TaskSlab> because std::atomic<int32_t> num_tokens makes
  // TaskSlab non-MoveConstructible, which std::vector's template machinery
  // (eagerly instantiated even for reserve-only flows) rejects.
  std::unique_ptr<TaskSlab[]> slabs_;
  int64_t slab_count_ = 0;

  // Upper bound on per-call num_tokens, set at install. Gates the
  // pinned-buffer bounds check in submit_on_stream + RunSlabOnWorker
  // so a misconfigured slab can't read past x_pinned / y_pinned.
  int64_t max_num_tokens_ = 0;

  // MLP gate/up/SwiGLU scratch. Accessed only on the single TaskQueue worker
  // thread, so no locking is needed. Resized lazily to the active live-token
  // and intermediate shape, then reused across MLP tasks.
  std::vector<uint16_t> mlp_scratch_bf16_;

  // Stable userData for sync_on_stream's cudaLaunchHostFunc — must be
  // a member, NOT a stack/heap-per-call alloc. CUDA graph capture freezes
  // the userData pointer at capture time.
  SyncArgs sync_args_{};

  // Cached at::set_num_threads value to avoid redundant rebuilds of the
  // at-thread-pool on every task. Accessed only on the worker thread.
  int32_t worker_current_n_threads_ = 0;

  std::atomic<int32_t> last_observed_num_threads_{0};

  // §1c.21 live-token override. Sentinel 0 = unset; positive values
  // override slab->num_tokens for row-count arithmetic in the
  // worker. Updated OUT OF GRAPH by set_runtime_num_tokens() before
  // each captured replay; read on the worker thread via acquire
  // load. See header comment on set_runtime_num_tokens.
  std::atomic<int32_t> runtime_num_tokens_{0};

  // Worker exception surfacing (see has_error / take_error above).
  std::atomic<bool> has_error_{false};
  std::mutex error_mtx_;
  std::string last_error_msg_;

  // §1c.21 counters (see get_counters / reset_counters above). All
  // increments use memory_order_relaxed — observational only.
  std::atomic<int64_t> submit_count_qkv_{0};
  std::atomic<int64_t> submit_count_mlp_{0};
  std::atomic<int64_t> submit_count_dryrun_{0};
  // 8-bin power-of-2 histogram per op kind (qkv | mlp | dryrun).
  // Indices: 0=nt<=1, 1=nt<=2, 2=nt<=4, 3=nt<=8, 4=nt<=16, 5=nt<=32,
  // 6=nt<=64, 7=nt>64.
  std::array<std::atomic<int64_t>, 8> nt_hist_qkv_{};
  std::array<std::atomic<int64_t>, 8> nt_hist_mlp_{};
  std::array<std::atomic<int64_t>, 8> nt_hist_dryrun_{};
  // §1c.22 review-fix: existing counters renamed for accuracy. Under
  // CUDA Graph capture, `submit_on_stream` runs ONCE at graph
  // record time; subsequent replays re-fire the captured
  // cudaMemcpyAsync directly without re-entering the C++ method.
  // So these record the RECORD-time activity, NOT per-replay
  // activity. Compare against `*_replay_bucket_bytes_` (incremented
  // at replay-time from `RunSlabOnWorker`) for the actual per-call
  // PCIe traffic during decode.
  std::atomic<int64_t> d2h_1d_count_{0};  // record-time
  std::atomic<int64_t> d2h_2d_count_{0};  // record-time
  std::atomic<int64_t> d2h_record_bytes_1d_{
      0};  // record-time, was d2h_1d_bytes_
  std::atomic<int64_t> d2h_record_bytes_2d_{
      0};  // record-time, was d2h_2d_bytes_
  // §1c.22: replay-time D2H / UVA bucket-byte accounting,
  // incremented in RunSlabOnWorker (which executes per replay
  // because the captured host callback fires per replay). Counts
  //   slab.bucket_capacity_tokens × slab.in_dim  × sizeof(bfloat16)  for D2H
  //   slab.bucket_capacity_tokens × slab.cpu_out_dim × sizeof(bfloat16)  for
  //   UVA
  // — the bucket-sized captured cudaMemcpyAsync (input) and Triton
  // UVA grid (output) that re-fire at every replay. §1c.22
  // review-fix: `bucket_capacity_tokens` is the IMMUTABLE
  // descriptor bucket populated at install, NOT the mutable
  // `slab.num_tokens` (which can be overwritten by later
  // submit_on_stream calls). Apples-to-apples with the per-replay
  // `worker_*_live_bytes_` counters below.
  std::atomic<int64_t> d2h_replay_bucket_bytes_{0};
  std::atomic<int64_t> uva_replay_bucket_bytes_{0};

  // §1c.21 fix-validation counters: distinct from the submit-time
  // num_tokens histogram. `runtime_set_calls_` counts how often
  // `set_runtime_num_tokens` was called (proves the plumb-through
  // is reaching C++); `runtime_last_value_` is the most recent
  // value (proves what's being pushed); `worker_effective_n_hist_`
  // bins what the worker actually used after the
  // `effective_n = override > 0 ? override : slab.num_tokens`
  // resolution. If submit-time histogram is dominated by `nt_gt_64`
  // but `worker_effective_n_hist` shows mostly `nt_le_1`, the
  // worker is correctly using the override.
  std::atomic<int64_t> runtime_set_calls_{0};
  std::atomic<int64_t> runtime_last_value_{0};
  std::array<std::atomic<int64_t>, 8> worker_effective_n_hist_{};

  // §1c.22 byte accounting — bucket-sized captured copies vs
  // live-token worker reads/writes. The captured cudaMemcpyAsync
  // (input D2H) and Triton UVA (output H2D) walk the full slab
  // bucket capacity, so their byte counts are tracked under
  // `d2h_replay_bucket_bytes_` / `uva_replay_bucket_bytes_` above
  // (sized at `slab.bucket_capacity_tokens`, the immutable
  // descriptor bucket). The "live" totals below are summed at
  // worker fire time using the effective_n the worker actually
  // processed (effective_n × in_dim × bf16 for input; effective_n
  // × cpu_out_dim × bf16 for output). The diff
  // (replay_bucket_bytes − live_bytes) is the wasted PCIe / UVA
  // bandwidth at B=1 decode under FULL capture; if material,
  // motivates §1c.23's live-masked transfer prototype.
  std::atomic<int64_t> worker_input_live_bytes_{0};
  std::atomic<int64_t> worker_output_live_bytes_{0};
  // §1c.22: bucket-sized UVA H2D transfer requested at GRAPH
  // RECORD time (captured Triton grid sized for bucket ×
  // cpu_out_dim). Bumped by `note_uva_request(num_tokens,
  // cpu_out_dim)` from `cots_sync_then_uva`'s Python impl, which
  // runs once during graph capture, NOT at replay. For per-replay
  // UVA bytes use `uva_replay_bucket_bytes_` above.
  std::atomic<int64_t> uva_record_bytes_{0};
  std::atomic<int64_t> uva_record_count_{0};

  // §1c.24 diagnostic attribution counters. Wall-clock totals
  // (steady_clock ns) and invocation counts that let the bench
  // summary split a generate's COTS time into worker compute,
  // queue wait, sync wait, and dispatch_cb fires — without nsys.
  // Useful as a quick check when nsys is unavailable; nsys NVTX
  // markers (also gated by VLLM_COTS_DIAG=1) are the primary tool
  // for full timeline attribution. All increments are gated on
  // `nvtx_internal::diag_enabled()` in the .cpp so this is purely
  // diagnostic — the production-default hot path skips the
  // timestamp + atomic-add work entirely.
  std::atomic<int64_t> dispatch_cb_count_{0};
  std::atomic<int64_t> sync_cb_count_{0};
  std::atomic<int64_t> sync_cb_wait_total_ns_{0};
  std::atomic<int64_t> worker_run_count_{0};
  std::atomic<int64_t> worker_busy_total_ns_{0};
  // (worker_start - dispatch_cb_enqueue_time) per task. Time the
  // task spent waiting in the TaskQueue cv before the worker picked
  // it up. Distinct from `sync_cb_wait_total_ns_` (the driver
  // thread's wait inside SyncCallback).
  std::atomic<int64_t> worker_queue_wait_total_ns_{0};
  // §1c.31: counts replays where `runtime_num_tokens > slab.num_tokens`
  // — the global live-token override was set for a larger row count
  // than this slab's bucket can hold. The worker clamps to slab
  // capacity rather than reading past the pinned buffer's tail. Most
  // common under eager mode where the global override applies to
  // whatever slab fires next, regardless of which bucket sized it.
  // Surfaced via get_counters(); non-zero means at least one
  // submit/replay was clamped (informational, not an error).
  std::atomic<int64_t> worker_clamp_override_count_{0};

  // §1c.29 wait-kernel sync diag counters. Allocated once at first
  // install_wait_kernel_sync_for_task call (lazily — most instances don't
  // use wait-kernel sync) as host-mapped pinned int64_t cells so the GPU
  // cots_wait_done_kernel_diag can atomicAdd directly. host_ptr is
  // CPU-readable for get_counters/reset_counters; dev_ptr is
  // GPU-visible for the kernel. Counters meaning:
  //   immediate_resume: wait fires that found done >= req on
  //     the first read (GPU window covered CPU work).
  //   lagging_wait: wait fires that had to spin at all.
  //   spin_iters_total: sum of spin iterations across all
  //     lagging waits.
  // Together they tell us how often the prototype's wait
  // actually serializes vs returns immediately — the §1c.29
  // canary for keeping SM occupancy small.
  int64_t* wait_kernel_immediate_resume_host_{nullptr};
  int64_t* wait_kernel_immediate_resume_dev_{nullptr};
  int64_t* wait_kernel_lagging_wait_host_{nullptr};
  int64_t* wait_kernel_lagging_wait_dev_{nullptr};
  int64_t* wait_kernel_spin_iters_host_{nullptr};
  int64_t* wait_kernel_spin_iters_dev_{nullptr};
};

}  // namespace cots
}  // namespace vllm

#endif  // VLLM_COTS_CPU_INFER_H_
