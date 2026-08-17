// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// pybind11 bindings for the Hybrid native weight task runner.
// No torch.ops.* registration in C++ - the torch.ops.vllm.hybrid_* ops are
// registered Python-side in vllm/model_executor/offloader/hybrid_ops.py via
// direct_register_custom_op.

#include <torch/extension.h>

#include "bf16_kernels.h"
#include "hybrid_weight_task_runner.h"

namespace py = pybind11;
using vllm::hybrid::HybridWeightTaskRunner;

PYBIND11_MODULE(_hybrid_C, m) {
  m.doc() = "Hybrid native weight task runner (vllm/csrc/hybrid/).";
  m.def("bf16_kernel_isa", &vllm::hybrid::bf16_kernel_isa);

  py::class_<HybridWeightTaskRunner>(m, "HybridWeightTaskRunner")
      .def(py::init<>())
      .def("install", &HybridWeightTaskRunner::install, py::arg("n_slabs"),
           py::arg("max_num_tokens"))
      .def("populate_slab_qkv", &HybridWeightTaskRunner::populate_slab_qkv,
           py::arg("task_id"), py::arg("n_threads"),
           py::arg("bucket_capacity_tokens"), py::arg("x_pinned_ptr"),
           py::arg("in_dim"), py::arg("y_pinned_ptr"), py::arg("cpu_out_dim"),
           py::arg("w_cpu_ptr"), py::arg("w_cpu_rows"))
      .def("populate_slab_mlp", &HybridWeightTaskRunner::populate_slab_mlp,
           py::arg("task_id"), py::arg("n_threads"),
           py::arg("bucket_capacity_tokens"), py::arg("x_pinned_ptr"),
           py::arg("in_dim"), py::arg("y_pinned_ptr"), py::arg("cpu_out_dim"),
           py::arg("w_gate_ptr"), py::arg("w_gate_rows"), py::arg("w_up_ptr"),
           py::arg("w_up_rows"), py::arg("w_down_ptr"), py::arg("w_down_rows"),
           py::arg("w_down_cols"))
      .def("populate_slab_dryrun",
           &HybridWeightTaskRunner::populate_slab_dryrun, py::arg("task_id"),
           py::arg("bucket_capacity_tokens"), py::arg("x_pinned_ptr"),
           py::arg("in_dim"), py::arg("y_pinned_ptr"), py::arg("cpu_out_dim"))
      .def("slab_bucket_capacity_tokens",
           &HybridWeightTaskRunner::slab_bucket_capacity_tokens,
           py::arg("task_id"))
      .def("slab_num_tokens", &HybridWeightTaskRunner::slab_num_tokens,
           py::arg("task_id"))
      .def("submit_on_stream", &HybridWeightTaskRunner::submit_on_stream,
           py::arg("task_id"), py::arg("num_tokens"), py::arg("x_gpu_ptr"),
           py::arg("x_cols"), py::arg("x_stride0"), py::arg("x_stride1"),
           py::arg("cuda_stream"))
      .def("sync_on_stream", &HybridWeightTaskRunner::sync_on_stream,
           py::arg("cuda_stream"))
      .def("sync_blocking", &HybridWeightTaskRunner::sync_blocking)
      .def("set_worker_affinity", &HybridWeightTaskRunner::set_worker_affinity,
           py::arg("cpu_set"))
      .def("last_observed_num_threads",
           &HybridWeightTaskRunner::last_observed_num_threads)
      .def("has_error", &HybridWeightTaskRunner::has_error)
      .def("take_error", &HybridWeightTaskRunner::take_error)
      .def("check_error", &HybridWeightTaskRunner::check_error)
      .def("run_at_linear_inline",
           &HybridWeightTaskRunner::run_at_linear_inline, py::arg("x"),
           py::arg("w"), py::arg("y_out"))
      // Runtime-dispatched BF16 GEMM kernel inline. Same signature shape as
      // run_at_linear_inline; w must be (K, N) row-major BF16 (the
      // transposed-storage layout rather than the natural Linear layout).
      .def("run_bf16_gemm_transposed_inline",
           &HybridWeightTaskRunner::run_bf16_gemm_transposed_inline,
           py::arg("x"), py::arg("w"), py::arg("y_out"))
      // Natural (N, K) BF16 GEMM kernel.
      .def("run_bf16_gemm_natural_inline",
           &HybridWeightTaskRunner::run_bf16_gemm_natural_inline, py::arg("x"),
           py::arg("w"), py::arg("y_out"))
      // Production fused MLP CPU kernel: gate/up + SwiGLU + down.
      .def("run_bf16_mlp_inline", &HybridWeightTaskRunner::run_bf16_mlp_inline,
           py::arg("x"), py::arg("w_gate"), py::arg("w_up"), py::arg("w_down"),
           py::arg("y_out"))
      .def("y_pinned_view", &HybridWeightTaskRunner::y_pinned_view,
           py::arg("task_id"), py::arg("num_tokens"))
      .def("publish_live_num_tokens_on_stream",
           &HybridWeightTaskRunner::publish_live_num_tokens_on_stream,
           py::arg("n"), py::arg("cuda_stream"))
      .def("note_uva_request", &HybridWeightTaskRunner::note_uva_request,
           py::arg("num_tokens"), py::arg("cpu_out_dim"))
      .def("get_counters",
           [](const HybridWeightTaskRunner& self) {
             // §1c.21: return as a Python dict for ergonomic
             // dump-and-print at the bench harness level.
             py::dict out;
             for (auto& [name, value] : self.get_counters()) {
               out[py::str(name)] = value;
             }
             return out;
           })
      .def("reset_counters", &HybridWeightTaskRunner::reset_counters);
}
