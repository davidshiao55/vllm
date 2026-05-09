// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// pybind11 bindings for the COTS native CPU runner. Exposes the
// `CotsCpuInfer` class only; no torch.ops.* registration in C++ —
// the torch.ops.vllm.cots_* ops are registered Python-side in
// vllm/model_executor/offloader/cots_ops.py via direct_register_custom_op.

#include <torch/extension.h>

#include "cots_cpu_infer.h"

namespace py = pybind11;
using vllm::cots::CotsCpuInfer;

PYBIND11_MODULE(_cots_C, m) {
  m.doc() = "COTS Phase 1c native CPU runner (vllm/csrc/cots/).";

  py::class_<CotsCpuInfer>(m, "CotsCpuInfer")
      .def(py::init<>())
      .def("install", &CotsCpuInfer::install, py::arg("n_slabs"),
           py::arg("scratch_max_tokens"),
           py::arg("scratch_max_intermediate_per_half"))
      .def("populate_slab_qkv", &CotsCpuInfer::populate_slab_qkv,
           py::arg("task_id"), py::arg("n_threads"),
           py::arg("bucket_capacity_tokens"), py::arg("x_pinned_ptr"),
           py::arg("in_dim"), py::arg("y_pinned_ptr"), py::arg("cpu_out_dim"),
           py::arg("w_cpu_ptr"), py::arg("w_cpu_rows"))
      .def("populate_slab_mlp", &CotsCpuInfer::populate_slab_mlp,
           py::arg("task_id"), py::arg("n_threads"),
           py::arg("bucket_capacity_tokens"), py::arg("x_pinned_ptr"),
           py::arg("in_dim"), py::arg("y_pinned_ptr"), py::arg("cpu_out_dim"),
           py::arg("w_gate_ptr"), py::arg("w_gate_rows"), py::arg("w_up_ptr"),
           py::arg("w_up_rows"), py::arg("w_down_ptr"), py::arg("w_down_rows"),
           py::arg("w_down_cols"), py::arg("w_down_stride_row"),
           py::arg("w_down_stride_col"), py::arg("intermediate_per_half"))
      .def("populate_slab_dryrun", &CotsCpuInfer::populate_slab_dryrun,
           py::arg("task_id"), py::arg("bucket_capacity_tokens"),
           py::arg("x_pinned_ptr"), py::arg("in_dim"), py::arg("y_pinned_ptr"),
           py::arg("cpu_out_dim"))
      .def("slab_bucket_capacity_tokens",
           &CotsCpuInfer::slab_bucket_capacity_tokens, py::arg("task_id"))
      .def("slab_num_tokens", &CotsCpuInfer::slab_num_tokens,
           py::arg("task_id"))
      .def("submit_on_stream", &CotsCpuInfer::submit_on_stream,
           py::arg("task_id"), py::arg("num_tokens"), py::arg("x_gpu_ptr"),
           py::arg("x_cols"), py::arg("x_stride0"), py::arg("x_stride1"),
           py::arg("cuda_stream"))
      .def("sync_on_stream", &CotsCpuInfer::sync_on_stream,
           py::arg("cuda_stream"))
      .def("submit_dryrun_burst", &CotsCpuInfer::submit_dryrun_burst,
           py::arg("n"))
      .def("sync_blocking", &CotsCpuInfer::sync_blocking)
      .def("set_worker_affinity", &CotsCpuInfer::set_worker_affinity,
           py::arg("cpu_set"))
      .def("last_observed_num_threads",
           &CotsCpuInfer::last_observed_num_threads)
      .def("has_error", &CotsCpuInfer::has_error)
      .def("take_error", &CotsCpuInfer::take_error)
      .def("check_error", &CotsCpuInfer::check_error)
      .def("run_at_linear_inline", &CotsCpuInfer::run_at_linear_inline,
           py::arg("x"), py::arg("w"), py::arg("y_out"))
      .def("y_pinned_view", &CotsCpuInfer::y_pinned_view, py::arg("task_id"),
           py::arg("num_tokens"))
      .def("set_runtime_num_tokens", &CotsCpuInfer::set_runtime_num_tokens,
           py::arg("n"))
      .def("note_uva_request", &CotsCpuInfer::note_uva_request,
           py::arg("num_tokens"), py::arg("cpu_out_dim"))
      .def("get_counters",
           [](const CotsCpuInfer& self) {
             // §1c.21: return as a Python dict for ergonomic
             // dump-and-print at the bench harness level.
             py::dict out;
             for (auto& [name, value] : self.get_counters()) {
               out[py::str(name)] = value;
             }
             return out;
           })
      .def("reset_counters", &CotsCpuInfer::reset_counters);
}
