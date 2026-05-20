# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom ops for COTS Phase 2 CPU suffix attention runners.

The prepared runner ops intentionally expose only CUDA tensors plus scalar
runner/task ids to torch.compile/CUDA graph capture. CPU tensor pointers live in
``CotsSuffixAttentionInfer`` tasks populated outside the custom-op signature,
matching the Phase 1c slab-pointer design.
"""

from __future__ import annotations

import itertools
from typing import Any

import torch

from vllm.utils.torch_utils import direct_register_custom_op

_COTS_SUFFIX_INFER: dict[int, Any] = {}
_NEXT_SUFFIX_RUNNER_ID = itertools.count(1)


def register_suffix_infer(infer: Any) -> int:
    runner_id = next(_NEXT_SUFFIX_RUNNER_ID)
    _COTS_SUFFIX_INFER[runner_id] = infer
    return runner_id


def unregister_suffix_infer(runner_id: int) -> None:
    _COTS_SUFFIX_INFER.pop(int(runner_id), None)


def lookup_suffix_infer(runner_id: int, op_name: str) -> Any:
    infer = _COTS_SUFFIX_INFER.get(int(runner_id))
    if infer is None:
        raise RuntimeError(
            f"{op_name}: suffix runner_id={int(runner_id)} is not registered "
            f"(known ids: {list(_COTS_SUFFIX_INFER.keys())})"
        )
    return infer


def install_suffix_infer(runner_id: int, n_tasks: int) -> None:
    infer = lookup_suffix_infer(runner_id, "install_suffix_infer")
    infer.install(int(n_tasks))


def install_suffix_wait_kernel_sync(runner_id: int, n_tasks: int) -> None:
    infer = lookup_suffix_infer(runner_id, "install_suffix_wait_kernel_sync")
    for task_id in range(int(n_tasks)):
        infer.install_wait_kernel_sync_for_task(int(task_id))


def suffix_wait_kernel_sync_installed(runner_id: int, task_id: int) -> bool:
    infer = lookup_suffix_infer(runner_id, "suffix_wait_kernel_sync_installed")
    return bool(infer.wait_kernel_sync_installed_for_task(int(task_id)))


def suffix_wait_kernel_slots(runner_id: int, task_id: int) -> tuple[int, int]:
    infer = lookup_suffix_infer(runner_id, "suffix_wait_kernel_slots")
    return (
        int(infer.wait_kernel_get_req_slot(int(task_id))),
        int(infer.wait_kernel_get_done_slot(int(task_id))),
    )


def get_suffix_counters(runner_id: int) -> dict[str, int]:
    infer = lookup_suffix_infer(runner_id, "get_suffix_counters")
    return {str(k): int(v) for k, v in dict(infer.get_counters()).items()}


def reset_suffix_counters(runner_id: int) -> None:
    infer = lookup_suffix_infer(runner_id, "reset_suffix_counters")
    infer.reset_counters()


def sync_suffix_infer_blocking(runner_id: int) -> None:
    infer = _COTS_SUFFIX_INFER.get(int(runner_id))
    if infer is None:
        return
    infer.sync_blocking()


def set_suffix_runtime_counts(
    runner_id: int, num_tokens: int, scatter_count: int
) -> None:
    infer = lookup_suffix_infer(runner_id, "set_suffix_runtime_counts")
    infer.set_runtime_counts(int(num_tokens), int(scatter_count))


def _cots_suffix_attn_submit_impl(
    anchor: torch.Tensor,
    runner_id: int,
    task_id: int,
) -> None:
    if not anchor.is_cuda:
        raise RuntimeError(
            f"cots_suffix_attn_submit: anchor must be CUDA, got {anchor.device}"
        )
    infer = lookup_suffix_infer(runner_id, "cots_suffix_attn_submit")
    stream = torch.cuda.current_stream(anchor.device).cuda_stream
    infer.submit_prepared_on_stream(int(task_id), stream)


def _cots_suffix_attn_submit_fake(
    anchor: torch.Tensor,
    runner_id: int,
    task_id: int,
) -> None:
    return


def _cots_suffix_attn_sync_impl(
    anchor: torch.Tensor,
    runner_id: int,
    task_id: int,
) -> None:
    if not anchor.is_cuda:
        raise RuntimeError(
            f"cots_suffix_attn_sync: anchor must be CUDA, got {anchor.device}"
        )
    infer = lookup_suffix_infer(runner_id, "cots_suffix_attn_sync")
    stream = torch.cuda.current_stream(anchor.device).cuda_stream
    infer.sync_or_wait_on_stream(int(task_id), stream)


def _cots_suffix_attn_sync_fake(
    anchor: torch.Tensor,
    runner_id: int,
    task_id: int,
) -> None:
    return


def register_cots_suffix_attention_ops() -> None:
    direct_register_custom_op(
        op_name="cots_suffix_attn_submit",
        op_func=_cots_suffix_attn_submit_impl,
        mutates_args=["anchor"],
        fake_impl=_cots_suffix_attn_submit_fake,
    )
    direct_register_custom_op(
        op_name="cots_suffix_attn_sync",
        op_func=_cots_suffix_attn_sync_impl,
        mutates_args=["anchor"],
        fake_impl=_cots_suffix_attn_sync_fake,
    )


register_cots_suffix_attention_ops()
