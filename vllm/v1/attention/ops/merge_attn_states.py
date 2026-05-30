# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.platforms import current_platform


def _merge_attn_supported(output: torch.Tensor) -> bool:
    # NOTE(DefTruth): Currently, custom merge_attn_states CUDA kernel
    # does not support FP8 dtype, fallback to use Triton kernel.
    supported_dtype = output.dtype in [torch.float32, torch.half, torch.bfloat16]
    headdim = output.shape[2]
    if output.dtype == torch.float32:
        supported_headdim = headdim % 4 == 0
    else:
        supported_headdim = headdim % 8 == 0
    return current_platform.is_cuda() and supported_dtype and supported_headdim


def merge_attn_states(
    output: torch.Tensor,
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
    output_lse: torch.Tensor | None = None,
) -> None:
    if _merge_attn_supported(output):
        from vllm._custom_ops import merge_attn_states

        return merge_attn_states(
            output, prefix_output, prefix_lse, suffix_output, suffix_lse, output_lse
        )
    else:
        from vllm.v1.attention.ops.triton_merge_attn_states import merge_attn_states

        return merge_attn_states(
            output, prefix_output, prefix_lse, suffix_output, suffix_lse, output_lse
        )


def merge_attn_states_indexed(
    output: torch.Tensor,
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
    token_indices: torch.Tensor,
    output_lse: torch.Tensor | None = None,
) -> None:
    if (
        _merge_attn_supported(output)
        and token_indices.is_cuda
        and token_indices.dtype == torch.long
    ):
        from vllm._custom_ops import (
            merge_attn_states_indexed as merge_attn_states_indexed_cuda,
        )

        return merge_attn_states_indexed_cuda(
            output,
            prefix_output,
            prefix_lse,
            suffix_output,
            suffix_lse,
            token_indices,
            output_lse,
        )

    if token_indices.device != prefix_output.device:
        token_indices = token_indices.to(device=prefix_output.device, non_blocking=True)
    compact_prefix_output = prefix_output.index_select(0, token_indices)
    compact_prefix_lse = prefix_lse.index_select(1, token_indices)
    compact_output = torch.empty_like(suffix_output)
    compact_output_lse = (
        torch.empty_like(suffix_lse) if output_lse is not None else None
    )
    merge_attn_states(
        compact_output,
        compact_prefix_output,
        compact_prefix_lse,
        suffix_output,
        suffix_lse,
        compact_output_lse,
    )
    output.index_copy_(0, token_indices, compact_output)
    if output_lse is not None:
        assert compact_output_lse is not None
        output_lse.index_copy_(1, token_indices, compact_output_lse)
