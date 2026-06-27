# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

_cots_C = pytest.importorskip("vllm._cots_C")
if not hasattr(_cots_C, "gqa_bf16_decode_attention"):
    pytest.skip(
        "COTS CPU GQA attention kernels are not built; rebuild vLLM",
        allow_module_level=True,
    )

from vllm.config.offload import CotsOffloadConfig  # noqa: E402
from vllm.model_executor.offloader.cots_offloader import (  # noqa: E402
    CotsHeadSplitQKVSidecar,
    CotsOffloader,
)
from vllm.model_executor.offloader.cots_storage import (  # noqa: E402
    QKV_ROLE,
    WO_INPUT_ROLE,
)
from vllm.v1.attention.backends.cots_head_split_attention import (  # noqa: E402
    CotsHeadSplitAttentionMetadata,
    cots_head_split_decode_attention,
    cots_head_split_kv_cache_update,
    cots_head_split_prefill_attention,
)
from vllm.v1.worker.cots_head_split_kv import CotsHeadSplitKVStore  # noqa: E402


def _canonical_group_indices(
    group: int,
    *,
    q_group: int,
    head_dim: int,
    q_size: int,
    kv_size: int,
) -> torch.Tensor:
    return torch.cat(
        [
            torch.arange(group * q_group, (group + 1) * q_group),
            torch.arange(q_size + group * head_dim, q_size + (group + 1) * head_dim),
            torch.arange(
                q_size + kv_size + group * head_dim,
                q_size + kv_size + (group + 1) * head_dim,
            ),
        ]
    ).long()


def _make_qkv_route_handle(*, cpu_weight_groups: int) -> SimpleNamespace:
    num_groups = 4
    q_heads_per_kv = 2
    head_dim = 2
    q_group = q_heads_per_kv * head_dim
    qkv_group = q_group + 2 * head_dim
    q_size = num_groups * q_group
    kv_size = num_groups * head_dim
    out_dim = q_size + 2 * kv_size
    cpu_compute_start = num_groups - cpu_weight_groups
    cpu_groups = range(cpu_compute_start, num_groups)
    if cpu_weight_groups > 0:
        cpu_idx = torch.cat(
            [
                _canonical_group_indices(
                    group,
                    q_group=q_group,
                    head_dim=head_dim,
                    q_size=q_size,
                    kv_size=kv_size,
                )
                for group in cpu_groups
            ]
        )
    else:
        cpu_idx = torch.empty(0, dtype=torch.long)
    gpu_idx = torch.tensor(
        [i for i in range(out_dim) if i not in set(cpu_idx.tolist())],
        dtype=torch.long,
    )
    return SimpleNamespace(
        role=QKV_ROLE,
        qkv_cpu_layout="gqa_group",
        qualified_name="test.qkv_proj",
        num_kv_heads=num_groups,
        head_dim=head_dim,
        q_size=q_size,
        kv_size=kv_size,
        out_dim=out_dim,
        dtype=torch.bfloat16,
        n_cpu=cpu_weight_groups * qkv_group,
        n_cpu_compute_by_bucket={1: cpu_weight_groups * qkv_group},
        gqa_q_group_size=q_group,
        gqa_qkv_group_size=qkv_group,
        gpu_indices_cuda=gpu_idx,
    )


def _make_wo_route_handle(*, cpu_weight_groups: int) -> SimpleNamespace:
    q_heads_per_kv = 2
    head_dim = 2
    q_group = q_heads_per_kv * head_dim
    return SimpleNamespace(
        role=WO_INPUT_ROLE,
        gqa_q_group_size=q_group,
        n_cpu=cpu_weight_groups * q_group,
        n_cpu_compute_by_bucket={1: cpu_weight_groups * q_group},
    )


def test_cots_head_split_qkv_route_assembles_cpu_plane_for_c_less_than_a() -> None:
    offloader = CotsOffloader(
        CotsOffloadConfig(kv_mode="head_split", f_cpu_kv_store=0.5)
    )
    handle = _make_qkv_route_handle(cpu_weight_groups=1)
    num_tokens = 2
    q_group = handle.gqa_q_group_size
    head_dim = handle.head_dim
    qkv_group = handle.gqa_qkv_group_size
    q_size = handle.q_size
    kv_size = handle.kv_size

    pref_idx = _canonical_group_indices(
        2, q_group=q_group, head_dim=head_dim, q_size=q_size, kv_size=kv_size
    )
    out_pref = torch.arange(num_tokens * qkv_group, dtype=torch.bfloat16).view(
        num_tokens, qkv_group
    )
    out_perm = torch.zeros(
        (num_tokens, handle.gpu_indices_cuda.numel()), dtype=torch.bfloat16
    )
    out_cpu = torch.full((num_tokens, qkv_group), 7.0, dtype=torch.bfloat16)

    out = offloader.route_head_split_qkv_output(
        handle=handle,
        bucket=1,
        num_tokens=num_tokens,
        reference=out_perm,
        out_perm=out_perm,
        out_pref=out_pref,
        out_cpu=out_cpu,
        pref_idx=pref_idx,
        bias=None,
    )
    sidecar = offloader.lookup_head_split_qkv_sidecar(out)
    assert sidecar is not None
    assert sidecar.cpu_attention_groups == 2
    assert sidecar.cpu_weight_groups == 1
    assert torch.equal(
        sidecar.query[:, :2, :].reshape(num_tokens, q_group),
        out_pref[:, :q_group],
    )
    assert torch.equal(
        sidecar.key[:, :1, :].reshape(num_tokens, head_dim),
        out_pref[:, q_group : q_group + head_dim],
    )
    assert torch.equal(
        sidecar.value[:, :1, :].reshape(num_tokens, head_dim),
        out_pref[:, q_group + head_dim : qkv_group],
    )
    assert torch.equal(
        sidecar.value[:, 1:2, :].reshape(num_tokens, head_dim),
        out_cpu[:, q_group + head_dim : qkv_group],
    )


def test_cots_head_split_qkv_route_copies_c_greater_than_a_mismatch() -> None:
    offloader = CotsOffloader(
        CotsOffloadConfig(kv_mode="head_split", f_cpu_kv_store=0.25)
    )
    handle = _make_qkv_route_handle(cpu_weight_groups=2)
    num_tokens = 1
    q_group = handle.gqa_q_group_size
    head_dim = handle.head_dim
    qkv_group = handle.gqa_qkv_group_size
    q_size = handle.q_size
    kv_size = handle.kv_size
    out_perm = torch.zeros(
        (num_tokens, handle.gpu_indices_cuda.numel()), dtype=torch.bfloat16
    )
    out_cpu = torch.arange(2 * qkv_group, dtype=torch.bfloat16).view(1, -1)

    out = offloader.route_head_split_qkv_output(
        handle=handle,
        bucket=1,
        num_tokens=num_tokens,
        reference=out_perm,
        out_perm=out_perm,
        out_pref=None,
        out_cpu=out_cpu,
        pref_idx=torch.empty(0, dtype=torch.long),
        bias=None,
    )
    sidecar = offloader.lookup_head_split_qkv_sidecar(out)
    assert sidecar is not None
    assert sidecar.cpu_attention_groups == 1
    assert sidecar.cpu_weight_groups == 2
    group2 = _canonical_group_indices(
        2, q_group=q_group, head_dim=head_dim, q_size=q_size, kv_size=kv_size
    )
    assert torch.equal(out[:, group2], out_cpu[:, :qkv_group])
    assert torch.equal(
        sidecar.query.reshape(num_tokens, q_group),
        out_cpu[:, qkv_group : qkv_group + q_group],
    )


def test_cots_head_split_qkv_route_reuses_cpu_work_buffers() -> None:
    offloader = CotsOffloader(
        CotsOffloadConfig(kv_mode="head_split", f_cpu_kv_store=0.25)
    )
    handle = _make_qkv_route_handle(cpu_weight_groups=1)
    num_tokens = 1
    qkv_group = handle.gqa_qkv_group_size
    out_perm = torch.zeros(
        (num_tokens, handle.gpu_indices_cuda.numel()), dtype=torch.bfloat16
    )

    first_out = offloader.route_head_split_qkv_output(
        handle=handle,
        bucket=1,
        num_tokens=num_tokens,
        reference=out_perm,
        out_perm=out_perm,
        out_pref=None,
        out_cpu=torch.full((num_tokens, qkv_group), 3.0, dtype=torch.bfloat16),
        pref_idx=torch.empty(0, dtype=torch.long),
        bias=None,
    )
    first_sidecar = offloader.lookup_head_split_qkv_sidecar(first_out)
    assert first_sidecar is not None
    first_q_ptr = first_sidecar.query.untyped_storage().data_ptr()
    first_k_ptr = first_sidecar.key.untyped_storage().data_ptr()
    first_v_ptr = first_sidecar.value.untyped_storage().data_ptr()

    second_out = offloader.route_head_split_qkv_output(
        handle=handle,
        bucket=1,
        num_tokens=num_tokens,
        reference=out_perm,
        out_perm=out_perm,
        out_pref=None,
        out_cpu=torch.full((num_tokens, qkv_group), 5.0, dtype=torch.bfloat16),
        pref_idx=torch.empty(0, dtype=torch.long),
        bias=None,
    )
    second_sidecar = offloader.lookup_head_split_qkv_sidecar(second_out)
    assert second_sidecar is not None
    assert second_sidecar.query.untyped_storage().data_ptr() == first_q_ptr
    assert second_sidecar.key.untyped_storage().data_ptr() == first_k_ptr
    assert second_sidecar.value.untyped_storage().data_ptr() == first_v_ptr
    assert torch.equal(
        second_sidecar.query.reshape(num_tokens, handle.gqa_q_group_size),
        torch.full((num_tokens, handle.gqa_q_group_size), 5.0, dtype=torch.bfloat16),
    )


def test_cots_head_split_cpu_rope_uses_published_cpu_positions() -> None:
    offloader = CotsOffloader(
        CotsOffloadConfig(kv_mode="head_split", f_cpu_kv_store=0.25)
    )
    handle = _make_qkv_route_handle(cpu_weight_groups=1)
    num_tokens = 1
    qkv_group = handle.gqa_qkv_group_size
    out_perm = torch.zeros(
        (num_tokens, handle.gpu_indices_cuda.numel()), dtype=torch.bfloat16
    )
    out_cpu = torch.arange(qkv_group, dtype=torch.bfloat16).view(1, -1)

    qkv_out = offloader.route_head_split_qkv_output(
        handle=handle,
        bucket=1,
        num_tokens=num_tokens,
        reference=out_perm,
        out_perm=out_perm,
        out_pref=None,
        out_cpu=out_cpu,
        pref_idx=torch.empty(0, dtype=torch.long),
        bias=None,
    )
    sidecar = offloader.lookup_head_split_qkv_sidecar(qkv_out)
    assert sidecar is not None
    offloader.set_head_split_cpu_positions(torch.tensor([0]), num_tokens)
    offloader.maybe_apply_head_split_cpu_rope(
        positions=torch.empty(num_tokens, dtype=torch.long, device="meta"),
        query=qkv_out,
        key=qkv_out,
        head_size=handle.head_dim,
        rotary_dim=handle.head_dim,
        cos_sin_cache=torch.tensor([[1.0, 0.0]], dtype=torch.bfloat16),
        is_neox_style=True,
    )

    assert sidecar.rope_applied
    assert torch.equal(
        sidecar.query.reshape(num_tokens, handle.gqa_q_group_size),
        out_cpu[:, : handle.gqa_q_group_size],
    )


def test_cots_head_split_qkv_route_uses_live_rows_for_padded_capacity() -> None:
    offloader = CotsOffloader(
        CotsOffloadConfig(kv_mode="head_split", f_cpu_kv_store=0.25)
    )
    handle = _make_qkv_route_handle(cpu_weight_groups=1)
    num_tokens_capacity = 4
    num_live_tokens = 1
    qkv_group = handle.gqa_qkv_group_size
    offloader.set_live_num_tokens(num_live_tokens)

    out_perm = torch.zeros(
        (num_tokens_capacity, handle.gpu_indices_cuda.numel()),
        dtype=torch.bfloat16,
    )
    out_cpu = torch.arange(qkv_group, dtype=torch.bfloat16).view(
        num_live_tokens, qkv_group
    )

    qkv_out = offloader.route_head_split_qkv_output(
        handle=handle,
        bucket=num_tokens_capacity,
        num_tokens=num_tokens_capacity,
        reference=out_perm,
        out_perm=out_perm,
        out_pref=None,
        out_cpu=out_cpu,
        pref_idx=torch.empty(0, dtype=torch.long),
        bias=None,
    )
    sidecar = offloader.lookup_head_split_qkv_sidecar(qkv_out)
    assert sidecar is not None
    assert qkv_out.shape == (num_tokens_capacity, handle.out_dim)
    assert sidecar.num_live_tokens == num_live_tokens
    assert sidecar.query.shape[0] == num_live_tokens
    assert torch.equal(
        sidecar.query.reshape(num_live_tokens, handle.gqa_q_group_size),
        out_cpu[:, : handle.gqa_q_group_size],
    )

    offloader.set_head_split_cpu_positions(torch.tensor([0]), num_live_tokens)
    offloader.maybe_apply_head_split_cpu_rope(
        positions=torch.empty(num_tokens_capacity, dtype=torch.long, device="meta"),
        query=qkv_out,
        key=qkv_out,
        head_size=handle.head_dim,
        rotary_dim=handle.head_dim,
        cos_sin_cache=torch.tensor([[1.0, 0.0]], dtype=torch.bfloat16),
        is_neox_style=True,
    )
    assert sidecar.rope_applied


def test_cots_head_split_wo_route_fills_preallocated_cpu_input() -> None:
    offloader = CotsOffloader(
        CotsOffloadConfig(kv_mode="head_split", f_cpu_kv_store=0.25)
    )
    num_tokens_capacity = 4
    num_live_tokens = 1
    num_groups = 4
    q_heads_per_kv = 2
    head_dim = 2
    q_group = q_heads_per_kv * head_dim
    handle = _make_wo_route_handle(cpu_weight_groups=2)
    x = torch.arange(
        num_tokens_capacity * num_groups * q_group,
        dtype=torch.bfloat16,
    ).view(num_tokens_capacity, num_groups * q_group)
    output_cpu = torch.full(
        (num_live_tokens, q_heads_per_kv, head_dim),
        9.0,
        dtype=torch.bfloat16,
    )
    qkv_sidecar = CotsHeadSplitQKVSidecar(
        storage_key=123,
        num_live_tokens=num_live_tokens,
        num_groups=num_groups,
        cpu_attention_groups=1,
        cpu_weight_groups=2,
        q_heads_per_kv=q_heads_per_kv,
        head_dim=head_dim,
        query=torch.empty(num_live_tokens, q_heads_per_kv, head_dim),
        key=torch.empty(num_live_tokens, 1, head_dim),
        value=torch.empty(num_live_tokens, 1, head_dim),
    )
    offloader.register_head_split_attention_output(
        output=x,
        qkv_sidecar=qkv_sidecar,
        output_cpu=output_cpu,
    )

    dst = torch.full(
        (num_tokens_capacity, 2 * q_group),
        -1.0,
        dtype=torch.bfloat16,
    )
    routed = offloader.build_head_split_wo_cpu_input(
        x=x,
        handle=handle,
        bucket=1,
        num_tokens=num_tokens_capacity,
        out=dst,
    )

    assert routed is dst
    assert torch.equal(
        dst[:num_live_tokens, :q_group],
        x[:num_live_tokens, 2 * q_group : 3 * q_group],
    )
    assert torch.equal(
        dst[:num_live_tokens, q_group:],
        output_cpu.reshape(num_live_tokens, q_group),
    )
    assert torch.equal(
        dst[num_live_tokens:],
        torch.full_like(dst[num_live_tokens:], -1.0),
    )
    assert offloader.lookup_head_split_attention_output(x) is None


def test_cots_head_split_a_zero_uses_gpu_attention_sidecar_for_wo() -> None:
    offloader = CotsOffloader(
        CotsOffloadConfig(kv_mode="head_split", f_cpu_kv_store=0.0)
    )
    assert offloader.head_split_activation_routing_enabled
    handle = _make_qkv_route_handle(cpu_weight_groups=1)
    num_tokens = 1
    q_group = handle.gqa_q_group_size
    head_dim = handle.head_dim
    qkv_group = handle.gqa_qkv_group_size
    q_size = handle.q_size
    kv_size = handle.kv_size
    out_perm = torch.zeros(
        (num_tokens, handle.gpu_indices_cuda.numel()), dtype=torch.bfloat16
    )
    out_cpu = torch.arange(qkv_group, dtype=torch.bfloat16).view(1, -1)

    qkv_out = offloader.route_head_split_qkv_output(
        handle=handle,
        bucket=1,
        num_tokens=num_tokens,
        reference=out_perm,
        out_perm=out_perm,
        out_pref=None,
        out_cpu=out_cpu,
        pref_idx=torch.empty(0, dtype=torch.long),
        bias=None,
    )
    sidecar = offloader.lookup_head_split_qkv_sidecar(qkv_out)
    assert sidecar is not None
    assert sidecar.cpu_attention_groups == 0
    assert sidecar.cpu_weight_groups == 1
    group3 = _canonical_group_indices(
        3, q_group=q_group, head_dim=head_dim, q_size=q_size, kv_size=kv_size
    )
    assert torch.equal(qkv_out[:, group3], out_cpu)

    attn_out = torch.arange(4 * q_group, dtype=torch.bfloat16).view(1, 4 * q_group)
    offloader.register_head_split_gpu_attention_output(
        output=attn_out,
        query=qkv_out,
        num_tokens=num_tokens,
    )
    assert offloader.lookup_head_split_qkv_sidecar(qkv_out) is None

    dst = torch.empty(num_tokens, q_group, dtype=torch.bfloat16)
    routed = offloader.build_head_split_wo_cpu_input(
        x=attn_out,
        handle=_make_wo_route_handle(cpu_weight_groups=1),
        bucket=1,
        num_tokens=num_tokens,
        out=dst,
    )
    assert routed is dst
    assert torch.equal(dst, attn_out[:, 3 * q_group : 4 * q_group])
    assert offloader.lookup_head_split_attention_output(attn_out) is None


def _make_cpu_kv_update_metadata(
    *,
    routed_key_cpu: torch.Tensor | None = None,
    routed_value_cpu: torch.Tensor | None = None,
) -> CotsHeadSplitAttentionMetadata:
    return CotsHeadSplitAttentionMetadata(
        cpu_key_cache=torch.zeros(2, 1, 4, 128, dtype=torch.bfloat16),
        cpu_value_cache=torch.zeros(2, 1, 4, 128, dtype=torch.bfloat16),
        cpu_block_table=torch.empty(1, 2, dtype=torch.int32),
        cpu_slot_mapping=torch.tensor([0, -1, 5], dtype=torch.long),
        cpu_seq_lens=torch.empty(1, dtype=torch.int32),
        gpu_kv_heads=3,
        cpu_kv_heads=1,
        q_heads_per_kv=7,
        cpu_query_start=21,
        cpu_query_heads=7,
        is_decode=False,
        num_actual_tokens=3,
        routed_key_cpu=routed_key_cpu,
        routed_value_cpu=routed_value_cpu,
    )


def test_cots_head_split_kv_update_skips_invalid_cpu_slots() -> None:
    metadata = _make_cpu_kv_update_metadata()
    key = torch.zeros(3, 4, 128, dtype=torch.bfloat16)
    value = torch.zeros_like(key)
    key[0, 3, :].fill_(1.0)
    key[1, 3, :].fill_(2.0)
    key[2, 3, :].fill_(3.0)
    value[0, 3, :].fill_(11.0)
    value[1, 3, :].fill_(12.0)
    value[2, 3, :].fill_(13.0)

    cots_head_split_kv_cache_update(
        key,
        value,
        torch.empty(3, dtype=torch.long, device="meta"),
        metadata,
    )

    assert torch.equal(metadata.cpu_key_cache[0, 0, 0], key[0, 3])
    assert torch.equal(metadata.cpu_key_cache[1, 0, 1], key[2, 3])
    assert torch.equal(metadata.cpu_value_cache[0, 0, 0], value[0, 3])
    assert torch.equal(metadata.cpu_value_cache[1, 0, 1], value[2, 3])
    assert torch.count_nonzero(metadata.cpu_key_cache[0, 0, 1]) == 0
    assert torch.count_nonzero(metadata.cpu_value_cache[0, 0, 1]) == 0


def test_cots_head_split_kv_update_masks_routed_cpu_kv_sidecar() -> None:
    routed_key_cpu = torch.zeros(3, 1, 128, dtype=torch.bfloat16)
    routed_value_cpu = torch.zeros_like(routed_key_cpu)
    routed_key_cpu[0, 0, :].fill_(21.0)
    routed_key_cpu[1, 0, :].fill_(22.0)
    routed_key_cpu[2, 0, :].fill_(23.0)
    routed_value_cpu[0, 0, :].fill_(31.0)
    routed_value_cpu[1, 0, :].fill_(32.0)
    routed_value_cpu[2, 0, :].fill_(33.0)
    metadata = _make_cpu_kv_update_metadata(
        routed_key_cpu=routed_key_cpu,
        routed_value_cpu=routed_value_cpu,
    )

    cots_head_split_kv_cache_update(
        torch.empty(3, 4, 128, dtype=torch.bfloat16),
        torch.empty(3, 4, 128, dtype=torch.bfloat16),
        torch.empty(3, dtype=torch.long, device="meta"),
        metadata,
    )

    assert torch.equal(metadata.cpu_key_cache[0, 0, 0], routed_key_cpu[0, 0])
    assert torch.equal(metadata.cpu_key_cache[1, 0, 1], routed_key_cpu[2, 0])
    assert torch.equal(metadata.cpu_value_cache[0, 0, 0], routed_value_cpu[0, 0])
    assert torch.equal(metadata.cpu_value_cache[1, 0, 1], routed_value_cpu[2, 0])
    assert torch.count_nonzero(metadata.cpu_key_cache[0, 0, 1]) == 0
    assert torch.count_nonzero(metadata.cpu_value_cache[0, 0, 1]) == 0


def test_cots_head_split_store_updates_cpu_heads_and_decodes() -> None:
    store = CotsHeadSplitKVStore(
        layer_names=["layer.0.attn"],
        num_blocks=4,
        block_size=4,
        num_kv_heads=4,
        num_query_heads=28,
        head_size=128,
        dtype=torch.bfloat16,
        f_cpu_kv_store=0.25,
        max_num_reqs=2,
        max_num_tokens=16,
        max_model_len=16,
        pin_memory=False,
    )

    assert store.gpu_kv_heads == 3
    assert store.cpu_kv_heads == 1
    assert store.cpu_query_start == 21
    assert store.cpu_query_heads == 7

    metadata = store.build_metadata(
        layer_name="layer.0.attn",
        block_table_cpu=torch.tensor([[0, 1, 2, 3]], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([1], dtype=torch.int32),
        is_prefilling_cpu=torch.tensor([False]),
        query_start_loc_cpu=torch.tensor([0, 1], dtype=torch.int32),
        positions_cpu=torch.tensor([0], dtype=torch.long),
        max_query_len=1,
        num_actual_tokens=1,
        num_reqs=1,
    )
    assert metadata.cpu_slot_mapping.tolist() == [0]

    key = torch.randn(3, 4, 128, dtype=torch.bfloat16)
    value = torch.randn(3, 4, 128, dtype=torch.bfloat16)
    cots_head_split_kv_cache_update(
        key,
        value,
        torch.empty(3, dtype=torch.long, device="meta"),
        metadata,
    )

    assert torch.equal(metadata.cpu_key_cache[0, 0, 0], key[0, 3])
    assert torch.equal(metadata.cpu_value_cache[0, 0, 0], value[0, 3])

    query = torch.randn(1, 28, 128, dtype=torch.bfloat16)
    output = torch.empty_like(query)
    cots_head_split_decode_attention(
        output,
        query,
        metadata,
        softmax_scale=1.0 / (128**0.5),
    )

    cpu_output = output[:, metadata.cpu_query_start :]
    assert cpu_output.shape == (1, 7, 128)
    assert torch.isfinite(cpu_output).all()


def test_cots_head_split_prefill_metadata_updates_cache_but_skips_cpu_decode() -> None:
    store = CotsHeadSplitKVStore(
        layer_names=["layer.0.attn"],
        num_blocks=4,
        block_size=4,
        num_kv_heads=4,
        num_query_heads=28,
        head_size=128,
        dtype=torch.bfloat16,
        f_cpu_kv_store=0.25,
        max_num_reqs=2,
        max_num_tokens=16,
        max_model_len=16,
        pin_memory=False,
    )

    metadata = store.build_metadata(
        layer_name="layer.0.attn",
        block_table_cpu=torch.tensor([[0, 1, 2, 3]], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([4], dtype=torch.int32),
        is_prefilling_cpu=torch.tensor([True]),
        query_start_loc_cpu=torch.tensor([0, 4], dtype=torch.int32),
        positions_cpu=torch.tensor([0, 1, 2, 3], dtype=torch.long),
        max_query_len=4,
        num_actual_tokens=4,
        num_reqs=1,
    )

    assert metadata.is_decode is False
    assert metadata.prefill_query_to_seq_cpu is not None
    assert metadata.prefill_seq_lens_cpu is not None
    assert metadata.cpu_slot_mapping.tolist() == [0, 1, 2, 3]
    assert metadata.prefill_query_to_seq_cpu.tolist() == [0, 0, 0, 0]
    assert metadata.prefill_seq_lens_cpu.tolist() == [1, 2, 3, 4]
    key = torch.randn(4, 4, 128, dtype=torch.bfloat16)
    value = torch.randn(4, 4, 128, dtype=torch.bfloat16)
    cots_head_split_kv_cache_update(
        key,
        value,
        torch.empty(4, dtype=torch.long, device="meta"),
        metadata,
    )
    assert torch.equal(metadata.cpu_key_cache[0, 0, 3], key[3, 3])

    query = torch.randn(4, 28, 128, dtype=torch.bfloat16)
    output = torch.empty_like(query)
    cots_head_split_prefill_attention(
        output,
        query,
        metadata,
        softmax_scale=1.0 / (128**0.5),
    )
    assert torch.isfinite(output[:, metadata.cpu_query_start :]).all()
