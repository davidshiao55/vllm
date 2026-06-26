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

from vllm.config import (  # noqa: E402
    CacheConfig,
    CotsOffloadConfig,
    OffloadConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.model_executor.offloader import (  # noqa: E402
    get_offloader,
    set_offloader,
)
from vllm.model_executor.offloader.cots_offloader import (  # noqa: E402
    CotsHeadSplitQKVSidecar,
    CotsOffloader,
)
from vllm.model_executor.offloader.cots_storage import (  # noqa: E402
    QKV_ROLE,
    WO_INPUT_ROLE,
    KVPrefetchBufferPool,
    KVPrefetchStreamer,
)
from vllm.v1.attention.backends.cots_head_split_attention import (  # noqa: E402
    CotsHeadSplitAttentionMetadata,
    _prepare_cpu_query,
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
        num_q_heads=num_groups * q_heads_per_kv,
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
        qualified_name="test.o_proj",
        num_kv_heads=4,
        gqa_q_group_size=q_group,
        n_cpu=cpu_weight_groups * q_group,
        n_cpu_compute_by_bucket={1: cpu_weight_groups * q_group},
    )


def _make_kv_prefetch_store(*, n_layers: int = 1) -> CotsHeadSplitKVStore:
    return CotsHeadSplitKVStore(
        layer_names=[f"layer.{idx}.attn" for idx in range(n_layers)],
        num_blocks=8,
        block_size=4,
        num_kv_heads=4,
        num_query_heads=28,
        head_size=128,
        dtype=torch.bfloat16,
        f_cpu_kv_store=0.5,
        max_num_reqs=2,
        max_num_tokens=16,
        max_model_len=16,
        pin_memory=torch.cuda.is_available(),
        kv_head_prefetch_enabled=True,
        kv_prefetch_max_active_blocks=3,
        kv_group_plan_by_bucket={
            1: (2, 0),
            2: (1, 1),
        },
    )


def _build_kv_prefetch_metadata(
    store: CotsHeadSplitKVStore,
    *,
    layer_name: str = "layer.0.attn",
    num_tokens: int = 2,
    seq_lens: tuple[int, ...] = (5, 3),
    positions: tuple[int, ...] = (4, 2),
):
    block_table = torch.tensor([[2, 4, 0, 0], [2, 5, 0, 0]], dtype=torch.int32)[
        :num_tokens
    ]
    return store.build_metadata(
        layer_name=layer_name,
        block_table_cpu=block_table,
        seq_lens_cpu=torch.tensor(seq_lens[:num_tokens], dtype=torch.int32),
        is_prefilling_cpu=torch.zeros(num_tokens, dtype=torch.bool),
        query_start_loc_cpu=torch.arange(num_tokens + 1, dtype=torch.int32),
        positions_cpu=torch.tensor(positions[:num_tokens], dtype=torch.long),
        max_query_len=1,
        num_actual_tokens=num_tokens,
        num_reqs=num_tokens,
    )


def _make_kv_prefetch_streamer(
    *,
    n_layers: int,
    max_prefetch_kv_heads: int = 1,
) -> KVPrefetchStreamer:
    pool = KVPrefetchBufferPool(
        n_layers=n_layers,
        max_active_blocks=3,
        max_num_reqs=2,
        max_blocks_per_req=4,
        block_size=4,
        max_prefetch_kv_heads=max_prefetch_kv_heads,
        head_dim=128,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
    )
    return KVPrefetchStreamer(n_layers=n_layers, buffer_pool=pool)


def _fill_prefetch_source_cache(
    metadata,
    *,
    key_base: int,
    value_base: int,
) -> None:
    metadata.cpu_key_cache.zero_()
    metadata.cpu_value_cache.zero_()
    for block in (2, 4):
        metadata.cpu_key_cache[block, 0].copy_(
            torch.arange(4 * 128, dtype=torch.bfloat16).view(4, 128) + key_base + block
        )
        metadata.cpu_value_cache[block, 0].copy_(
            torch.arange(4 * 128, dtype=torch.bfloat16).view(4, 128)
            + value_base
            + block
        )


def _make_current_kv(
    *,
    num_tokens: int,
    gpu_kv_heads: int,
    key_base: int,
    value_base: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    key = torch.zeros(num_tokens, 4, 128, dtype=torch.bfloat16, device="cuda")
    value = torch.zeros_like(key)
    for row in range(num_tokens):
        key[row, gpu_kv_heads].copy_(
            torch.arange(128, dtype=torch.bfloat16, device="cuda")
            + key_base
            + row * 1000
        )
        value[row, gpu_kv_heads].copy_(
            torch.arange(128, dtype=torch.bfloat16, device="cuda")
            + value_base
            + row * 1000
        )
    return key, value


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


def test_cots_head_split_qkv_route_exposes_prefetch_group_geometry() -> None:
    offloader = CotsOffloader(
        CotsOffloadConfig(
            f_cpu_store=0.5,
            kv_mode="head_split",
            f_cpu_kv_store=0.5,
            kv_head_prefetch_enabled=True,
            kv_prefetch_max_active_blocks=16,
        ),
        dispatch_table_factory=lambda buckets: {
            int(bucket): ((0.0, 0.5), (1, 1)) for bucket in buckets
        },
    )
    offloader._dispatch_buckets = (1,)
    offloader._graph_capture_buckets = ()
    offloader._build_dispatch_table()
    handle = _make_qkv_route_handle(cpu_weight_groups=1)
    num_tokens = 1
    qkv_group = handle.gqa_qkv_group_size
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
        out_pref=None,
        out_cpu=out_cpu,
        pref_idx=torch.empty(0, dtype=torch.long),
        bias=None,
    )

    sidecar = offloader.lookup_head_split_qkv_sidecar(out)
    assert sidecar is not None
    assert sidecar.cpu_attention_groups == 1
    assert sidecar.cpu_weight_groups == 1
    assert sidecar.cpu_compute_kv_heads == 1
    assert sidecar.prefetch_kv_heads == 1
    assert tuple(sidecar.query.shape) == (
        1,
        handle.gqa_q_group_size // handle.head_dim,
        handle.head_dim,
    )
    assert tuple(sidecar.key.shape) == (1, 1, handle.head_dim)
    assert tuple(sidecar.value.shape) == (1, 1, handle.head_dim)


@pytest.mark.parametrize(
    ("cpu_compute_groups", "prefetch_groups"),
    [
        (0, 2),
        (1, 1),
        (2, 0),
    ],
)
def test_cots_head_split_qkv_route_three_way_geometries_are_aligned(
    cpu_compute_groups: int,
    prefetch_groups: int,
) -> None:
    offloader = CotsOffloader(
        CotsOffloadConfig(
            f_cpu_store=0.5,
            kv_mode="head_split",
            f_cpu_kv_store=0.5,
            kv_head_prefetch_enabled=True,
            kv_prefetch_max_active_blocks=16,
        ),
        dispatch_table_factory=lambda buckets: {
            int(bucket): (
                (0.0, 0.5),
                (cpu_compute_groups, prefetch_groups),
            )
            for bucket in buckets
        },
    )
    offloader._dispatch_buckets = (1,)
    offloader._graph_capture_buckets = ()
    offloader._build_dispatch_table()
    handle = _make_qkv_route_handle(cpu_weight_groups=cpu_compute_groups)
    num_tokens = 1
    q_group = handle.gqa_q_group_size
    qkv_group = handle.gqa_qkv_group_size
    out_perm = torch.zeros(
        (num_tokens, handle.gpu_indices_cuda.numel()),
        dtype=torch.bfloat16,
    )
    out_cpu = None
    if cpu_compute_groups > 0:
        out_cpu = torch.arange(
            num_tokens * cpu_compute_groups * qkv_group,
            dtype=torch.bfloat16,
        ).view(num_tokens, cpu_compute_groups * qkv_group)

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
    assert sidecar.cpu_attention_groups == cpu_compute_groups
    assert sidecar.cpu_weight_groups == cpu_compute_groups
    assert sidecar.cpu_compute_kv_heads == cpu_compute_groups
    assert sidecar.prefetch_kv_heads == prefetch_groups
    assert tuple(sidecar.query.shape) == (
        num_tokens,
        cpu_compute_groups * (q_group // handle.head_dim),
        handle.head_dim,
    )
    assert tuple(sidecar.key.shape) == (
        num_tokens,
        cpu_compute_groups,
        handle.head_dim,
    )
    assert tuple(sidecar.value.shape) == (
        num_tokens,
        cpu_compute_groups,
        handle.head_dim,
    )

    if out_cpu is not None:
        expected_q = torch.cat(
            [
                out_cpu[:, group * qkv_group : group * qkv_group + q_group]
                for group in range(cpu_compute_groups)
            ],
            dim=1,
        )
        expected_k = torch.cat(
            [
                out_cpu[
                    :,
                    group * qkv_group + q_group : group * qkv_group
                    + q_group
                    + handle.head_dim,
                ]
                for group in range(cpu_compute_groups)
            ],
            dim=1,
        )
        expected_v = torch.cat(
            [
                out_cpu[
                    :,
                    group * qkv_group + q_group + handle.head_dim : (group + 1)
                    * qkv_group,
                ]
                for group in range(cpu_compute_groups)
            ],
            dim=1,
        )
        assert torch.equal(sidecar.query.reshape(num_tokens, -1), expected_q)
        assert torch.equal(sidecar.key.reshape(num_tokens, -1), expected_k)
        assert torch.equal(sidecar.value.reshape(num_tokens, -1), expected_v)


def test_cots_head_split_qkv_route_uses_live_rows_for_padded_bucket() -> None:
    offloader = CotsOffloader(
        CotsOffloadConfig(
            f_cpu_store=0.5,
            kv_mode="head_split",
            f_cpu_kv_store=0.5,
            kv_head_prefetch_enabled=True,
            kv_prefetch_max_active_blocks=16,
        ),
        dispatch_table_factory=lambda buckets: {
            int(bucket): ((0.0, 0.5), (1, 1)) for bucket in buckets
        },
    )
    offloader._dispatch_buckets = (4,)
    offloader._graph_capture_buckets = ()
    offloader._build_dispatch_table()
    handle = _make_qkv_route_handle(cpu_weight_groups=1)
    padded_tokens = 4
    cpu_rows = 1
    offloader.set_live_num_tokens(padded_tokens)
    q_group = handle.gqa_q_group_size
    qkv_group = handle.gqa_qkv_group_size
    out_perm = torch.zeros(
        (padded_tokens, handle.gpu_indices_cuda.numel()),
        dtype=torch.bfloat16,
    )
    out_cpu = torch.arange(
        cpu_rows * qkv_group,
        dtype=torch.bfloat16,
    ).view(cpu_rows, qkv_group)

    out = offloader.route_head_split_qkv_output(
        handle=handle,
        bucket=4,
        num_tokens=padded_tokens,
        reference=out_perm,
        out_perm=out_perm,
        out_pref=None,
        out_cpu=out_cpu,
        pref_idx=torch.empty(0, dtype=torch.long),
        bias=None,
    )

    sidecar = offloader.lookup_head_split_qkv_sidecar(out)
    assert sidecar is not None
    assert tuple(out.shape) == (padded_tokens, handle.out_dim)
    assert sidecar.num_tokens == cpu_rows
    assert tuple(sidecar.query.shape) == (
        cpu_rows,
        q_group // handle.head_dim,
        handle.head_dim,
    )
    assert torch.equal(
        sidecar.query.reshape(cpu_rows, q_group),
        out_cpu[:, :q_group],
    )


def test_cots_head_split_routed_query_is_c_compact() -> None:
    q_heads_per_kv = 2
    head_dim = 4
    routed_query = torch.arange(
        q_heads_per_kv * head_dim,
        dtype=torch.bfloat16,
    ).view(1, q_heads_per_kv, head_dim)
    metadata = CotsHeadSplitAttentionMetadata(
        cpu_key_cache=torch.empty(1, 2, 1, head_dim, dtype=torch.bfloat16),
        cpu_value_cache=torch.empty(1, 2, 1, head_dim, dtype=torch.bfloat16),
        cpu_block_table=torch.empty(1, 1, dtype=torch.int32),
        cpu_slot_mapping=torch.empty(1, dtype=torch.long),
        cpu_seq_lens=torch.empty(1, dtype=torch.int32),
        gpu_kv_heads=2,
        cpu_kv_heads=2,
        q_heads_per_kv=q_heads_per_kv,
        cpu_query_start=4,
        cpu_query_heads=4,
        is_decode=True,
        num_actual_tokens=1,
        prefetch_kv_heads=1,
        cpu_compute_kv_heads=1,
        prefetch_query_start=4,
        prefetch_query_heads=2,
        cpu_compute_query_start=6,
        cpu_compute_query_heads=2,
        prefetch_cpu_kv_start=0,
        cpu_compute_cpu_kv_start=1,
        routed_query_cpu=routed_query,
    )

    query_cpu, used_routed = _prepare_cpu_query(
        query=torch.empty(1, 8, head_dim, dtype=torch.bfloat16),
        metadata=metadata,
        num_tokens=1,
        q_start=6,
        q_end=8,
        query_heads=2,
    )

    assert used_routed
    assert torch.equal(query_cpu, routed_query)


def test_cots_head_split_qkv_route_marks_p_only_sidecar_rope_applied() -> None:
    offloader = CotsOffloader(
        CotsOffloadConfig(
            f_cpu_store=0.25,
            kv_mode="head_split",
            f_cpu_kv_store=0.25,
            kv_head_prefetch_enabled=True,
            kv_prefetch_max_active_blocks=16,
        ),
        dispatch_table_factory=lambda buckets: {
            int(bucket): ((0.0, 0.25), (0, 1)) for bucket in buckets
        },
    )
    offloader._dispatch_buckets = (1,)
    offloader._graph_capture_buckets = ()
    offloader._build_dispatch_table()
    handle = _make_qkv_route_handle(cpu_weight_groups=0)
    out_perm = torch.zeros((1, handle.out_dim), dtype=torch.bfloat16)

    out = offloader.route_head_split_qkv_output(
        handle=handle,
        bucket=1,
        num_tokens=1,
        reference=out_perm,
        out_perm=out_perm,
        out_pref=None,
        out_cpu=None,
        pref_idx=torch.empty(0, dtype=torch.long),
        bias=None,
    )

    sidecar = offloader.lookup_head_split_qkv_sidecar(out)
    assert sidecar is not None
    assert sidecar.cpu_attention_groups == 0
    assert sidecar.cpu_compute_kv_heads == 0
    assert sidecar.prefetch_kv_heads == 1
    assert sidecar.rope_applied


def test_cots_head_split_qkv_route_rejects_dispatch_weight_mismatch() -> None:
    offloader = CotsOffloader(
        CotsOffloadConfig(
            f_cpu_store=0.5,
            kv_mode="head_split",
            f_cpu_kv_store=0.5,
            kv_head_prefetch_enabled=True,
            kv_prefetch_max_active_blocks=16,
        ),
        dispatch_table_factory=lambda buckets: {
            int(bucket): ((0.0, 0.5), (0, 2)) for bucket in buckets
        },
    )
    offloader._dispatch_buckets = (1,)
    offloader._graph_capture_buckets = ()
    offloader._build_dispatch_table()
    handle = _make_qkv_route_handle(cpu_weight_groups=1)
    num_tokens = 1
    out_perm = torch.zeros(
        (num_tokens, handle.gpu_indices_cuda.numel()), dtype=torch.bfloat16
    )

    with pytest.raises(RuntimeError, match="must match snapped WQKV"):
        offloader.route_head_split_qkv_output(
            handle=handle,
            bucket=1,
            num_tokens=num_tokens,
            reference=out_perm,
            out_perm=out_perm,
            out_pref=None,
            out_cpu=torch.full(
                (num_tokens, handle.gqa_qkv_group_size),
                7.0,
                dtype=torch.bfloat16,
            ),
            pref_idx=torch.empty(0, dtype=torch.long),
            bias=None,
        )


def test_cots_head_split_dispatch_validation_matches_snapped_weight_groups() -> None:
    offloader = CotsOffloader(
        CotsOffloadConfig(
            f_cpu_store=0.5,
            kv_mode="head_split",
            f_cpu_kv_store=0.5,
            kv_head_prefetch_enabled=True,
            kv_prefetch_max_active_blocks=16,
        ),
        dispatch_table_factory=lambda buckets: {
            int(bucket): ((0.0, 0.5), (1, 1)) for bucket in buckets
        },
    )
    offloader._dispatch_buckets = (1,)
    offloader._graph_capture_buckets = ()
    offloader._build_dispatch_table()
    offloader._handles = [
        _make_qkv_route_handle(cpu_weight_groups=1),
        _make_wo_route_handle(cpu_weight_groups=1),
    ]

    offloader._validate_head_split_kv_dispatch_geometry()


def test_cots_head_split_dispatch_validation_rejects_bad_group_sum() -> None:
    offloader = CotsOffloader(
        CotsOffloadConfig(
            f_cpu_store=0.5,
            kv_mode="head_split",
            f_cpu_kv_store=0.5,
            kv_head_prefetch_enabled=True,
            kv_prefetch_max_active_blocks=16,
        ),
        dispatch_table_factory=lambda buckets: {
            int(bucket): ((0.0, 0.5), (1, 0)) for bucket in buckets
        },
    )
    offloader._dispatch_buckets = (1,)
    offloader._graph_capture_buckets = ()
    offloader._build_dispatch_table()
    offloader._handles = [
        _make_qkv_route_handle(cpu_weight_groups=1),
        _make_wo_route_handle(cpu_weight_groups=1),
    ]

    with pytest.raises(ValueError, match="cover exactly"):
        offloader._validate_head_split_kv_dispatch_geometry()


def test_cots_head_split_dispatch_validation_rejects_weight_mismatch() -> None:
    offloader = CotsOffloader(
        CotsOffloadConfig(
            f_cpu_store=0.5,
            kv_mode="head_split",
            f_cpu_kv_store=0.5,
            kv_head_prefetch_enabled=True,
            kv_prefetch_max_active_blocks=16,
        ),
        dispatch_table_factory=lambda buckets: {
            int(bucket): ((0.0, 0.5), (2, 0)) for bucket in buckets
        },
    )
    offloader._dispatch_buckets = (1,)
    offloader._graph_capture_buckets = ()
    offloader._build_dispatch_table()
    offloader._handles = [
        _make_qkv_route_handle(cpu_weight_groups=1),
        _make_wo_route_handle(cpu_weight_groups=1),
    ]

    with pytest.raises(ValueError, match="snapped WQKV"):
        offloader._validate_head_split_kv_dispatch_geometry()


def test_cots_head_split_kv_prefetch_buffer_uses_workload_contract() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    cots_config = CotsOffloadConfig(
        f_cpu_store=0.5,
        kv_mode="head_split",
        f_cpu_kv_store=0.5,
        kv_head_prefetch_enabled=True,
        kv_prefetch_max_active_blocks=7,
    )
    vllm_config = VllmConfig(
        cache_config=CacheConfig(block_size=4),
        offload_config=OffloadConfig(offload_backend="cots", cots=cots_config),
    )
    offloader = CotsOffloader(
        cots_config,
        dispatch_table_factory=lambda buckets: {
            int(bucket): (
                (0.0, 0.5),
                (1, 1) if int(bucket) == 1 else (0, 2),
            )
            for bucket in buckets
        },
    )
    offloader._dispatch_buckets = (1, 8)
    offloader._graph_capture_buckets = ()
    offloader._layer_modules = [torch.nn.Identity()]
    offloader._layer_handles = [[]]
    offloader._handles = [_make_qkv_route_handle(cpu_weight_groups=1)]
    offloader._build_dispatch_table()

    with set_current_vllm_config(vllm_config):
        offloader._install_kv_prefetch_machinery()

    pool = offloader._kv_prefetch_buffer_pool
    assert pool is not None
    assert pool.max_active_blocks == 7
    assert pool.block_size == 4
    assert pool.max_prefetch_kv_heads == 2
    assert tuple(pool.slot_for_layer(0).shape) == (2, 7, 4, 2, 2)
    assert offloader._kv_prefetch_streamer is not None
    assert offloader._kv_prefetch_streamer.buffer_pool is pool


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


def test_cots_head_split_wo_route_fills_preallocated_cpu_input() -> None:
    offloader = CotsOffloader(
        CotsOffloadConfig(kv_mode="head_split", f_cpu_kv_store=0.25)
    )
    num_tokens = 1
    num_groups = 4
    q_heads_per_kv = 2
    head_dim = 2
    q_group = q_heads_per_kv * head_dim
    handle = _make_wo_route_handle(cpu_weight_groups=2)
    x = torch.arange(num_tokens * num_groups * q_group, dtype=torch.bfloat16).view(
        num_tokens, num_groups * q_group
    )
    output_cpu = torch.full(
        (num_tokens, q_heads_per_kv, head_dim),
        9.0,
        dtype=torch.bfloat16,
    )
    qkv_sidecar = CotsHeadSplitQKVSidecar(
        storage_key=123,
        num_tokens=num_tokens,
        num_groups=num_groups,
        cpu_attention_groups=1,
        cpu_weight_groups=2,
        cpu_compute_kv_heads=1,
        prefetch_kv_heads=0,
        q_heads_per_kv=q_heads_per_kv,
        head_dim=head_dim,
        query=torch.empty(num_tokens, q_heads_per_kv, head_dim),
        key=torch.empty(num_tokens, 1, head_dim),
        value=torch.empty(num_tokens, 1, head_dim),
    )
    offloader.register_head_split_attention_output(
        output=x,
        qkv_sidecar=qkv_sidecar,
        output_cpu=output_cpu,
    )

    dst = torch.empty(num_tokens, 2 * q_group, dtype=torch.bfloat16)
    routed = offloader.build_head_split_wo_cpu_input(
        x=x,
        handle=handle,
        bucket=1,
        num_tokens=num_tokens,
        out=dst,
    )

    assert routed is dst
    assert torch.equal(dst[:, :q_group], x[:, 2 * q_group : 3 * q_group])
    assert torch.equal(dst[:, q_group:], output_cpu.reshape(num_tokens, q_group))
    assert offloader.lookup_head_split_attention_output(x) is None


def test_cots_head_split_wo_route_accepts_c_only_cpu_attention_output() -> None:
    offloader = CotsOffloader(
        CotsOffloadConfig(
            f_cpu_store=0.5,
            kv_mode="head_split",
            f_cpu_kv_store=0.5,
            kv_head_prefetch_enabled=True,
            kv_prefetch_max_active_blocks=16,
        )
    )
    num_tokens = 1
    num_groups = 4
    q_heads_per_kv = 2
    head_dim = 2
    q_group = q_heads_per_kv * head_dim
    handle = _make_wo_route_handle(cpu_weight_groups=1)
    x = torch.arange(num_tokens * num_groups * q_group, dtype=torch.bfloat16).view(
        num_tokens, num_groups * q_group
    )
    c_only_output_cpu = torch.full(
        (num_tokens, q_heads_per_kv, head_dim),
        11.0,
        dtype=torch.bfloat16,
    )
    qkv_sidecar = CotsHeadSplitQKVSidecar(
        storage_key=456,
        num_tokens=num_tokens,
        num_groups=num_groups,
        cpu_attention_groups=1,
        cpu_weight_groups=1,
        cpu_compute_kv_heads=1,
        prefetch_kv_heads=1,
        q_heads_per_kv=q_heads_per_kv,
        head_dim=head_dim,
        query=torch.empty(num_tokens, q_heads_per_kv, head_dim),
        key=torch.empty(num_tokens, 1, head_dim),
        value=torch.empty(num_tokens, 1, head_dim),
    )
    offloader.register_head_split_attention_output(
        output=x,
        qkv_sidecar=qkv_sidecar,
        output_cpu=c_only_output_cpu,
    )

    dst = torch.empty(num_tokens, q_group, dtype=torch.bfloat16)
    routed = offloader.build_head_split_wo_cpu_input(
        x=x,
        handle=handle,
        bucket=1,
        num_tokens=num_tokens,
        out=dst,
    )

    assert routed is dst
    assert torch.equal(dst, c_only_output_cpu.reshape(num_tokens, q_group))
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
    assert not store.kv_head_prefetch_enabled

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
    assert metadata.cpu_compute_kv_heads == 1
    assert metadata.prefetch_kv_heads == 0
    assert metadata.cpu_compute_query_start == 21
    assert metadata.cpu_compute_query_heads == 7

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


def test_cots_head_split_store_keeps_cpu_geometry_without_sidecar() -> None:
    store = CotsHeadSplitKVStore(
        layer_names=["layer.0.attn"],
        num_blocks=4,
        block_size=4,
        num_kv_heads=4,
        num_query_heads=28,
        head_size=128,
        dtype=torch.bfloat16,
        f_cpu_kv_store=0.5,
        max_num_reqs=2,
        max_num_tokens=16,
        max_model_len=16,
        pin_memory=False,
        kv_head_prefetch_enabled=True,
        kv_prefetch_max_active_blocks=4,
        kv_group_plan_by_bucket={1: (2, 0)},
    )

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

    assert metadata.cpu_kv_heads == 2
    assert metadata.cpu_compute_kv_heads == 2
    assert metadata.prefetch_kv_heads == 0
    assert metadata.prefetch_query_start == 14
    assert metadata.prefetch_query_heads == 0
    assert metadata.cpu_compute_query_start == 14
    assert metadata.cpu_compute_query_heads == 14
    assert metadata.cpu_compute_cpu_kv_start == 0


def test_cots_head_split_kv_prefetch_descriptor_compacts_active_blocks() -> None:
    store = CotsHeadSplitKVStore(
        layer_names=["layer.0.attn", "layer.1.attn"],
        num_blocks=8,
        block_size=4,
        num_kv_heads=4,
        num_query_heads=28,
        head_size=128,
        dtype=torch.bfloat16,
        f_cpu_kv_store=0.5,
        max_num_reqs=2,
        max_num_tokens=16,
        max_model_len=16,
        pin_memory=False,
        kv_head_prefetch_enabled=True,
        kv_prefetch_max_active_blocks=3,
        kv_group_plan_by_bucket={2: (1, 1)},
    )

    metadata = store.build_metadata(
        layer_name="layer.0.attn",
        block_table_cpu=torch.tensor([[2, 4, 0, 0], [2, 5, 0, 0]], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([5, 3], dtype=torch.int32),
        is_prefilling_cpu=torch.tensor([False, False]),
        query_start_loc_cpu=torch.tensor([0, 1, 2], dtype=torch.int32),
        positions_cpu=torch.tensor([4, 2], dtype=torch.long),
        max_query_len=1,
        num_actual_tokens=2,
        num_reqs=2,
    )

    assert metadata.prefetch_kv_heads == 1
    assert metadata.cpu_compute_kv_heads == 1
    assert metadata.prefetch_cpu_kv_start == 0
    assert metadata.cpu_compute_cpu_kv_start == 1
    descriptor = metadata.kv_prefetch
    assert descriptor is not None
    assert descriptor.layer_idx == 0
    assert descriptor.num_active_blocks == 2
    assert descriptor.source_block_ids.tolist() == [2, 4]
    assert descriptor.destination_block_ids.tolist() == [0, 1]
    assert descriptor.compact_block_table.tolist() == [
        [0, 1, 0, 0],
        [0, 0, 0, 0],
    ]
    assert descriptor.prefetch_kv_heads == 1
    assert descriptor.block_size == 4
    assert descriptor.source_key_cache is metadata.cpu_key_cache

    layer1_metadata = store.build_metadata_from_common(
        layer_name="layer.1.attn",
        common_metadata=metadata,
    )
    assert layer1_metadata.kv_prefetch is not None
    assert layer1_metadata.kv_prefetch.source_block_ids is descriptor.source_block_ids
    assert layer1_metadata.kv_prefetch.compact_block_table is (
        descriptor.compact_block_table
    )
    assert layer1_metadata.kv_prefetch.source_key_cache is (
        layer1_metadata.cpu_key_cache
    )
    assert layer1_metadata.kv_prefetch.source_key_cache is not (
        descriptor.source_key_cache
    )
    assert layer1_metadata.kv_prefetch.layer_idx == 1


def test_cots_head_split_kv_prefetch_descriptor_checks_contract() -> None:
    store = CotsHeadSplitKVStore(
        layer_names=["layer.0.attn"],
        num_blocks=8,
        block_size=4,
        num_kv_heads=4,
        num_query_heads=28,
        head_size=128,
        dtype=torch.bfloat16,
        f_cpu_kv_store=0.5,
        max_num_reqs=2,
        max_num_tokens=16,
        max_model_len=16,
        pin_memory=False,
        kv_head_prefetch_enabled=True,
        kv_prefetch_max_active_blocks=1,
        kv_group_plan_by_bucket={2: (1, 1)},
    )

    with pytest.raises(RuntimeError, match="exceeds workload contract"):
        store.build_metadata(
            layer_name="layer.0.attn",
            block_table_cpu=torch.tensor(
                [[2, 4, 0, 0], [2, 5, 0, 0]], dtype=torch.int32
            ),
            seq_lens_cpu=torch.tensor([5, 3], dtype=torch.int32),
            is_prefilling_cpu=torch.tensor([False, False]),
            query_start_loc_cpu=torch.tensor([0, 1, 2], dtype=torch.int32),
            positions_cpu=torch.tensor([4, 2], dtype=torch.long),
            max_query_len=1,
            num_actual_tokens=2,
            num_reqs=2,
        )


def test_cots_head_split_kv_prefetch_streamer_stages_compact_blocks() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    store = CotsHeadSplitKVStore(
        layer_names=["layer.0.attn"],
        num_blocks=8,
        block_size=4,
        num_kv_heads=4,
        num_query_heads=28,
        head_size=128,
        dtype=torch.bfloat16,
        f_cpu_kv_store=0.5,
        max_num_reqs=2,
        max_num_tokens=16,
        max_model_len=16,
        pin_memory=torch.cuda.is_available(),
        kv_head_prefetch_enabled=True,
        kv_prefetch_max_active_blocks=3,
        kv_group_plan_by_bucket={2: (1, 1)},
    )
    metadata = store.build_metadata(
        layer_name="layer.0.attn",
        block_table_cpu=torch.tensor([[2, 4, 0, 0], [2, 5, 0, 0]], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([5, 3], dtype=torch.int32),
        is_prefilling_cpu=torch.tensor([False, False]),
        query_start_loc_cpu=torch.tensor([0, 1, 2], dtype=torch.int32),
        positions_cpu=torch.tensor([4, 2], dtype=torch.long),
        max_query_len=1,
        num_actual_tokens=2,
        num_reqs=2,
    )
    descriptor = metadata.kv_prefetch
    assert descriptor is not None

    key_cache = metadata.cpu_key_cache
    value_cache = metadata.cpu_value_cache
    key_cache.zero_()
    value_cache.zero_()
    for block in (2, 4):
        key_cache[block, 0].copy_(
            torch.arange(4 * 128, dtype=torch.bfloat16).view(4, 128) + block
        )
        value_cache[block, 0].copy_(
            torch.arange(4 * 128, dtype=torch.bfloat16).view(4, 128) + block + 100
        )

    pool = KVPrefetchBufferPool(
        n_layers=1,
        max_active_blocks=3,
        max_num_reqs=2,
        max_blocks_per_req=4,
        block_size=4,
        max_prefetch_kv_heads=1,
        head_dim=128,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
    )
    streamer = KVPrefetchStreamer(n_layers=1, buffer_pool=pool)
    streamer.publish_descriptor(0, descriptor)
    streamer.prepare_for_forward_bucket(0)
    streamer.wait(0)
    torch.cuda.current_stream().synchronize()

    key_slot, value_slot = pool.key_value_slots(0)
    key_slot_cpu = key_slot[:2, :, :1, :].cpu()
    value_slot_cpu = value_slot[:2, :, :1, :].cpu()
    assert torch.equal(key_slot_cpu[0, :, 0, :], key_cache[2, 0])
    assert torch.equal(key_slot_cpu[1, :, 0, :], key_cache[4, 0])
    assert torch.equal(value_slot_cpu[0, :, 0, :], value_cache[2, 0])
    assert torch.equal(value_slot_cpu[1, :, 0, :], value_cache[4, 0])
    compact_table = pool.compact_block_table_slot(0)
    assert compact_table[:2, :4].cpu().tolist() == [
        [0, 1, 0, 0],
        [0, 0, 0, 0],
    ]
    assert pool.owner_layer_in_slot[0] == 0
    assert pool.available_blocks_in_slot[0] == 2
    assert pool.available_heads_in_slot[0] == 1
    attn_key, attn_value, attn_table = streamer.attention_inputs(
        layer_idx=0,
        descriptor=descriptor,
    )
    assert tuple(attn_key.shape) == (2, 4, 1, 128)
    assert tuple(attn_value.shape) == (2, 4, 1, 128)
    assert tuple(attn_table.shape) == (2, 4)
    assert torch.equal(attn_table.cpu(), descriptor.compact_block_table)

    key = torch.zeros(2, 4, 128, dtype=torch.bfloat16, device="cuda")
    value = torch.zeros_like(key)
    key[0, 2].copy_(torch.arange(128, dtype=torch.bfloat16, device="cuda") + 1000)
    key[1, 2].copy_(torch.arange(128, dtype=torch.bfloat16, device="cuda") + 2000)
    key[0, 3].copy_(torch.arange(128, dtype=torch.bfloat16, device="cuda") + 3000)
    value[0, 2].copy_(torch.arange(128, dtype=torch.bfloat16, device="cuda") + 4000)
    value[1, 2].copy_(torch.arange(128, dtype=torch.bfloat16, device="cuda") + 5000)
    value[0, 3].copy_(torch.arange(128, dtype=torch.bfloat16, device="cuda") + 6000)
    routed_key_cpu = torch.empty(2, 1, 128, dtype=torch.bfloat16)
    routed_value_cpu = torch.empty_like(routed_key_cpu)
    routed_key_cpu[0, 0].copy_(torch.arange(128, dtype=torch.bfloat16) + 7000)
    routed_key_cpu[1, 0].copy_(torch.arange(128, dtype=torch.bfloat16) + 8000)
    routed_value_cpu[0, 0].copy_(torch.arange(128, dtype=torch.bfloat16) + 9000)
    routed_value_cpu[1, 0].copy_(torch.arange(128, dtype=torch.bfloat16) + 10000)
    metadata.routed_key_cpu = routed_key_cpu
    metadata.routed_value_cpu = routed_value_cpu

    cots_head_split_kv_cache_update(
        key,
        value,
        torch.empty(2, dtype=torch.long, device="meta"),
        metadata,
    )
    streamer.patch_current_kv(
        layer_idx=0,
        descriptor=descriptor,
        key=key,
        value=value,
        cpu_slot_mapping=metadata.cpu_slot_mapping,
        gpu_kv_heads=metadata.gpu_kv_heads,
        num_actual_tokens=metadata.num_actual_tokens,
    )
    assert streamer.wait_for_slot_writeback(0)

    key_slot_cpu = key_slot[:2, :, :1, :].cpu()
    value_slot_cpu = value_slot[:2, :, :1, :].cpu()
    assert torch.equal(key_slot_cpu[1, 0, 0, :], key[0, 2].cpu())
    assert torch.equal(key_slot_cpu[0, 2, 0, :], key[1, 2].cpu())
    assert torch.equal(value_slot_cpu[1, 0, 0, :], value[0, 2].cpu())
    assert torch.equal(value_slot_cpu[0, 2, 0, :], value[1, 2].cpu())
    assert torch.equal(key_cache[4, 0, 0], key[0, 2].cpu())
    assert torch.equal(key_cache[2, 0, 2], key[1, 2].cpu())
    assert torch.equal(key_cache[4, 1, 0], routed_key_cpu[0, 0])
    assert torch.equal(key_cache[2, 1, 2], routed_key_cpu[1, 0])
    assert torch.equal(value_cache[4, 1, 0], routed_value_cpu[0, 0])


def test_cots_head_split_kv_prefetch_slot_reuse_waits_for_writeback() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    store = _make_kv_prefetch_store(n_layers=3)
    metadata0 = _build_kv_prefetch_metadata(store)
    descriptor0 = metadata0.kv_prefetch
    assert descriptor0 is not None
    _fill_prefetch_source_cache(metadata0, key_base=100, value_base=1000)

    streamer = _make_kv_prefetch_streamer(n_layers=3)
    streamer.publish_descriptor(0, descriptor0)
    streamer.prepare_for_forward_bucket(0)
    streamer.wait(0)
    torch.cuda.current_stream().synchronize()

    key, value = _make_current_kv(
        num_tokens=2,
        gpu_kv_heads=metadata0.gpu_kv_heads,
        key_base=2000,
        value_base=3000,
    )
    streamer.patch_current_kv(
        layer_idx=0,
        descriptor=descriptor0,
        key=key,
        value=value,
        cpu_slot_mapping=metadata0.cpu_slot_mapping,
        gpu_kv_heads=metadata0.gpu_kv_heads,
        num_actual_tokens=metadata0.num_actual_tokens,
    )
    assert streamer._writeback_pending[0]
    assert streamer._writeback_owner_layer[0] == 0

    metadata2 = store.build_metadata_from_common(
        layer_name="layer.2.attn",
        common_metadata=metadata0,
    )
    descriptor2 = metadata2.kv_prefetch
    assert descriptor2 is not None
    _fill_prefetch_source_cache(metadata2, key_base=10000, value_base=11000)

    streamer.publish_descriptor(2, descriptor2)
    streamer.start(2)
    streamer.wait(2)
    torch.cuda.current_stream().synchronize()

    pool = streamer.buffer_pool
    assert pool is not None
    assert not streamer._writeback_pending[0]
    assert streamer._writeback_owner_layer[0] is None
    assert pool.owner_layer_in_slot[0] == 2
    assert torch.equal(metadata0.cpu_key_cache[4, 0, 0], key[0, 2].cpu())
    assert torch.equal(metadata0.cpu_value_cache[2, 0, 2], value[1, 2].cpu())

    key_slot, _ = pool.key_value_slots(2)
    key_slot_cpu = key_slot[:2, :, :1, :].cpu()
    assert torch.equal(key_slot_cpu[0, :, 0, :], metadata2.cpu_key_cache[2, 0])
    assert torch.equal(key_slot_cpu[1, :, 0, :], metadata2.cpu_key_cache[4, 0])


def test_cots_head_split_kv_prefetch_bucket_change_waits_before_cpu_update() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    store = _make_kv_prefetch_store()
    metadata_p = _build_kv_prefetch_metadata(store)
    descriptor = metadata_p.kv_prefetch
    assert descriptor is not None
    streamer = _make_kv_prefetch_streamer(n_layers=1)
    key_p, value_p = _make_current_kv(
        num_tokens=2,
        gpu_kv_heads=metadata_p.gpu_kv_heads,
        key_base=4000,
        value_base=5000,
    )
    streamer.patch_current_kv(
        layer_idx=0,
        descriptor=descriptor,
        key=key_p,
        value=value_p,
        cpu_slot_mapping=metadata_p.cpu_slot_mapping,
        gpu_kv_heads=metadata_p.gpu_kv_heads,
        num_actual_tokens=metadata_p.num_actual_tokens,
    )
    assert streamer._writeback_pending[0]

    class _WaitStub:
        def __init__(self) -> None:
            self.calls = 0

        def wait_head_split_kv_prefetch_writeback(self, layer_idx: int) -> bool:
            self.calls += 1
            return streamer.wait_for_layer_writeback(layer_idx)

    metadata_c = _build_kv_prefetch_metadata(
        store,
        num_tokens=1,
        seq_lens=(6,),
        positions=(5,),
    )
    assert metadata_c.kv_prefetch is None
    assert metadata_c.cpu_compute_kv_heads == 2
    key_c = torch.zeros(1, 4, 128, dtype=torch.bfloat16, device="cuda")
    value_c = torch.zeros_like(key_c)
    key_c[0, 2].copy_(torch.arange(128, dtype=torch.bfloat16, device="cuda") + 6000)
    value_c[0, 2].copy_(torch.arange(128, dtype=torch.bfloat16, device="cuda") + 7000)

    original_offloader = get_offloader()
    wait_stub = _WaitStub()
    set_offloader(wait_stub)
    try:
        cots_head_split_kv_cache_update(
            key_c,
            value_c,
            torch.empty(1, dtype=torch.long, device="meta"),
            metadata_c,
        )
    finally:
        set_offloader(original_offloader)

    assert wait_stub.calls == 1
    assert not streamer._writeback_pending[0]
    assert streamer._writeback_owner_layer[0] is None
    assert torch.equal(metadata_p.cpu_key_cache[4, 0, 0], key_p[0, 2].cpu())
    assert torch.equal(metadata_p.cpu_value_cache[2, 0, 2], value_p[1, 2].cpu())
    assert torch.equal(metadata_c.cpu_key_cache[4, 0, 1], key_c[0, 2].cpu())
    assert torch.equal(metadata_c.cpu_value_cache[4, 0, 1], value_c[0, 2].cpu())


def test_cots_head_split_kv_prefetch_unrelated_layer_keeps_writeback_pending() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    store = _make_kv_prefetch_store(n_layers=2)
    metadata0 = _build_kv_prefetch_metadata(store)
    descriptor0 = metadata0.kv_prefetch
    assert descriptor0 is not None
    metadata1 = store.build_metadata_from_common(
        layer_name="layer.1.attn",
        common_metadata=metadata0,
    )
    descriptor1 = metadata1.kv_prefetch
    assert descriptor1 is not None

    streamer = _make_kv_prefetch_streamer(n_layers=2)
    key, value = _make_current_kv(
        num_tokens=2,
        gpu_kv_heads=metadata0.gpu_kv_heads,
        key_base=8000,
        value_base=9000,
    )
    streamer.patch_current_kv(
        layer_idx=0,
        descriptor=descriptor0,
        key=key,
        value=value,
        cpu_slot_mapping=metadata0.cpu_slot_mapping,
        gpu_kv_heads=metadata0.gpu_kv_heads,
        num_actual_tokens=metadata0.num_actual_tokens,
    )
    assert streamer._writeback_pending[0]
    assert streamer._writeback_owner_layer[0] == 0

    assert not streamer.wait_for_layer_writeback(1)
    assert streamer._writeback_pending[0]
    assert streamer._writeback_owner_layer[0] == 0

    _fill_prefetch_source_cache(metadata1, key_base=12000, value_base=13000)
    streamer.publish_descriptor(1, descriptor1)
    streamer.start(1)
    streamer.wait(1)
    torch.cuda.current_stream().synchronize()

    pool = streamer.buffer_pool
    assert pool is not None
    assert pool.owner_layer_in_slot[1] == 1
    assert streamer._writeback_pending[0]
    assert streamer._writeback_owner_layer[0] == 0
    assert streamer.wait_for_slot_writeback(0)
    assert torch.equal(metadata0.cpu_key_cache[4, 0, 0], key[0, 2].cpu())


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
