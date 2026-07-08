# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
from vllm.config import CUDAGraphMode
from vllm.forward_context import (
    BatchDescriptor,
    ForwardContext,
    override_forward_context,
)


def _make_forward_context(
    desc: BatchDescriptor,
    mode: CUDAGraphMode = CUDAGraphMode.NONE,
) -> ForwardContext:
    return ForwardContext(
        no_compile_layers={},
        attn_metadata={},
        slot_mapping={},
        cudagraph_runtime_mode=mode,
        batch_descriptor=desc,
    )


def _make_wrapper(cots_active: bool) -> TorchCompileWithNoGuardsWrapper:
    wrapper = TorchCompileWithNoGuardsWrapper.__new__(TorchCompileWithNoGuardsWrapper)
    wrapper.vllm_config = SimpleNamespace(
        offload_config=SimpleNamespace(
            offload_backend=("cots" if cots_active else "none"),
            cots=SimpleNamespace(f_cpu_store=(0.3 if cots_active else 0.0)),
        )
    )
    return wrapper


def test_cots_compile_key_is_absent_without_cots_weight_offload() -> None:
    wrapper = _make_wrapper(cots_active=False)
    desc = BatchDescriptor(num_tokens=256)

    with override_forward_context(_make_forward_context(desc)):
        assert wrapper._cots_compile_key() is None


def test_cots_compile_key_is_full_batch_descriptor() -> None:
    wrapper = _make_wrapper(cots_active=True)
    desc_256 = BatchDescriptor(
        num_tokens=256,
        num_reqs=256,
        uniform=True,
    )
    desc_272 = BatchDescriptor(
        num_tokens=272,
        num_reqs=None,
        uniform=False,
    )

    with override_forward_context(_make_forward_context(desc_256, CUDAGraphMode.NONE)):
        warmup_256_key = wrapper._cots_compile_key()
    with override_forward_context(_make_forward_context(desc_256, CUDAGraphMode.FULL)):
        capture_256_key = wrapper._cots_compile_key()
    with override_forward_context(
        _make_forward_context(desc_272, CUDAGraphMode.PIECEWISE)
    ):
        piecewise_272_key = wrapper._cots_compile_key()

    assert warmup_256_key == desc_256
    assert capture_256_key == desc_256
    assert piecewise_272_key == desc_272
    assert warmup_256_key == capture_256_key
    assert capture_256_key != piecewise_272_key
