# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper


def test_cots_compile_key_specialization_is_removed() -> None:
    """COTS should follow vLLM's normal compile identity.

    Bucket routing lives behind COTS custom ops now, so the compile wrapper must
    not mint separate callables or bytecode caches from BatchDescriptor.
    """

    removed = {
        "_cots_compile_key",
        "_uses_cots_compile_variants",
        "_cots_compile_prefix",
        "_compiled_callable_for_cots_key",
        "_log_cots_compile_key",
    }
    for name in removed:
        assert not hasattr(TorchCompileWithNoGuardsWrapper, name)
