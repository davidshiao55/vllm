# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper


def test_hybrid_compile_key_specialization_is_removed() -> None:
    """Hybrid should follow vLLM's normal compile identity.

    Bucket routing lives behind Hybrid custom ops now, so the compile wrapper must
    not mint separate callables or bytecode caches from BatchDescriptor.
    """

    removed = {
        "_hybrid_compile_key",
        "_uses_hybrid_compile_variants",
        "_hybrid_compile_prefix",
        "_compiled_callable_for_hybrid_key",
        "_log_hybrid_compile_key",
    }
    for name in removed:
        assert not hasattr(TorchCompileWithNoGuardsWrapper, name)
