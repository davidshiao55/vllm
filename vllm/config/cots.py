# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared COTS weight-module policy helpers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

CotsWeightModule = Literal["qkv", "mlp", "wo"]

COTS_WEIGHT_MODULE_ORDER: tuple[str, ...] = ("qkv", "mlp", "wo")
VALID_COTS_WEIGHT_MODULES: frozenset[str] = frozenset(COTS_WEIGHT_MODULE_ORDER)
DEFAULT_COTS_WEIGHT_MODULES: frozenset[str] = frozenset(("qkv", "mlp"))

COTS_WEIGHT_MODULE_SUFFIXES: dict[str, tuple[str, ...]] = {
    "qkv": ("qkv_proj",),
    "mlp": ("gate_up_proj", "down_proj"),
    "wo": ("o_proj",),
}


def normalize_cots_weight_modules(
    raw: object,
    *,
    default: Iterable[str] = DEFAULT_COTS_WEIGHT_MODULES,
) -> set[str]:
    """Normalize user-facing COTS module selection.

    Accepts set/list style values as well as comma-separated strings so CLI,
    JSON, and programmatic configs converge on the same representation.
    """
    entries: Iterable[object]
    if raw is None:
        entries = default
    elif isinstance(raw, str):
        entries = (raw,)
    elif isinstance(raw, Iterable):
        entries = raw
    else:
        entries = (raw,)

    modules: set[str] = set()
    for entry in entries:
        for module in str(entry).split(","):
            module = module.strip().lower()
            if module:
                modules.add(module)

    unknown = modules - VALID_COTS_WEIGHT_MODULES
    if unknown:
        raise ValueError(
            f"cots.weight_modules contains unsupported entries "
            f"{sorted(unknown)}; expected subset of "
            f"{sorted(VALID_COTS_WEIGHT_MODULES)}"
        )
    return modules


def cots_weight_module_for_name(
    enabled_modules: Iterable[str],
    qualified_name: str,
) -> str | None:
    """Return the enabled semantic COTS module for a linear name."""
    enabled = set(enabled_modules)
    for module in COTS_WEIGHT_MODULE_ORDER:
        if module not in enabled:
            continue
        suffixes = COTS_WEIGHT_MODULE_SUFFIXES[module]
        if any(qualified_name.endswith(suffix) for suffix in suffixes):
            return module
    return None
