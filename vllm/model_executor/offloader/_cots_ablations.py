# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""§1c.26 / §1c.27 / §1c.34-cleanup-D: diagnostic-only ablation helper.

Five env vars (each `=1` to enable) skip specific captured-graph node
classes for attribution probing:

    VLLM_COTS_ABLATE_HOSTFN
        §1c.26 broad: skip captured `cudaLaunchHostFunc` for BOTH
        submit/dispatch and sync ("submit+sync" macro).
    VLLM_COTS_ABLATE_SUBMIT_HOSTFN
        §1c.27 narrow: skip ONLY the submit/dispatch host_fn.
    VLLM_COTS_ABLATE_SYNC_HOSTFN
        §1c.27 narrow: skip ONLY the sync host_fn.
    VLLM_COTS_ABLATE_D2H
        Skip captured `cudaMemcpyAsync` (per-op activation D2H).
    VLLM_COTS_ABLATE_UVA
        Skip the captured Triton UVA copy.

Honored ONLY when:
    * `cpu_runner='native'` — Python runner has no captured graph
    * `dry_run=True`         — worker doesn't compute; ablations safe
    * `VLLM_COTS_DIAG=1`     — diagnostic tooling on

The ablations silently corrupt outputs in real (non-dryrun) mode,
so misuse — any env var set without both gate conditions — must
HARD-FAIL with `RuntimeError`. Warn-and-skip would silently measure
the NON-ablated path.

§1c.34 cleanup D: moved out of `cots.py` so the production hot
path doesn't import this module at all unless `VLLM_COTS_DIAG=1`.
`CotsOffloader.post_init` short-circuits the env reads upstream;
this file is only imported when diagnostics are active.

This module is import-fenced from production: only
`CotsOffloader._install_ablations` (a thin shim) imports it, and
only when `VLLM_COTS_DIAG=1`.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.model_executor.offloader.cots import CotsOffloader

logger = logging.getLogger(__name__)


_ABLATION_ENVS = (
    "VLLM_COTS_ABLATE_HOSTFN",
    "VLLM_COTS_ABLATE_SUBMIT_HOSTFN",
    "VLLM_COTS_ABLATE_SYNC_HOSTFN",
    "VLLM_COTS_ABLATE_D2H",
    "VLLM_COTS_ABLATE_UVA",
)


def any_ablation_env_set() -> bool:
    """Cheap O(5)-env-read precheck. Called from
    `CotsOffloader.post_init` so the rest of this module never has
    to import on the production hot path."""
    return any(os.environ.get(k, "0") == "1" for k in _ABLATION_ENVS)


def install_ablations_from_env(offloader: CotsOffloader) -> None:
    """Read the five ablation env vars, validate the gate
    (`dry_run + VLLM_COTS_DIAG=1`), and push the active flags into
    the C++ infer + the cots_ops UVA flag. Hard-fail on misuse.

    Resets the process-global `_COTS_ABLATE_UVA` first so a prior
    install in the same process can't leak its UVA ablation into a
    later non-ablating offloader.
    """
    from vllm.model_executor.offloader import cots_ops
    from vllm.model_executor.offloader.cots import NativeCotsRunner
    from vllm.utils.cots_diag import ENABLED as _diag_on

    if not isinstance(offloader._runner, NativeCotsRunner):
        return

    # Clear any leaked state from a prior offloader in the same
    # process. C++ side is per-CotsCpuInfer instance and starts
    # fresh (atomic-bool default = false).
    cots_ops.set_uva_ablation(False)

    ablate_hostfn = os.environ.get("VLLM_COTS_ABLATE_HOSTFN", "0") == "1"
    ablate_submit_hostfn = os.environ.get("VLLM_COTS_ABLATE_SUBMIT_HOSTFN", "0") == "1"
    ablate_sync_hostfn = os.environ.get("VLLM_COTS_ABLATE_SYNC_HOSTFN", "0") == "1"
    ablate_d2h = os.environ.get("VLLM_COTS_ABLATE_D2H", "0") == "1"
    ablate_uva = os.environ.get("VLLM_COTS_ABLATE_UVA", "0") == "1"
    any_active = (
        ablate_hostfn
        or ablate_submit_hostfn
        or ablate_sync_hostfn
        or ablate_d2h
        or ablate_uva
    )
    if not any_active:
        return

    if not (offloader.config.dry_run and _diag_on):
        raise RuntimeError(
            "[cots §1c.26/§1c.27] ablation env vars set "
            f"(HOSTFN={int(ablate_hostfn)}, "
            f"SUBMIT_HOSTFN={int(ablate_submit_hostfn)}, "
            f"SYNC_HOSTFN={int(ablate_sync_hostfn)}, "
            f"D2H={int(ablate_d2h)}, UVA={int(ablate_uva)}) but "
            f"gate not met: dry_run={offloader.config.dry_run}, "
            f"VLLM_COTS_DIAG={int(_diag_on)}. These flags will "
            "silently corrupt outputs in real (non-dryrun) mode "
            "and would produce a false measurement here. To "
            "enable: pass --cots-dry-run AND set VLLM_COTS_DIAG=1. "
            "To disable: unset the VLLM_COTS_ABLATE_* env vars."
        )

    infer = cots_ops._lookup_infer(
        offloader._runner._runner_id, "install_ablations_from_env"
    )
    infer.set_ablations(
        ablate_d2h=ablate_d2h,
        ablate_hostfn=ablate_hostfn,
        ablate_submit_hostfn=ablate_submit_hostfn,
        ablate_sync_hostfn=ablate_sync_hostfn,
    )
    cots_ops.set_uva_ablation(ablate_uva)
    logger.info(
        "[cots §1c.26/§1c.27] ablations active: HOSTFN=%d, "
        "SUBMIT_HOSTFN=%d, SYNC_HOSTFN=%d, D2H=%d, UVA=%d "
        "(probe-only; dryrun outputs are garbage)",
        int(ablate_hostfn),
        int(ablate_submit_hostfn),
        int(ablate_sync_hostfn),
        int(ablate_d2h),
        int(ablate_uva),
    )
