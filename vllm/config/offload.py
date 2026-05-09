# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for model weight offloading."""

import warnings
from typing import Literal

from pydantic import Field, model_validator

from vllm.config.utils import config

OffloadBackend = Literal["auto", "uva", "prefetch", "prefetch_defer", "cots"]


@config
class UVAOffloadConfig:
    """Configuration for UVA (Unified Virtual Addressing) CPU offloading.

    Uses zero-copy access from CPU-pinned memory. Simple but requires
    fast CPU-GPU interconnect.
    """

    cpu_offload_gb: float = Field(default=0, ge=0)
    """The space in GiB to offload to CPU, per GPU. Default is 0, which means
    no offloading. Intuitively, this argument can be seen as a virtual way to
    increase the GPU memory size. For example, if you have one 24 GB GPU and
    set this to 10, virtually you can think of it as a 34 GB GPU. Then you can
    load a 13B model with BF16 weight, which requires at least 26GB GPU memory.
    Note that this requires fast CPU-GPU interconnect, as part of the model is
    loaded from CPU memory to GPU memory on the fly in each model forward pass.
    This uses UVA (Unified Virtual Addressing) for zero-copy access.
    """

    cpu_offload_params: set[str] = Field(default_factory=set)
    """The set of parameter name segments to target for CPU offloading.
    Unmatched parameters are not offloaded. If this set is empty, parameters
    are offloaded non-selectively until the memory limit defined by
    `cpu_offload_gb` is reached.
    Examples:
        - For parameter name "mlp.experts.w2_weight":
            - "experts" or "experts.w2_weight" will match.
            - "expert" or "w2" will NOT match (must be exact segments).
    This allows distinguishing parameters like "w2_weight" and "w2_weight_scale".
    """


@config
class PrefetchOffloadConfig:
    """Configuration for prefetch-based CPU offloading.

    Groups layers and uses async H2D prefetch to hide transfer latency.
    """

    offload_group_size: int = Field(default=0, ge=0)
    """Group every N layers together. Offload last `offload_num_in_group`
    layers of each group. Default is 0 (disabled).
    Example: group_size=8, num_in_group=2 offloads layers 6,7,14,15,22,23,...
    Unlike cpu_offload_gb, this uses explicit async prefetching to hide transfer
    latency.
    """

    offload_num_in_group: int = Field(default=1, ge=1)
    """Number of layers to offload per group.
    Must be <= offload_group_size. Default is 1."""

    offload_prefetch_step: int = Field(default=1, ge=0)
    """Number of layers to prefetch ahead.
    Higher values hide more latency but use more GPU memory. Default is 1."""

    offload_params: set[str] = Field(default_factory=set)
    """The set of parameter name segments to target for prefetch offloading.
    Unmatched parameters are not offloaded. If this set is empty, ALL
    parameters of each offloaded layer are offloaded.
    Uses segment matching: "w13_weight" matches "mlp.experts.w13_weight"
    but not "mlp.experts.w13_weight_scale".
    """

    dry_run: bool = Field(default=False)
    """Diagnostic for `offload_backend="prefetch_defer"`: install wrappers
    (forward hooks, custom ops, stream/event sync) but skip the actual H2D
    copies. Token output is garbage; only host bookkeeping + stream/event cost
    is measured. The stock `prefetch` backend ignores this field."""


@config
class CotsOffloadConfig:
    """Configuration for COTS collaborative CPU-GPU offloading (thesis backend).

    Splits each WQKV / MLP1 / MLP2 weight along its tensor-parallel-native axis
    so a fraction `f_cpu_store` of the bytes lives in pinned CPU memory and is
    GEMM'd on the CPU each forward pass, in parallel with the GPU's compute on
    the GPU-resident slice. Activation returns from CPU use an SM-issued UVA
    copy kernel that bypasses the H2D copy engine. WO (`o_proj`) is NOT
    offloaded in Phase 1/2.

    See `David/Docs/implementation_roadmap.md` and `David/Docs/phase0_findings.md`
    for the full design and the empirical numbers that justify it.
    """

    f_cpu_store: float = Field(default=0.0, ge=0.0, le=1.0)
    """Fraction of WQKV / MLP1 / MLP2 weight bytes resident on CPU. Single
    uniform scalar applied to all three sub-modules (matched-index invariant
    between MLP1 col-parallel and MLP2 row-parallel is automatic under uniform
    dispatch). Phase 1a default 0.0 means no offload. Typical thesis value at
    7B B=1 decode is ~0.09 ("free" regime per phase0 §0.3.3)."""

    f_prefetch: float = Field(default=0.0, ge=0.0, le=1.0)
    """Manual fallback for the layer-ahead weight-prefetch fraction. Only
    consulted when the offloader is constructed without a
    `dispatch_table_factory`; the Planner's factory output overrides this
    value entirely. Constraint: f_prefetch <= f_cpu_store (prefetch consumes
    CPU-stored bytes; the f_cpu_compute = f_cpu_store - f_prefetch portion is
    CPU-computed). Default 0.0 reproduces Phase 1a behavior bit-exactly."""

    kv_biased: bool = Field(default=True)
    """If True, the WQKV column picker is biased toward K/V: K+V column groups
    are assigned to CPU before any Q columns. Preserves the suffix-cache layout
    and minimizes H2D contention with weight prefetch. See
    `weight_offload_design.md §WQKV Column Choice`. If False, columns are
    picked from the front of WQKV's output (Q first), useful only for
    ablation."""

    cpu_dtype: Literal["bfloat16"] = "bfloat16"
    """CPU weight dtype. Locked to BF16: phase0 §0.3.2 confirmed F.linear with
    BF16 weights uses oneDNN's optimized BF16 path (2x faster than FP32 at
    small batch on AVX2 hardware), while torch.mm with BF16 falls back to a
    naive scalar path that is 100x slower."""

    cpu_num_threads: int = Field(default=16, ge=1)
    """PyTorch CPU intra-op thread count. Scalar fallback when
    `cpu_num_threads_by_bucket` is unset (Phase 1c will add per-bucket
    policy at Stage 4). See `phase1a_findings.md §1.13b`."""

    cpu_runner: Literal["native", "python"] = "python"
    """Phase 1c kill-switch. "native" routes CPU work through
    `cudaLaunchHostFunc` + a C++ `TaskQueue` worker (the production
    Phase 1c substrate; supports CUDA Graph capture once
    `enforce_eager=False`). "python" keeps the Phase 1a/1b
    `ThreadPoolExecutor` substrate for A/B diagnostics — that path is
    NOT graph-capturable, so `cpu_runner="python"` requires
    `enforce_eager=True` and will be rejected at engine launch
    otherwise. The default flips to "native" at Phase 1c Stage 5
    once graph capture is verified end-to-end; until then it stays
    "python" so existing Phase 1a/1b workflows are unchanged. Slated
    for deprecation one quarter post-Phase-1c. See
    `David/Docs/implementation_roadmap.md` Phase 1c and the approved
    plan at /root/.claude/plans/pleaes-implement-phase1c-in-quizzical-mist.md."""

    dry_run: bool = Field(default=False)
    """Diagnostic: install all wrappers but skip the CPU GEMM in the worker.
    Used by `bench_cots_dryrun_vs_none.py` to attribute the COTS-vs-none gap
    into orchestration vs active CPU-work penalty (`phase1a_findings.md §1.14`).
    Token output is garbage; only host bookkeeping cost is measured. Permanent
    diagnostic — useful for verifying Phase 1c collapsed the orchestration
    column and for catching future regressions in the COTS host path."""


@config
class OffloadConfig:
    """Configuration for model weight offloading to reduce GPU memory usage."""

    offload_backend: OffloadBackend = "auto"
    """The backend for weight offloading. Options:
    - "auto": Selects based on which sub-config has non-default values
      (prefetch if offload_group_size > 0, uva if cpu_offload_gb > 0).
    - "uva": UVA (Unified Virtual Addressing) zero-copy offloading.
    - "prefetch": Stock async prefetch with group-based layer offloading.
    - "prefetch_defer": Thesis native-prefetch baseline with deferred
      wrap-around scheduling.
    - "cots": Collaborative CPU-GPU offloading (thesis backend). Must be set
      explicitly; "auto" never selects "cots".
    """

    uva: UVAOffloadConfig = Field(default_factory=UVAOffloadConfig)
    """Parameters for UVA offloading backend."""

    prefetch: PrefetchOffloadConfig = Field(default_factory=PrefetchOffloadConfig)
    """Parameters for prefetch offloading backend."""

    cots: CotsOffloadConfig = Field(default_factory=CotsOffloadConfig)
    """Parameters for COTS collaborative CPU-GPU offloading backend."""

    @model_validator(mode="after")
    def validate_offload_config(self) -> "OffloadConfig":
        """Validate offload configuration constraints."""
        prefetch_backend = self.offload_backend in ("prefetch", "prefetch_defer")
        if prefetch_backend or self.prefetch.offload_group_size > 0:
            if self.prefetch.offload_num_in_group > self.prefetch.offload_group_size:
                raise ValueError(
                    f"offload_num_in_group ({self.prefetch.offload_num_in_group})"
                    f" must be <= offload_group_size"
                    f" ({self.prefetch.offload_group_size})"
                )
            if self.prefetch.offload_prefetch_step < 1:
                raise ValueError(
                    f"offload_prefetch_step"
                    f" ({self.prefetch.offload_prefetch_step})"
                    f" must be >= 1 when prefetch offloading is enabled"
                    f" (offload_group_size > 0)"
                )

        # Warn if both backends have non-default values
        uva_active = self.uva.cpu_offload_gb > 0
        prefetch_active = self.prefetch.offload_group_size > 0
        cots_active = self.cots.f_cpu_store > 0
        if self.offload_backend == "uva" and prefetch_active:
            warnings.warn(
                "Prefetch offload fields are set but offload_backend='uva'. "
                "Prefetch settings will be ignored.",
                stacklevel=2,
            )
        elif prefetch_backend and uva_active:
            warnings.warn(
                "UVA offload fields are set but offload_backend="
                f"'{self.offload_backend}'. UVA settings will be ignored.",
                stacklevel=2,
            )
        elif self.offload_backend == "prefetch" and self.prefetch.dry_run:
            warnings.warn(
                "prefetch.dry_run is set but offload_backend='prefetch'. "
                "The stock prefetch backend ignores dry_run; use "
                "offload_backend='prefetch_defer' for the diagnostic.",
                stacklevel=2,
            )
        elif self.offload_backend == "auto" and uva_active and prefetch_active:
            warnings.warn(
                "Both UVA and prefetch offload fields are set with "
                "offload_backend='auto'. Prefetch backend will be selected. "
                "Set offload_backend explicitly to suppress this warning.",
                stacklevel=2,
            )
        elif (
            self.offload_backend == "auto" and prefetch_active and self.prefetch.dry_run
        ):
            warnings.warn(
                "prefetch.dry_run is set with offload_backend='auto'. Auto "
                "selects the stock prefetch backend, which ignores dry_run; "
                "use offload_backend='prefetch_defer' for the diagnostic.",
                stacklevel=2,
            )

        if self.offload_backend == "cots":
            if uva_active:
                raise ValueError(
                    "offload_backend='cots' is incompatible with non-zero "
                    "uva.cpu_offload_gb. Disable UVA when using cots."
                )
            if prefetch_active:
                raise ValueError(
                    "offload_backend='cots' is incompatible with non-zero "
                    "prefetch.offload_group_size. Disable prefetch offload "
                    "when using cots."
                )
            if self.cots.f_prefetch > self.cots.f_cpu_store:
                raise ValueError(
                    f"cots.f_prefetch ({self.cots.f_prefetch}) must be <= "
                    f"cots.f_cpu_store ({self.cots.f_cpu_store}); prefetch "
                    f"consumes CPU-stored bytes."
                )
        elif cots_active and self.offload_backend != "cots":
            warnings.warn(
                "cots.f_cpu_store is set but offload_backend is "
                f"'{self.offload_backend}', not 'cots'. cots settings will "
                "be ignored. Pass --offload-backend cots to enable.",
                stacklevel=2,
            )
        return self

    def compute_hash(self) -> str:
        """
        Provide a hash that uniquely identifies all the offload configs.

        All fields are included because PrefetchOffloader patches module
        forwards and inserts custom ops (wait_prefetch, start_prefetch)
        into the computation graph. Changing any offload setting can
        alter which layers are hooked and how prefetch indices are
        computed, so the compilation cache must distinguish them.
        """
        from vllm.config.utils import get_hash_factors, hash_factors

        factors = get_hash_factors(self, ignored_factors=set())
        hash_str = hash_factors(factors)
        return hash_str
