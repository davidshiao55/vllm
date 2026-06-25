# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for model weight offloading."""

import warnings
from typing import Literal

from pydantic import Field, field_validator, model_validator

from vllm.config.cots import (
    DEFAULT_COTS_WEIGHT_MODULES,
    CotsWeightModule,
    normalize_cots_weight_modules,
)
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

    Splits selected decoder sub-modules along their COTS policy axis so a
    fraction `f_cpu_store` of the bytes lives in pinned CPU memory and is
    GEMM'd on the CPU each forward pass, in parallel with the GPU's compute on
    the GPU-resident slice. Activation returns from CPU use an SM-issued UVA
    copy kernel that bypasses the H2D copy engine. The production module set
    covers WQKV, MLP, and WO with a single storage fraction and fixed snapping
    policy.

    See `docs/implementation_roadmap.md` and `docs/phase0_findings.md`
    for the full design and the empirical numbers that justify it.
    """

    f_cpu_store: float = Field(default=0.0, ge=0.0, le=1.0)
    """Fraction of enabled COTS weight bytes resident on CPU. Single uniform
    scalar applied to the selected module set (default WQKV / MLP1 / MLP2 /
    WO). The matched-index invariant between MLP1 col-parallel and
    MLP2 row-parallel is automatic under uniform dispatch. Default 0.0 means no
    offload. Typical thesis value at 7B B=1 decode is ~0.09 ("free" regime per
    phase0 §0.3.3)."""

    f_prefetch: float = Field(default=0.0, ge=0.0, le=1.0)
    """Manual fallback for the layer-ahead weight-prefetch fraction. Only
    consulted when the offloader is constructed without a
    `dispatch_table`/factory; the Planner's table output overrides this
    value entirely. Constraint: f_prefetch <= f_cpu_store (prefetch consumes
    CPU-stored bytes; the f_cpu_compute = f_cpu_store - f_prefetch portion is
    CPU-computed). Default 0.0 keeps the CPU-stored slice CPU-computed."""

    dispatch_table: dict[int, tuple[float, float]] | None = Field(default=None)
    """Optional engine-local COTS compute dispatch table emitted by the
    Planner. Keys are vLLM `BatchDescriptor.num_tokens` bucket values. Values
    are `(f_cpu_compute, f_prefetch_compute)` for that bucket. When set, this
    table overrides the uniform `f_prefetch` fallback above. Runtime validates
    that all COTS dispatch buckets are present before installing slabs. In
    graph mode, every CUDA graph capture bucket must also have a matching
    dispatch row.

    This is an engine-local interface: FastTTS/global planning decides the
    storage budget and may provide a table; vLLM still owns snapping, bucket
    validation, and tensor geometry."""

    weight_modules: set[CotsWeightModule | str] = Field(
        default_factory=lambda: set(DEFAULT_COTS_WEIGHT_MODULES)
    )
    """COTS weight sub-modules to offload. Valid entries:
    * `"qkv"`: WQKV output split using the WQKV-specific KV-biased picker.
    * `"mlp"`: fused MLP block (`gate_up_proj` col split + `down_proj` row
      split) with matched intermediate indices.
    * `"wo"`: WO (`o_proj`) dense output split using a larger production snap
      quantum so the first CPU-compute slice amortizes WO's extra per-layer
      task/sync cost."""

    kv_split_blocks: int = Field(default=0, ge=0)
    """Phase 2 hybrid-KV split point in KV-cache blocks. Blocks before this
    point remain GPU-resident and blocks at/after this point belong to the CPU
    suffix pool. Default 0 disables the hybrid-KV path."""

    kv_cpu_pool_bytes: int = Field(default=0, ge=0)
    """Phase 2 CPU suffix KV pool size in bytes. This is planner-owned
    capacity, distinct from vLLM's native prefix-cache KV offload feature."""

    f_cpu_kv_store: float = Field(default=0.0, ge=0.0, le=1.0)
    """Experimental TP-style KV head-split storage fraction.

    When ``kv_mode='head_split'``, this fraction is snapped down to whole GQA
    KV-head groups and the last groups are owned by the CPU path. This is a
    separate redesign knob from the Phase 2 prefix/suffix CPU pool; it does not
    use ``kv_split_blocks`` or ``kv_cpu_pool_bytes``."""

    kv_mode: Literal["prefix_suffix", "head_split"] = "prefix_suffix"
    """Experimental COTS KV topology selector.

    * `"prefix_suffix"` keeps the Phase 2 position-split implementation:
      GPU prefix KV, CPU suffix KV, and online-softmax merge.
    * `"head_split"` enables the TP-style GQA-group redesign path. CPU KV
      ownership is controlled by ``f_cpu_kv_store`` and snaps to whole GQA
      groups."""

    kv_h2d_mode: Literal["uva"] = "uva"
    """Phase 2 CPU->GPU artifact path. MVP supports only UVA so small CPU
    attention outputs/LSE and CPU-produced prefix K/V do not use the explicit
    weight-prefetch H2D copy path."""

    cpu_dtype: Literal["bfloat16"] = "bfloat16"
    """CPU weight dtype. Locked to BF16: phase0 §0.3.2 confirmed F.linear with
    BF16 weights uses oneDNN's optimized BF16 path (2x faster than FP32 at
    small batch on AVX2 hardware), while torch.mm with BF16 falls back to a
    naive scalar path that is 100x slower."""

    cpu_num_threads: int = Field(default=16, ge=1)
    """PyTorch CPU intra-op thread count. Scalar fallback when
    `cpu_num_threads_by_bucket` is unset. See `phase1a_findings.md §1.13b`."""

    cpu_num_threads_by_bucket: dict[int, int] | None = Field(default=None)
    """Per-`BatchDescriptor` thread count for the native CPU
    GEMM worker. Keys are COTS dispatch bucket values; values are >= 1. When
    unset, every bucket uses the scalar `cpu_num_threads`.

    This is a runtime policy hook, not an additional Planner search axis.
    The Profiler/Planner may provide the map explicitly, or derive it from
    the chosen weight dispatch table using a calibrated work-score policy
    such as `bucket * f_cpu_compute`. The Planner should model costs after
    this policy is applied rather than sweep thread count independently.

    Only consulted by `cpu_runner='native'`; the Python runner uses the
    process-wide `torch.set_num_threads(cpu_num_threads)` set once at
    offloader init. Per-bucket policy on the Python path would race
    other threads in the process holding torch ops."""

    cpu_worker_affinity: list[int] | None = Field(default=None)
    """Optional CPU affinity mask for the native
    runner's TaskQueue worker thread. List of CPU IDs to pin the
    worker to, or None (default) for no opinion. The C++ implementation
    intersects this with the process's existing `sched_getaffinity` mask
    and warns-and-skips on empty intersection.

    Recommended on i9-14900KF: P-cores 1..7 (i.e., `[1, 2, 3, 4, 5, 6,
    7]`) — keeps the worker off P-core 0 where the main thread / CUDA
    dispatch / kernel tend to land. Hardware-specific; left as None by
    default so we don't bake i9-14900KF assumptions into the config.

    Only consulted by `cpu_runner='native'`."""

    cpu_runner: Literal["native", "python"] = "native"
    """COTS CPU runner selector.

    * `"native"` (default): CPU work runs on a C++ `TaskQueue`
      worker; submit/sync go through `cudaLaunchHostFunc` host
      callbacks. Supports both eager mode and graph capture
      (`enforce_eager=False`). With `auto_graph_split=True`, graph
      mode defaults to the measured fast split path; legacy full
      capture remains available for A/B diagnostics via
      `--no-cots-auto-graph-split`.
    * `"python"` (kill-switch): keeps the Phase 1a/1b
      `ThreadPoolExecutor` substrate for A/B diagnostics. NOT
      graph-capturable; `cpu_runner='python'` requires
      `enforce_eager=True` and is rejected at engine launch
      otherwise.

    See `docs/implementation_roadmap.md` Phase 1c and
    `docs/phase1c_findings.md` for the capture-gap
    measurements that motivate the split-graph default."""

    dry_run: bool = Field(default=False)
    """Diagnostic: install COTS wrappers and preserve bucket/slot/graph
    control flow, but skip active offloaded work from both paths. CPU-compute
    contribution is omitted; prefetch H2D and prefetched-slice GPU compute are
    omitted. Token output is garbage; tensor shapes remain valid. Permanent
    diagnostic for measuring the COTS control-plane floor."""

    auto_graph_split: bool = Field(default=True)
    """When COTS runs with CUDA graphs (`enforce_eager=False`), default to
    the measured fast graph policy.

    Phase 1 weight offload uses piecewise CUDA graphs, COTS weight
    submit/sync split points, and `wait_kernel` weight sync. Phase 2 hybrid KV
    uses piecewise CUDA graphs through the normal attention split points and
    does not add Phase 1 weight split points unless weight offload is also
    active. Disable this with `--no-cots-auto-graph-split` to reproduce
    full-capture or host-callback weight-capture experiments explicitly."""

    weight_capture_sync_mode: Literal["host_callback", "wait_kernel"] = "host_callback"
    """Phase 1 weight-offload sync mechanism for the captured-replay path.

    * `"host_callback"` (field default / eager fallback): the
      captured graph node is `cudaLaunchHostFunc(SyncCallback)`
      which blocks the CUDA driver thread on
      `TaskQueue::sync(0)`. This is the host-callback measured-baseline
      mechanism. With `auto_graph_split=True`, native COTS weight graph
      mode upgrades this to `"wait_kernel"` unless explicitly
      opting out of the auto graph policy.
    * `"wait_kernel"`: the captured node
      is a custom GPU wait kernel that spins on a host-mapped
      pinned `done_slot` written by the CPU worker. Replaces the
      captured `cudaLaunchHostFunc(sync_cb)` only. Submit side
      (dispatch_cb host_fn) is unchanged in both modes so CPU
      GEMM still overlaps with GPU GEMM. By itself on the legacy
      full-capture path, this did not beat native eager; paired
      with the COTS split-graph default it is the fastest measured
      capture path on the focused Qwen2.5-7B grid.
    Hard-fail safety gates (in `CotsOffloader.post_init`) for
    `wait_kernel` mode:
    * Requires `cpu_runner='native'` — Python runner has no
      slabs / worker thread / host-mapped `done_slot`.
    * Requires `enforce_eager=False` — kernel only makes sense
      as a captured node.
    * Requires CUDA + `_cots_C` extension built.

    This knob controls only the Phase 1 weight runner. Phase 2 hybrid KV
    suffix attention has its own prepared suffix runner and wait-kernel sync
    path; graph mode for hybrid KV uses piecewise attention boundaries, not
    the Phase 1 weight custom-op split points. Full-capture weight modes
    remain available by disabling `auto_graph_split`.
    """

    @field_validator("weight_modules", mode="before")
    @classmethod
    def normalize_weight_modules_field(cls, value: object) -> set[str]:
        return normalize_cots_weight_modules(value)

    @property
    def hybrid_kv_enabled(self) -> bool:
        """Whether the Phase 2 hybrid CPU-suffix KV path is configured."""
        return (
            self.kv_mode == "prefix_suffix"
            and self.kv_split_blocks > 0
            and self.kv_cpu_pool_bytes > 0
        )

    @property
    def head_split_kv_enabled(self) -> bool:
        """Whether the experimental TP-style KV head split is configured."""
        return self.kv_mode == "head_split" and self.f_cpu_kv_store > 0

    @property
    def cots_kv_enabled(self) -> bool:
        """Whether any COTS KV offload path is configured."""
        return self.hybrid_kv_enabled or self.head_split_kv_enabled

    @model_validator(mode="after")
    def validate_cots_config(self) -> "CotsOffloadConfig":
        """Validate COTS storage/dispatch invariants."""
        self.weight_modules = normalize_cots_weight_modules(self.weight_modules)
        if self.f_prefetch > self.f_cpu_store:
            raise ValueError(
                f"cots.f_prefetch ({self.f_prefetch}) must be <= "
                f"cots.f_cpu_store ({self.f_cpu_store}); prefetch consumes "
                "CPU-stored bytes."
            )

        def validate_table(label: str, table: dict[int, tuple[float, float]]) -> None:
            for bucket, entry in table.items():
                if bucket <= 0:
                    raise ValueError(
                        f"{label} bucket keys must be positive; got {bucket}"
                    )
                if len(entry) != 2:
                    raise ValueError(
                        f"{label} values must be (f_cpu_compute, f_prefetch_compute)"
                    )
                f_cpu_compute, f_prefetch_compute = entry
                if f_cpu_compute < 0 or f_prefetch_compute < 0:
                    raise ValueError(
                        f"{label} fractions must be non-negative; "
                        f"got {entry} for bucket {bucket}"
                    )
                total = f_cpu_compute + f_prefetch_compute
                if total > self.f_cpu_store + 1e-9:
                    raise ValueError(
                        f"{label} entry exceeds f_cpu_store: "
                        f"bucket={bucket}, entry={entry}, "
                        f"f_cpu_store={self.f_cpu_store}"
                    )
                if total < self.f_cpu_store - 1e-9:
                    raise ValueError(
                        f"{label} entry must sum to f_cpu_store: "
                        f"bucket={bucket}, entry={entry}, "
                        f"f_cpu_store={self.f_cpu_store}"
                    )

        if self.dispatch_table is not None:
            validate_table("cots.dispatch_table", self.dispatch_table)

        if self.kv_mode == "prefix_suffix" and self.f_cpu_kv_store > 0:
            raise ValueError(
                "cots.f_cpu_kv_store applies only when cots.kv_mode='head_split'."
            )
        if self.kv_mode == "head_split" and (
            self.kv_split_blocks > 0 or self.kv_cpu_pool_bytes > 0
        ):
            raise ValueError(
                "cots.kv_split_blocks/cots.kv_cpu_pool_bytes apply only to "
                "cots.kv_mode='prefix_suffix'. Use cots.f_cpu_kv_store for "
                "head_split KV."
            )

        return self


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
        cots_active = self.cots.f_cpu_store > 0 or self.cots.cots_kv_enabled
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
                "COTS settings are set but offload_backend is "
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
