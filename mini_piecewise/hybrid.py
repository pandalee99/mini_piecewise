from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
import torch.fx as fx

from .backends import CaptureBackend
from .config import PiecePolicy, PieceSelector, PiecewiseHybridConfig
from .errors import CaptureNotPerformedError, CudaNotAvailableError, ShapeOutOfRangeError
from .fx_split import split_graph_by_attention

logger = logging.getLogger("mini_piecewise")


@dataclass
class _RecordedCall:
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


class _Recorder(torch.nn.Module):
    """Temporary wrapper that records per-piece inputs during a profiling pass."""

    def __init__(self, mod: torch.nn.Module, sink: dict[str, _RecordedCall], name: str):
        super().__init__()
        self._mod = mod
        self._sink = sink
        self._name = name

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        self._sink[self._name] = _RecordedCall(args=args, kwargs=dict(kwargs))
        return self._mod(*args, **kwargs)


class PiecewiseHybridModel(torch.nn.Module):
    """Stitched FX GraphModule with per-piece backend dispatch.

    After capture, pieces assigned PiecePolicy.CAPTURE are replaced by
    their respective backend instances, PiecePolicy.EAGER pieces remain
    as-is, and PiecePolicy.SKIP pieces are left in their original state.
    The stitched GraphModule is executed under inference_mode once installed.
    """

    def __init__(
        self,
        split_gm: fx.GraphModule,
        items: list,
        config: PiecewiseHybridConfig,
        *,
        example_inputs_fn: Callable[[int], Any],
        device: Optional[torch.device] = None,
        graph_pool: Any = None,
    ) -> None:
        super().__init__()
        self._split_gm = split_gm
        self._items = items
        self._config = config
        self._example_inputs_fn = example_inputs_fn

        if not torch.cuda.is_available():
            raise CudaNotAvailableError("CUDA is required for PiecewiseHybridModel")

        self._device = device or torch.device("cuda")
        self._graph_pool = graph_pool if graph_pool is not None else torch.cuda.graph_pool_handle()

        # Keep original piece modules.
        self._original: dict[str, torch.nn.Module] = {}
        for it in self._items:
            self._original[it.submod_name] = split_gm.get_submodule(it.submod_name)

        # Built after capture.
        self._backends: dict[str, CaptureBackend] = {}
        self._installed: bool = False

    @property
    def split_gm(self) -> fx.GraphModule:
        return self._split_gm

    @property
    def items(self) -> list:
        return list(self._items)

    def _normalize_example_inputs(self, ex: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:
        if isinstance(ex, tuple) and len(ex) == 2 and isinstance(ex[1], dict):
            args, kwargs = ex
            if not isinstance(args, tuple):
                args = tuple(args)
            return args, dict(kwargs)
        if isinstance(ex, (list, tuple)):
            return tuple(ex), {}
        return (ex,), {}

    def _ensure_cuda(self, tree: Any) -> None:
        def _walk(x: Any) -> None:
            if isinstance(x, torch.Tensor):
                if not x.is_cuda:
                    raise ValueError("example inputs must be CUDA tensors")
                return
            if isinstance(x, dict):
                for v in x.values():
                    _walk(v)
            elif isinstance(x, (list, tuple)):
                for v in x:
                    _walk(v)

        _walk(tree)

    def capture(self) -> None:
        """Capture backends for CAPTURE-policy pieces across all bucket sizes.

        EAGER pieces remain in their original form. SKIP pieces are
        not optimized but kept for graph connectivity.
        """
        piece_selector = self._config._effective_piece_selector()

        sizes = list(self._config.capture_sizes)
        sizes.sort(reverse=True)

        logger.info("Starting capture for %d sizes: %s", len(sizes), sizes)

        # Capture from large to small to improve memory pool reuse.
        for static_size in sizes:
            logger.debug("Capturing static_size=%d", static_size)
            ex = self._example_inputs_fn(static_size)
            args, kwargs = self._normalize_example_inputs(ex)
            self._ensure_cuda((args, kwargs))

            # Install recorders temporarily.
            recorded: dict[str, _RecordedCall] = {}
            for it in self._items:
                if it.policy == PiecePolicy.SKIP:
                    continue
                name = it.submod_name
                orig = self._split_gm.get_submodule(name)
                setattr(self._split_gm, name, _Recorder(orig, recorded, name))

            # Run once to collect per-piece inputs.
            with torch.inference_mode():
                _ = self._split_gm(*args, **kwargs)

            # Restore originals.
            for it in self._items:
                if it.policy == PiecePolicy.SKIP:
                    continue
                name = it.submod_name
                setattr(self._split_gm, name, self._original[name])

            # Capture for CAPTURE-policy pieces.
            for it in self._items:
                if it.policy != PiecePolicy.CAPTURE:
                    continue
                call = recorded.get(it.submod_name)
                if call is None:
                    raise RuntimeError(f"Did not record inputs for {it.submod_name}")

                backend = self._backends.get(it.submod_name)
                if backend is None:
                    backend = self._config.backend_factory(
                        self._original[it.submod_name],
                        self._config,
                        graph_pool=self._graph_pool,
                        device=self._device,
                    )
                    self._backends[it.submod_name] = backend

                backend.capture_from_recorded_inputs(
                    static_size=static_size,
                    recorded_args=call.args,
                    recorded_kwargs=call.kwargs,
                    runtime_size=static_size,
                )
                logger.debug("Captured %s for static_size=%d", it.submod_name, static_size)

        # Install backends into stitched graph.
        for it in self._items:
            if it.policy == PiecePolicy.SKIP:
                # Replace skip pieces with a passthrough (identity).
                # This is tricky with FX graphs; for now, keep the original
                # but mark it as skipped. Users should handle SKIP at model level.
                setattr(self._split_gm, it.submod_name, self._original[it.submod_name])
            elif it.policy == PiecePolicy.EAGER:
                setattr(self._split_gm, it.submod_name, self._original[it.submod_name])
            else:
                setattr(self._split_gm, it.submod_name, self._backends[it.submod_name])

        self._installed = True
        logger.info("Capture complete. %d pieces: %d captured, %d eager, %d skip",
                     len(self._items),
                     sum(1 for it in self._items if it.policy == PiecePolicy.CAPTURE),
                     sum(1 for it in self._items if it.policy == PiecePolicy.EAGER),
                     sum(1 for it in self._items if it.policy == PiecePolicy.SKIP))

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if not self._installed:
            return self._split_gm(*args, **kwargs)
        with torch.inference_mode():
            return self._split_gm(*args, **kwargs)

    def summary(self) -> dict[str, Any]:
        """Return diagnostic summary of the hybrid model."""
        piece_info = []
        for it in self._items:
            info = {
                "name": it.submod_name,
                "policy": it.policy.value,
            }
            if it.policy == PiecePolicy.CAPTURE and it.submod_name in self._backends:
                backend = self._backends[it.submod_name]
                if hasattr(backend, "info"):
                    info["backend"] = backend.info()
            piece_info.append(info)

        return {
            "num_pieces": len(self._items),
            "installed": self._installed,
            "capture_sizes": list(self._config.capture_sizes),
            "pieces": piece_info,
        }

    def recapture(self, new_sizes: list[int] | None = None) -> None:
        """Re-capture with potentially different bucket sizes.

        Args:
            new_sizes: New capture sizes. If None, re-capture with existing sizes.
        """
        # Free existing backends
        self.free()

        if new_sizes is not None:
            from .config import PiecewiseHybridConfig
            self._config = PiecewiseHybridConfig(
                capture_sizes=tuple(sorted(new_sizes)),
                warmup_iters=self._config.warmup_iters,
                zero_pad_inputs=self._config.zero_pad_inputs,
                runtime_size_fn=self._config.runtime_size_fn,
                piece_selector=self._config.piece_selector,
                backend_factory=self._config.backend_factory,
                check_input_addresses=self._config.check_input_addresses,
            )

        # Re-capture
        self.capture()

    def free(self) -> None:
        """Release all captured backend resources."""
        for name, backend in self._backends.items():
            if hasattr(backend, "free"):
                backend.free()
            # Restore original module
            setattr(self._split_gm, name, self._original[name])

        self._backends.clear()
        self._installed = False
        logger.info("Freed all captured resources")


def _trace_to_fx(
    model: torch.nn.Module,
    *,
    example_args: tuple[Any, ...],
    example_kwargs: dict[str, Any],
    piece_selector: PieceSelector,
) -> fx.GraphModule:
    """Trace a module to FX.

    Preference order:
    1) symbolic_trace (keeps call_module boundaries)
    2) proxy-tensor make_fx (fallback)
    """

    # Derive is_leaf_module from piece_selector for FX tracing.
    # Modules with EAGER policy should be treated as leaves so they
    # appear as single call_module nodes that can be isolated.
    def _is_leaf(m: torch.nn.Module, qualified_name: str) -> bool:
        policy = piece_selector(m, qualified_name)
        if policy == PiecePolicy.EAGER:
            return True
        return False

    class _Tracer(fx.Tracer):
        def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
            try:
                if _is_leaf(m, module_qualified_name):
                    return True
            except Exception:
                pass
            return super().is_leaf_module(m, module_qualified_name)

    try:
        tracer = _Tracer()
        graph = tracer.trace(model)
        gm = fx.GraphModule(model, graph)
        assert isinstance(gm, fx.GraphModule)
        return gm
    except Exception:
        try:
            from torch.fx.experimental.proxy_tensor import make_fx

            gm = make_fx(model)(*example_args, **example_kwargs)
            assert isinstance(gm, fx.GraphModule)
            return gm
        except Exception as e:
            raise RuntimeError("Failed to trace model to FX") from e


def make_piecewise_hybrid_model(
    model: torch.nn.Module,
    config: PiecewiseHybridConfig,
    *,
    example_inputs_fn: Callable[[int], Any],
    device: Optional[torch.device] = None,
    graph_pool: Any = None,
) -> PiecewiseHybridModel:
    """Build a piecewise hybrid model using FX tracing and splitting.

    Traces the model to FX, splits the graph according to the configured
    piece_selector policy, and returns a PiecewiseHybridModel that can
    capture CAPTURE-policy pieces with the configured backend factory.
    """
    piece_selector = config._effective_piece_selector()

    max_size = config.capture_sizes[-1]
    ex = example_inputs_fn(max_size)

    # Normalize (args, kwargs).
    if isinstance(ex, tuple) and len(ex) == 2 and isinstance(ex[1], dict):
        example_args, example_kwargs = ex
        if not isinstance(example_args, tuple):
            example_args = tuple(example_args)
        example_kwargs = dict(example_kwargs)
    elif isinstance(ex, (list, tuple)):
        example_args, example_kwargs = tuple(ex), {}
    else:
        example_args, example_kwargs = (ex,), {}

    gm = _trace_to_fx(
        model,
        example_args=example_args,
        example_kwargs=example_kwargs,
        piece_selector=piece_selector,
    )

    split_gm, items = split_graph_by_attention(gm, piece_selector=piece_selector)

    return PiecewiseHybridModel(
        split_gm,
        items,
        config,
        example_inputs_fn=example_inputs_fn,
        device=device,
        graph_pool=graph_pool,
    )