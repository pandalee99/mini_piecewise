from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
import torch.fx as fx

from .config import PiecewiseHybridConfig
from .cudagraph_backend import CUDAGraphPiece
from .errors import CudaNotAvailableError
from .fx_split import SplitItem, split_graph_by_attention


@dataclass
class _RecordedCall:
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


class _Recorder(torch.nn.Module):
    """A small wrapper to record args/kwargs for a piece call."""

    def __init__(self, mod: torch.nn.Module, sink: dict[str, _RecordedCall], name: str):
        super().__init__()
        self._mod = mod
        self._sink = sink
        self._name = name

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        # Record references (tensors are ok; we use them only for shapes/dtypes/devices).
        self._sink[self._name] = _RecordedCall(args=args, kwargs=dict(kwargs))
        return self._mod(*args, **kwargs)


class PiecewiseHybridModel(torch.nn.Module):
    """A stitched FX GraphModule whose pieces are a mix of eager and CUDAGraph replay."""

    def __init__(
        self,
        split_gm: fx.GraphModule,
        items: list[SplitItem],
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
        for it in items:
            self._original[it.submod_name] = split_gm.get_submodule(it.submod_name)

        # Built after capture.
        self._backends: dict[str, CUDAGraphPiece] = {}
        self._installed: bool = False

    @property
    def split_gm(self) -> fx.GraphModule:
        return self._split_gm

    @property
    def items(self) -> list[SplitItem]:
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
        # Minimal: check only top-level args/kwargs leaves.
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
        """Capture CUDA graphs for non-attention pieces across all buckets."""

        sizes = list(self._config.capture_sizes)
        sizes.sort(reverse=True)

        # Capture from large to small to improve memory pool reuse.
        for static_size in sizes:
            ex = self._example_inputs_fn(static_size)
            args, kwargs = self._normalize_example_inputs(ex)
            self._ensure_cuda((args, kwargs))

            # Install recorders temporarily.
            recorded: dict[str, _RecordedCall] = {}
            for it in self._items:
                name = it.submod_name
                orig = self._split_gm.get_submodule(name)
                setattr(self._split_gm, name, _Recorder(orig, recorded, name))

            # Run once to collect per-piece inputs.
            with torch.inference_mode():
                _ = self._split_gm(*args, **kwargs)

            # Restore originals.
            for it in self._items:
                name = it.submod_name
                setattr(self._split_gm, name, self._original[name])

            # Capture for non-attention pieces.
            for it in self._items:
                if it.is_attention_piece:
                    continue
                call = recorded.get(it.submod_name)
                if call is None:
                    raise RuntimeError(f"Did not record inputs for {it.submod_name}")

                backend = self._backends.get(it.submod_name)
                if backend is None:
                    backend = CUDAGraphPiece(
                        self._original[it.submod_name],
                        capture_sizes=self._config.capture_sizes,
                        warmup_iters=self._config.warmup_iters,
                        zero_pad_inputs=self._config.zero_pad_inputs,
                        runtime_size_fn=self._config.runtime_size_fn,
                        check_input_addresses=self._config.check_input_addresses,
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

        # Install backends into stitched graph.
        for it in self._items:
            if it.is_attention_piece:
                setattr(self._split_gm, it.submod_name, self._original[it.submod_name])
            else:
                setattr(self._split_gm, it.submod_name, self._backends[it.submod_name])

        self._installed = True

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if not self._installed:
            # Allow eager execution before capture for debugging.
            return self._split_gm(*args, **kwargs)
        # IMPORTANT: During capture and CUDAGraphPiece replay we operate under
        # torch.inference_mode(), which produces inference tensors.
        # Mixing inference tensors with grad-enabled eager ops can trigger:
        # "Inference tensors cannot be saved for backward".
        # This minimal implementation is intended for inference, so we run the
        # stitched graph under inference_mode once installed.
        with torch.inference_mode():
            return self._split_gm(*args, **kwargs)


def _trace_to_fx(
    model: torch.nn.Module,
    *,
    example_args: tuple[Any, ...],
    example_kwargs: dict[str, Any],
    is_attention_module: Callable[[torch.nn.Module, str], bool],
) -> fx.GraphModule:
    """Trace a module to FX.

    Preference order:
    1) symbolic_trace (keeps call_module boundaries; simpler to split by module type)
    2) proxy-tensor make_fx (fallback; may decompose module calls into call_function)
    """

    class _Tracer(fx.Tracer):
        def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:  # type: ignore[override]
            # Treat attention modules as leaves so they show up as a single call_module node
            # and can be isolated by the splitter.
            try:
                if is_attention_module(m, module_qualified_name):
                    return True
            except Exception:
                # If the predicate fails, fall back to default behavior.
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
            from torch.fx.experimental.proxy_tensor import make_fx  # type: ignore

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
    """Build a minimal FX-split hybrid model.

    Steps:
    - Trace model to FX using a representative example input (max capture size).
    - Split graph to isolate attention nodes into their own pieces.
    - Return a wrapper that can capture non-attention pieces with CUDA Graph.
    """

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
        is_attention_module=config.is_attention_module,
    )

    split_gm, items = split_graph_by_attention(gm, is_attention_module=config.is_attention_module)

    return PiecewiseHybridModel(
        split_gm,
        items,
        config,
        example_inputs_fn=example_inputs_fn,
        device=device,
        graph_pool=graph_pool,
    )
