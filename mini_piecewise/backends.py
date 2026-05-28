"""Backend abstraction for piece capture/replay.

Defines the CaptureBackend protocol that all optimization backends must
implement. Provides the default CUDAGraphPiece backend factory. Custom
backends (e.g., torch.compile, ONNX) can be plugged in by implementing
the protocol and registering a factory via PiecewiseHybridConfig.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable

import torch

from .cudagraph_backend import CUDAGraphPiece


@runtime_checkable
class CaptureBackend(Protocol):
    """Protocol for piece capture/replay backends.

    Any object implementing capture_from_recorded_inputs() and forward()
    satisfies this protocol and can serve as a backend for PiecewiseHybridModel
    pieces. This enables custom optimization strategies beyond CUDA graphs
    (e.g., torch.compile, ONNX runtime).
    """

    def capture_from_recorded_inputs(
        self,
        *,
        static_size: int,
        recorded_args: tuple[Any, ...],
        recorded_kwargs: dict[str, Any],
        runtime_size: int,
    ) -> None:
        """Capture one bucket using recorded runtime args/kwargs."""
        ...

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Run inference using the captured backend."""
        ...


BackendFactory = Callable[
    [torch.nn.Module, "PiecewiseHybridConfig", Any, Any],
    CaptureBackend,
]


def cudagraph_backend_factory(
    fn: torch.nn.Module,
    config: "PiecewiseHybridConfig",
    *,
    graph_pool: Any = None,
    device: torch.device | None = None,
) -> CUDAGraphPiece:
    """Factory that creates CUDAGraphPiece instances from configuration.

    This is the default backend factory. Custom factories should follow
    the same signature: (fn, config, *, graph_pool, device) -> CaptureBackend.
    """
    return CUDAGraphPiece(
        fn,
        capture_sizes=config.capture_sizes,
        warmup_iters=config.warmup_iters,
        zero_pad_inputs=config.zero_pad_inputs,
        runtime_size_fn=config.runtime_size_fn,
        check_input_addresses=config.check_input_addresses,
        graph_pool=graph_pool,
        device=device,
    )