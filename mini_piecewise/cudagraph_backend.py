from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch

from .errors import CaptureNotPerformedError, CudaNotAvailableError, ShapeOutOfRangeError
from .tree_utils import (
    tree_copy_into,
    tree_make_static_like,
    tree_slice_dim0,
    tree_tensor_addresses,
)


@dataclass
class _Entry:
    static_size: int
    static_args: tuple[Any, ...]
    static_kwargs: dict[str, Any]
    graph: torch.cuda.CUDAGraph
    static_output: Any
    input_addresses: Optional[list[int]] = None


class CUDAGraphPiece(torch.nn.Module):
    """CUDA Graph capture/replay backend for a single FX submodule piece.

    Wraps a GraphModule piece with CUDA graph capture and replay.
    Buckets runtime sizes against a set of pre-captured static sizes,
    padding inputs to the nearest bucket and slicing outputs back
    to the original runtime size.

    Runtime size is inferred from the first tensor argument's dim0
    by default. Inputs and outputs are assumed to be token-major:
    [T, ...] where T is the dynamic sequence dimension.

    All runtime inputs are copied into pre-allocated static buffers
    before replay to ensure address stability across calls.
    """

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        capture_sizes: tuple[int, ...],
        warmup_iters: int,
        zero_pad_inputs: bool,
        runtime_size_fn: Callable[[tuple[Any, ...], dict[str, Any]], int],
        check_input_addresses: bool = False,
        graph_pool: Any = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self._fn = fn
        self._capture_sizes = capture_sizes
        self._warmup_iters = warmup_iters
        self._zero_pad_inputs = zero_pad_inputs
        self._runtime_size_fn = runtime_size_fn
        self._check_input_addresses = check_input_addresses

        if not torch.cuda.is_available():
            raise CudaNotAvailableError("CUDA is required for CUDAGraphPiece")

        self._device = device or torch.device("cuda")
        self._graph_pool = graph_pool if graph_pool is not None else torch.cuda.graph_pool_handle()

        self._entries: dict[int, _Entry] = {}

    def _iter_tensors(self, tree: Any):
        if isinstance(tree, torch.Tensor):
            yield tree
        elif isinstance(tree, dict):
            for v in tree.values():
                yield from self._iter_tensors(v)
        elif isinstance(tree, (list, tuple)):
            for v in tree:
                yield from self._iter_tensors(v)

    @property
    def capture_sizes(self) -> tuple[int, ...]:
        return self._capture_sizes

    def _select_static_size(self, runtime_size: int) -> int:
        sizes = self._capture_sizes
        idx = bisect.bisect_left(sizes, runtime_size)
        if idx >= len(sizes):
            raise ShapeOutOfRangeError(
                f"runtime_size={runtime_size} exceeds max capture size={sizes[-1]}"
            )
        return sizes[idx]

    def capture_from_recorded_inputs(
        self,
        *,
        static_size: int,
        recorded_args: tuple[Any, ...],
        recorded_kwargs: dict[str, Any],
        runtime_size: int,
    ) -> None:
        """Capture a single bucket using recorded runtime inputs.

        Allocates static buffers based on the recorded input shapes and
        captures the wrapped function against those buffers. Valid data
        must be materialized before warmup/capture to avoid device-side
        asserts in modules like Embedding that index into tensors.
        """

        if static_size in self._entries:
            return

        # Allocate static input buffers from runtime samples.
        static_args = tree_make_static_like(recorded_args, static_size=static_size, runtime_size=runtime_size)
        static_kwargs = tree_make_static_like(recorded_kwargs, static_size=static_size, runtime_size=runtime_size)

        # Static buffers must contain valid data before warmup/capture.
        # tree_make_static_like allocates with torch.empty, which yields
        # uninitialized memory. Modules like Embedding may index into
        # these tensors and trigger device-side asserts on garbage values.
        with torch.inference_mode():
            tree_copy_into(
                static_args,
                recorded_args,
                runtime_size=runtime_size,
                static_size=static_size,
                zero_pad=self._zero_pad_inputs,
            )
            tree_copy_into(
                static_kwargs,
                recorded_kwargs,
                runtime_size=runtime_size,
                static_size=static_size,
                zero_pad=self._zero_pad_inputs,
            )

        # Ensure all tensor buffers are on CUDA (addresses must be device-local).
        for t in self._iter_tensors((static_args, static_kwargs)):
            if not t.is_cuda:
                raise ValueError("Recorded inputs must be CUDA tensors")

        g = torch.cuda.CUDAGraph()

        # Warmup outside capture.
        with torch.inference_mode():
            for _ in range(self._warmup_iters):
                _ = self._fn(*static_args, **static_kwargs)

        # Capture.
        with torch.inference_mode(), torch.cuda.graph(g, pool=self._graph_pool):
            static_output = self._fn(*static_args, **static_kwargs)

        torch.cuda.synchronize()

        input_addresses = None
        if self._check_input_addresses:
            input_addresses = tree_tensor_addresses((static_args, static_kwargs))

        self._entries[static_size] = _Entry(
            static_size=static_size,
            static_args=static_args,
            static_kwargs=static_kwargs,
            graph=g,
            static_output=static_output,
            input_addresses=input_addresses,
        )

    def forward(self, *runtime_args: Any, **runtime_kwargs: Any) -> Any:
        if not self._entries:
            raise CaptureNotPerformedError("CUDAGraphPiece.capture_* must be called before forward")

        runtime_size = int(self._runtime_size_fn(runtime_args, runtime_kwargs))
        static_size = self._select_static_size(runtime_size)
        entry = self._entries.get(static_size)
        if entry is None:
            raise CaptureNotPerformedError(
                f"No capture entry for static_size={static_size}. Did you call capture()?"
            )

        # In-place updates to inference-mode-allocated buffers
            # must occur under inference_mode.
        with torch.inference_mode():
            tree_copy_into(
                entry.static_args,
                runtime_args,
                runtime_size=runtime_size,
                static_size=static_size,
                zero_pad=self._zero_pad_inputs,
            )
            tree_copy_into(
                entry.static_kwargs,
                runtime_kwargs,
                runtime_size=runtime_size,
                static_size=static_size,
                zero_pad=self._zero_pad_inputs,
            )

            if entry.input_addresses is not None:
                cur = tree_tensor_addresses((entry.static_args, entry.static_kwargs))
                if cur != entry.input_addresses:
                    raise RuntimeError(
                        "Input addresses changed after capture. "
                        "This violates CUDA Graph requirements."
                    )

            entry.graph.replay()
            out = tree_slice_dim0(entry.static_output, runtime_size=runtime_size, static_size=static_size)

        return out

    def info(self) -> dict[str, Any]:
        """Return diagnostic info about captured entries."""
        captured_sizes = sorted(self._entries.keys())
        # Estimate memory from static buffers
        mem_estimate = 0
        for entry in self._entries.values():
            for t in self._iter_tensors((entry.static_args, entry.static_kwargs, entry.static_output)):
                mem_estimate += t.nelement() * t.element_size()
        return {
            "backend": "CUDAGraphPiece",
            "fn_type": type(self._fn).__name__,
            "capture_sizes": captured_sizes,
            "num_entries": len(self._entries),
            "memory_estimate_bytes": mem_estimate,
        }

    def free(self) -> None:
        """Release all captured graph memory."""
        self._entries.clear()
        self._installed = False
