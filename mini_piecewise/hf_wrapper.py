"""CUDA Graph runner with adapter-based input/output handling.

Provides a general-purpose CUDA graph runner that supports arbitrary
model signatures via pluggable input/output adapters, plus a convenience
wrapper for HuggingFace CausalLM models.
"""

from __future__ import annotations

import bisect
import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
import torch.nn as nn

from .config import default_runtime_size_fn
from .errors import CaptureNotPerformedError, CudaNotAvailableError

logger = logging.getLogger("mini_piecewise")


@dataclass
class _CapturedGraph:
    """A captured CUDA graph for a specific input size."""
    static_size: int
    static_inputs: tuple[Any, ...]
    static_kwargs: dict[str, Any]
    graph: torch.cuda.CUDAGraph
    static_output: Any


def _default_input_adapter(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Identity input adapter: pass user args/kwargs directly to model."""
    return args, kwargs


def _default_output_adapter(output: Any, runtime_size: int | None = None, static_size: int | None = None) -> Any:
    """Identity output adapter: return model output directly.

    When runtime_size and static_size are provided, slices tensor
    leaves on dim0 that match static_size (similar to tree_slice_dim0).
    """
    if runtime_size is not None and static_size is not None:
        from .tree_utils import tree_slice_dim0
        return tree_slice_dim0(output, runtime_size=runtime_size, static_size=static_size)
    return output


class CudaGraphRunner(nn.Module):
    """CUDA graph runner with pluggable adapters for any nn.Module.

    Captures CUDA graphs for a set of bucket sizes and replays them
    during inference. Supports arbitrary model signatures through
    input_adapter, output_adapter, and runtime_size_fn callbacks.

    Args:
        model: nn.Module to optimize (must be on CUDA, in eval mode)
        capture_sizes: Bucket sizes for CUDA graph capture
        warmup_iters: Warmup iterations before capture
        input_adapter: Transform user inputs to model inputs
        output_adapter: Transform model output, receives runtime/static size for slicing
        runtime_size_fn: Infer runtime size from user inputs
        zero_pad_inputs: Zero-pad inputs that fall below bucket size
        example_inputs_fn: Callable(int) -> user_args that produces example
            inputs for a given size. Used during capture to create valid
            initialization data. Default creates long dtype zeros (for HF models).
        device: Target CUDA device
        graph_pool: Shared memory pool for CUDA graphs
    """

    def __init__(
        self,
        model: nn.Module,
        capture_sizes: list[int],
        *,
        warmup_iters: int = 2,
        input_adapter: Callable[[tuple[Any, ...], dict[str, Any]], tuple[tuple[Any, ...], dict[str, Any]]] = _default_input_adapter,
        output_adapter: Callable[[Any], Any] = _default_output_adapter,
        runtime_size_fn: Callable[[tuple[Any, ...], dict[str, Any]], int] = default_runtime_size_fn,
        zero_pad_inputs: bool = True,
        example_inputs_fn: Callable[[int], Any] | None = None,
        device: Optional[torch.device] = None,
        graph_pool: Optional[Any] = None,
    ):
        super().__init__()
        self.model = model
        self.capture_sizes = tuple(sorted(capture_sizes))
        self.warmup_iters = warmup_iters
        self.input_adapter = input_adapter
        self.output_adapter = output_adapter
        self.runtime_size_fn = runtime_size_fn
        self.zero_pad_inputs = zero_pad_inputs

        # Default example_inputs_fn: long dtype zeros (HF convention).
        # Override for models that expect float inputs.
        if example_inputs_fn is None:
            self._example_inputs_fn = lambda s: (torch.zeros((s,), device=self.device if hasattr(self, 'device') else torch.device("cuda"), dtype=torch.long),)
        else:
            self._example_inputs_fn = example_inputs_fn

        # Determine device
        if device is None:
            try:
                param = next(model.parameters())
                device = param.device if param.is_cuda else torch.device("cuda")
            except StopIteration:
                device = torch.device("cuda")

        self.device = device
        self.graph_pool = graph_pool or torch.cuda.graph_pool_handle()

        # Storage for captured graphs
        self._entries: dict[int, _CapturedGraph] = {}
        self._captured = False

    def _select_static_size(self, runtime_size: int) -> int:
        """Select the appropriate bucket size for a runtime size."""
        idx = bisect.bisect_left(self.capture_sizes, runtime_size)
        if idx >= len(self.capture_sizes):
            raise ValueError(
                f"Size {runtime_size} exceeds max capture size {self.capture_sizes[-1]}"
            )
        return self.capture_sizes[idx]

    def capture(self) -> None:
        """Capture CUDA graphs for all configured sizes.

        This method should be called once before inference. It captures
        CUDA graphs from largest to smallest for better memory reuse.
        """
        self.model.eval()

        # First, do an initialization run with the largest size.
        max_size = max(self.capture_sizes)
        ex = self._example_inputs_fn(max_size)
        init_args, init_kwargs = self.input_adapter(ex, {})
        with torch.inference_mode():
            _ = self.model(*init_args, **init_kwargs)
        torch.cuda.synchronize()

        logger.info("Starting CudaGraphRunner capture for sizes: %s", list(self.capture_sizes))

        # Capture from large to small for better memory reuse
        sizes_to_capture = sorted(self.capture_sizes, reverse=True)

        for static_size in sizes_to_capture:
            self._capture_one_size(static_size)

        self._captured = True
        logger.info("Capture complete. %d graphs captured.", len(self._entries))

    def _capture_one_size(self, static_size: int) -> None:
        """Capture CUDA graph for a single input size."""
        # Create example inputs via the adapter
        ex = self._example_inputs_fn(static_size)
        example_args, example_kwargs = self.input_adapter(ex, {})

        # Find runtime_size to use for creating static buffers
        runtime_size = self.runtime_size_fn(example_args, example_kwargs)

        # Create static input buffers
        from .tree_utils import tree_make_static_like, tree_copy_into
        static_args = tree_make_static_like(example_args, static_size=static_size, runtime_size=runtime_size)
        static_kwargs = tree_make_static_like(example_kwargs, static_size=static_size, runtime_size=runtime_size)

        # Fill static buffers with valid data
        with torch.inference_mode():
            tree_copy_into(static_args, example_args, runtime_size=runtime_size, static_size=static_size, zero_pad=self.zero_pad_inputs)
            tree_copy_into(static_kwargs, example_kwargs, runtime_size=runtime_size, static_size=static_size, zero_pad=self.zero_pad_inputs)

        # Warmup runs
        with torch.inference_mode():
            for _ in range(self.warmup_iters):
                _ = self.model(*static_args, **static_kwargs)

        # Capture
        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(graph, pool=self.graph_pool):
            static_output = self.model(*static_args, **static_kwargs)

        torch.cuda.synchronize()

        self._entries[static_size] = _CapturedGraph(
            static_size=static_size,
            static_inputs=static_args,
            static_kwargs=static_kwargs,
            graph=graph,
            static_output=static_output,
        )

        logger.debug("Captured size=%d", static_size)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Run inference using captured CUDA graphs.

        Args and kwargs are passed through input_adapter first, then
        the model runs via CUDA graph replay, and the output is
        passed through output_adapter.
        """
        if not self._captured:
            # Fall back to eager execution if not captured yet
            model_args, model_kwargs = self.input_adapter(args, kwargs)
            with torch.inference_mode():
                output = self.model(*model_args, **model_kwargs)
            return self.output_adapter(output, None, None)

        # Determine runtime size from user-facing inputs
        runtime_size = self.runtime_size_fn(args, kwargs)
        static_size = self._select_static_size(runtime_size)
        entry = self._entries[static_size]

        # Transform user inputs to model inputs
        model_args, model_kwargs = self.input_adapter(args, kwargs)

        # Copy into static buffers
        from .tree_utils import tree_copy_into

        with torch.inference_mode():
            tree_copy_into(
                entry.static_inputs, model_args,
                runtime_size=runtime_size, static_size=static_size,
                zero_pad=self.zero_pad_inputs,
            )
            tree_copy_into(
                entry.static_kwargs, model_kwargs,
                runtime_size=runtime_size, static_size=static_size,
                zero_pad=self.zero_pad_inputs,
            )

            # Replay graph
            entry.graph.replay()

        # Apply output adapter with runtime_size/static_size for slicing
        return self.output_adapter(entry.static_output, runtime_size, static_size)

    def summary(self) -> dict[str, Any]:
        """Return diagnostic summary of the runner."""
        return {
            "captured": self._captured,
            "capture_sizes": list(self.capture_sizes),
            "num_entries": len(self._entries),
            "model_type": type(self.model).__name__,
        }

    def recapture(self, new_sizes: list[int] | None = None) -> None:
        """Re-capture with potentially different bucket sizes."""
        self.free()
        if new_sizes is not None:
            self.capture_sizes = tuple(sorted(new_sizes))
        self.capture()

    def free(self) -> None:
        """Release all captured graph memory."""
        self._entries.clear()
        self._captured = False
        logger.info("Freed all captured CudaGraphRunner resources")

    @property
    def items(self) -> list[dict[str, Any]]:
        """Return info about captured graphs (for compatibility)."""
        return [
            {"static_size": size, "is_attention_piece": False, "policy": "capture"}
            for size in self.capture_sizes
        ]


# --- HuggingFace convenience wrappers ---


class _HFCausalLMInputAdapter:
    """Input adapter for HuggingFace CausalLM models.

    Transforms unbatched input_ids [seq_len] into batched [1, seq_len]
    and injects use_cache=False into kwargs.
    """

    def __call__(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any]]:
        input_ids = args[0]
        if isinstance(input_ids, torch.Tensor) and input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)  # [seq_len] -> [1, seq_len]
        return (input_ids,), {"use_cache": False}


class _HFCausalLMOutputAdapter:
    """Output adapter for HuggingFace CausalLM models.

    Extracts logits from CausalLMOutput, slices to runtime_size,
    and squeezes batch dim if input was unbatched.
    """

    def __call__(self, output: Any, runtime_size: int | None = None, static_size: int | None = None) -> Any:
        logits = output.logits if hasattr(output, "logits") else output
        # logits shape: [1, static_size, vocab_size]
        if runtime_size is not None and static_size is not None and runtime_size < static_size:
            # Clone and slice to avoid stale values from subsequent calls
            logits = logits[:, :runtime_size, :].clone()
        else:
            logits = logits.clone()
        # Squeeze batch dim: [1, seq_len, vocab] -> [seq_len, vocab]
        return logits.squeeze(0)


def _hf_runtime_size_fn(args: tuple[Any, ...], kwargs: dict[str, Any]) -> int:
    """Runtime size fn for HF models: infer from input_ids dim.

    Handles both unbatched [seq_len] and batched [1, seq_len] inputs.
    """
    input_ids = args[0]
    if isinstance(input_ids, torch.Tensor):
        if input_ids.dim() == 1:
            return int(input_ids.shape[0])
        elif input_ids.dim() >= 2:
            return int(input_ids.shape[-1])  # last dim is seq_len
    raise ValueError("Cannot infer runtime size from HF input")


def _hf_example_inputs_fn(device: torch.device) -> Callable[[int], Any]:
    """Create example_inputs_fn for HF models (long dtype input_ids)."""
    def fn(static_size: int):
        return (torch.zeros((static_size,), device=device, dtype=torch.long),)
    return fn


def cudagraph_compile_hf(
    model: nn.Module,
    capture_sizes: list[int],
    *,
    warmup_iters: int = 2,
    device: Optional[torch.device] = None,
) -> CudaGraphRunner:
    """One-line API to apply CUDA graph optimization to a HuggingFace CausalLM model.

    This function wraps a HuggingFace CausalLM model with CUDA graph
    capture/replay for faster inference. It uses input/output adapters
    to handle the HF model's specific signature:

    - Input: unbatched input_ids [seq_len] or batched [1, seq_len]
    - Output: logits [seq_len, vocab_size] or [1, seq_len, vocab_size]
    - Model call: model(input_ids, use_cache=False) -> CausalLMOutput.logits

    Args:
        model: HuggingFace CausalLM model (must be on CUDA, in eval mode)
        capture_sizes: List of sequence lengths to capture CUDA graphs for.
        warmup_iters: Number of warmup iterations before capture (default: 2)
        device: Target device (default: infer from model)

    Returns:
        CudaGraphRunner ready for capture and inference

    Example:
        >>> from transformers import AutoModelForCausalLM
        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B-Base")
        >>> model = model.cuda().eval()
        >>> runner = cudagraph_compile_hf(model, [32, 64, 128, 256])
        >>> runner.capture()  # Capture CUDA graphs
        >>> output = runner(input_ids)  # Run inference

    Note:
        - The model should already be on CUDA before calling this function
        - The returned runner outputs logits (not CausalLMOutput)
        - KV cache is disabled for simplicity
    """
    model.eval()
    # Determine device for example_inputs_fn
    if device is None:
        try:
            param = next(model.parameters())
            runner_device = param.device if param.is_cuda else torch.device("cuda")
        except StopIteration:
            runner_device = torch.device("cuda")
    else:
        runner_device = device

    return CudaGraphRunner(
        model,
        capture_sizes,
        warmup_iters=warmup_iters,
        input_adapter=_HFCausalLMInputAdapter(),
        output_adapter=_HFCausalLMOutputAdapter(),
        runtime_size_fn=_hf_runtime_size_fn,
        example_inputs_fn=_hf_example_inputs_fn(runner_device),
        device=device,
    )


def get_attention_modules(model: nn.Module, detector: Callable | None = None) -> list[str]:
    """Get list of attention module names in a model.

    Useful for debugging and understanding model structure.

    Args:
        model: PyTorch model
        detector: Attention detection function (default: auto_attention_detector)

    Returns:
        List of qualified names of attention modules
    """
    from .config import auto_attention_detector

    if detector is None:
        detector = auto_attention_detector

    attention_modules = []
    for name, mod in model.named_modules():
        if detector(mod, name):
            attention_modules.append(name)

    return attention_modules


# Internal compatibility aliases (not exported via __init__.py)
class HFCausalLMWrapper(nn.Module):
    """Adapter that unwraps CausalLMOutput to logits.

    Internal-only. Prefer CudaGraphRunner with output_adapter for new code.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.config = getattr(model, "config", None)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        outputs = self.model(
            input_ids=input_ids,
            use_cache=False,
            return_dict=True,
        )
        return outputs.logits


# Internal backward-compatibility alias (not exported).
HFCudaGraphRunner = CudaGraphRunner