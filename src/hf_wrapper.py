"""HuggingFace model wrapper for CUDA graph optimization.

This module provides utilities to wrap HuggingFace CausalLM models
with CUDA graph capture for faster inference.

Note: HuggingFace models use dynamic control flow that cannot be traced
with torch.fx. This module uses full-model CUDA graph capture instead of
piecewise capture. This is simpler and works with any HF model.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Callable, Optional, Any, List, Dict

import torch
import torch.nn as nn


@dataclass
class _CapturedGraph:
    """A captured CUDA graph for a specific input size."""
    static_size: int
    static_input_ids: torch.Tensor
    graph: torch.cuda.CUDAGraph
    static_output: torch.Tensor


class HFCudaGraphRunner(nn.Module):
    """CUDA Graph runner for HuggingFace CausalLM models.

    This class captures CUDA graphs for different input sizes (buckets)
    and replays them during inference for better performance.

    Unlike piecewise capture, this captures the entire model forward pass
    as a single CUDA graph. This works with HF models that have complex
    control flow that cannot be traced with torch.fx.
    """

    def __init__(
        self,
        model: nn.Module,
        capture_sizes: List[int],
        *,
        warmup_iters: int = 2,
        device: Optional[torch.device] = None,
        graph_pool: Optional[Any] = None,
    ):
        """Initialize the CUDA graph runner.

        Args:
            model: HuggingFace CausalLM model (must be on CUDA)
            capture_sizes: List of sequence lengths to capture
            warmup_iters: Number of warmup iterations before capture
            device: Target device (default: cuda)
            graph_pool: Optional shared graph memory pool
        """
        super().__init__()
        self.model = model
        self.capture_sizes = tuple(sorted(capture_sizes))
        self.warmup_iters = warmup_iters

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
        self._entries: Dict[int, _CapturedGraph] = {}
        self._captured = False

        # Get vocab size for validation
        model_config = getattr(model, 'config', None)
        self.vocab_size = getattr(model_config, 'vocab_size', 32000) if model_config else 32000

    def _select_static_size(self, runtime_size: int) -> int:
        """Select the appropriate bucket size for a runtime sequence length."""
        idx = bisect.bisect_left(self.capture_sizes, runtime_size)
        if idx >= len(self.capture_sizes):
            raise ValueError(
                f"Sequence length {runtime_size} exceeds max capture size {self.capture_sizes[-1]}"
            )
        return self.capture_sizes[idx]

    def capture(self) -> None:
        """Capture CUDA graphs for all configured sizes.

        This method should be called once before inference. It captures
        CUDA graphs from largest to smallest for better memory reuse.
        """
        self.model.eval()

        # First, do an initialization run with the largest size to
        # initialize any cached buffers (like rotary embeddings)
        max_size = max(self.capture_sizes)
        init_ids = torch.zeros((1, max_size), device=self.device, dtype=torch.long)
        with torch.inference_mode():
            _ = self.model(init_ids, use_cache=False)
        torch.cuda.synchronize()

        # Capture from large to small for better memory reuse
        sizes_to_capture = sorted(self.capture_sizes, reverse=True)

        for static_size in sizes_to_capture:
            self._capture_one_size(static_size)

        self._captured = True

    def _capture_one_size(self, static_size: int) -> None:
        """Capture CUDA graph for a single input size."""
        # Allocate static input buffer with batch dimension [1, seq_len]
        static_input_ids = torch.zeros(
            (1, static_size),
            device=self.device,
            dtype=torch.long
        )

        # Warmup runs
        with torch.inference_mode():
            for _ in range(self.warmup_iters):
                _ = self.model(static_input_ids, use_cache=False)

        # Capture
        graph = torch.cuda.CUDAGraph()

        with torch.inference_mode(), torch.cuda.graph(graph, pool=self.graph_pool):
            output = self.model(static_input_ids, use_cache=False)
            # Output shape: [1, seq_len, vocab_size]
            static_output = output.logits

        torch.cuda.synchronize()

        self._entries[static_size] = _CapturedGraph(
            static_size=static_size,
            static_input_ids=static_input_ids,
            graph=graph,
            static_output=static_output,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run inference using captured CUDA graphs.

        Args:
            input_ids: Input token IDs, shape [seq_len] or [1, seq_len]

        Returns:
            Logits tensor, shape [seq_len, vocab_size] or [1, seq_len, vocab_size]
        """
        # Handle input shape
        was_unbatched = input_ids.dim() == 1
        if was_unbatched:
            input_ids = input_ids.unsqueeze(0)  # [seq_len] -> [1, seq_len]

        if not self._captured:
            # Fall back to eager execution if not captured yet
            with torch.inference_mode():
                out = self.model(input_ids, use_cache=False).logits
                return out.squeeze(0) if was_unbatched else out

        runtime_size = input_ids.shape[1]
        static_size = self._select_static_size(runtime_size)
        entry = self._entries[static_size]

        # Copy input to static buffer [1, static_size]
        entry.static_input_ids[0, :runtime_size].copy_(input_ids[0])
        # Zero pad if needed
        if runtime_size < static_size:
            entry.static_input_ids[0, runtime_size:].zero_()

        # Replay graph
        entry.graph.replay()

        # Return only the valid portion of output
        # Clone to avoid subsequent calls overwriting the result
        out = entry.static_output[:, :runtime_size, :].clone()
        return out.squeeze(0) if was_unbatched else out

    @property
    def items(self) -> List[Dict[str, Any]]:
        """Return info about captured graphs (for compatibility)."""
        return [
            {"static_size": size, "is_attention_piece": False}
            for size in self.capture_sizes
        ]


def cudagraph_compile_hf(
    model: nn.Module,
    capture_sizes: List[int],
    *,
    warmup_iters: int = 2,
    device: Optional[torch.device] = None,
) -> HFCudaGraphRunner:
    """One-line API to apply CUDA graph optimization to HuggingFace model.

    This function wraps a HuggingFace CausalLM model with CUDA graph
    capture/replay for faster inference. Unlike piecewise_compile_hf,
    this captures the entire forward pass as a single graph.

    Args:
        model: HuggingFace CausalLM model (must be on CUDA, in eval mode)
        capture_sizes: List of sequence lengths to capture CUDA graphs for.
                      Should be sorted ascending. Runtime sequences will be
                      padded to the nearest capture size.
        warmup_iters: Number of warmup iterations before capture (default: 2)
        device: Target device (default: infer from model)

    Returns:
        HFCudaGraphRunner ready for capture and inference

    Example:
        >>> from transformers import AutoModelForCausalLM
        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B-Base")
        >>> model = model.cuda().eval()
        >>> runner = cudagraph_compile_hf(model, [32, 64, 128, 256])
        >>> runner.capture()  # Capture CUDA graphs
        >>> output = runner(input_ids)  # Run inference

    Note:
        - The model should already be on CUDA before calling this function
        - The returned runner only outputs logits (not CausalLMOutput)
        - KV cache is disabled for simplicity
    """
    model.eval()
    return HFCudaGraphRunner(
        model,
        capture_sizes,
        warmup_iters=warmup_iters,
        device=device,
    )


# Alias for backward compatibility
piecewise_compile_hf = cudagraph_compile_hf


def get_attention_modules(model: nn.Module, detector: Optional[Callable] = None) -> List[str]:
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


# Keep HFCausalLMWrapper for backward compatibility
class HFCausalLMWrapper(nn.Module):
    """Wrapper that simplifies HF CausalLM forward.

    Deprecated: Use HFCudaGraphRunner directly instead.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.config = getattr(model, 'config', None)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        outputs = self.model(
            input_ids=input_ids,
            use_cache=False,
            return_dict=True
        )
        return outputs.logits
