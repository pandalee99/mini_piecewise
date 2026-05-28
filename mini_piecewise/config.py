from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterable, Any

import torch


class PiecePolicy(Enum):
    """Policy for how a piece should be handled during capture.

    - CAPTURE: Optimize this piece using the configured backend (e.g., CUDA graph)
    - EAGER: Keep this piece in eager mode (like attention modules)
    - SKIP: Skip this piece entirely (rarely used, but useful for debugging)
    """

    CAPTURE = "capture"
    EAGER = "eager"
    SKIP = "skip"


PieceSelector = Callable[[torch.nn.Module, str], PiecePolicy]


def default_is_attention_module(mod: torch.nn.Module, qualname: str) -> bool:
    """Heuristic attention detector based on class name and module path.

    Matches modules whose class name contains 'Attention' or 'Attn',
    modules whose qualified path contains 'attn'/'attention', and
    torch.nn.MultiheadAttention instances.

    For production use, consider providing a stricter allow/deny list
    tailored to your specific model architecture.
    """
    name = qualname.lower()
    cls = mod.__class__.__name__.lower()

    if isinstance(mod, torch.nn.MultiheadAttention):
        return True
    if "attention" in cls or "attn" in cls:
        return True
    if "attention" in name or "/attn" in name or "attn" in name:
        return True
    return False


def qwen_attention_detector(mod: torch.nn.Module, qualname: str) -> bool:
    """Detect Qwen series attention modules.

    Supports Qwen, Qwen2, Qwen3 and variants.
    """
    cls = mod.__class__.__name__
    return any(k in cls for k in [
        "Qwen3Attention", "Qwen2Attention", "QwenAttention",
        "Qwen3SdpaAttention", "Qwen2SdpaAttention",
        "Qwen3FlashAttention2", "Qwen2FlashAttention2",
    ])


def llama_attention_detector(mod: torch.nn.Module, qualname: str) -> bool:
    """Detect LLaMA/Llama series attention modules.

    Supports LLaMA, Mistral, Gemma and similar architectures.
    """
    cls = mod.__class__.__name__
    return any(k in cls for k in [
        "LlamaAttention", "LlamaSdpaAttention", "LlamaFlashAttention2",
        "MistralAttention", "MistralSdpaAttention", "MistralFlashAttention2",
        "GemmaAttention", "GemmaSdpaAttention", "GemmaFlashAttention2",
        "Gemma2Attention", "Gemma2SdpaAttention", "Gemma2FlashAttention2",
    ])


def auto_attention_detector(mod: torch.nn.Module, qualname: str) -> bool:
    """Auto-detect attention modules across common transformer architectures.

    Combines architecture-specific detectors (Qwen, LLaMA, Mistral, Gemma)
    with generic heuristics. Use this when the model architecture is unknown
    or when you want broad coverage across multiple model families.
    """
    cls = mod.__class__.__name__

    # Check specific model families first
    if qwen_attention_detector(mod, qualname):
        return True
    if llama_attention_detector(mod, qualname):
        return True

    # Check for torch built-in attention
    if isinstance(mod, torch.nn.MultiheadAttention):
        return True

    # Generic heuristic: class name contains "Attention" or "Attn"
    cls_lower = cls.lower()
    if "attention" in cls_lower or "attn" in cls_lower:
        # Exclude non-attention modules that might have these keywords
        exclude_keywords = ["mask", "norm", "dropout", "bias", "weight"]
        if not any(k in cls_lower for k in exclude_keywords):
            return True

    return False


def attention_piece_selector(mod: torch.nn.Module, qualname: str) -> PiecePolicy:
    """Default piece selector: keep attention modules eager, capture the rest.

    Returns PiecePolicy.EAGER for detected attention modules and
    PiecePolicy.CAPTURE for all other modules. This mirrors the
    standard optimization strategy where attention (dynamic, data-dependent)
    stays in eager mode while compute-heavy MLP/embedding layers benefit
    from CUDA graph capture.
    """
    if auto_attention_detector(mod, qualname):
        return PiecePolicy.EAGER
    return PiecePolicy.CAPTURE


def default_runtime_size_fn(args: tuple[object, ...], kwargs: dict[str, object]) -> int:
    """Infer runtime size from the first tensor argument leading dimension."""

    for a in args:
        if isinstance(a, torch.Tensor):
            if a.ndim < 1:
                raise ValueError("Cannot infer runtime size from a 0-dim tensor")
            return int(a.shape[0])
    for v in kwargs.values():
        if isinstance(v, torch.Tensor):
            if v.ndim < 1:
                raise ValueError("Cannot infer runtime size from a 0-dim tensor")
            return int(v.shape[0])
    raise ValueError("Cannot infer runtime size: no tensor inputs")


def _default_backend_factory(
    fn: torch.nn.Module,
    config: PiecewiseHybridConfig,
    *,
    graph_pool: Any = None,
    device: torch.device | None = None,
) -> Any:
    """Default backend factory using CUDAGraphPiece.

    Imported lazily to avoid circular imports. See backends.py.
    """
    from .backends import cudagraph_backend_factory
    return cudagraph_backend_factory(fn, config, graph_pool=graph_pool, device=device)


@dataclass(frozen=True)
class PiecewiseHybridConfig:
    """Configuration for the piecewise hybrid runner with backend and policy support."""

    capture_sizes: tuple[int, ...]

    # Capture behavior
    warmup_iters: int = 2
    zero_pad_inputs: bool = True

    # Dispatch
    runtime_size_fn: Callable[[tuple[object, ...], dict[str, object]], int] = default_runtime_size_fn

    # How to classify pieces for capture vs eager.
    # piece_selector is the new general mechanism (returns PiecePolicy).
    # is_attention_module is kept for backward compatibility.
    piece_selector: Callable[[torch.nn.Module, str], PiecePolicy] = attention_piece_selector
    is_attention_module: Callable[[torch.nn.Module, str], bool] | None = None

    # Backend factory: determines which backend to use for CAPTURE pieces.
    backend_factory: Callable[
        [torch.nn.Module, PiecewiseHybridConfig, Any, Any],
        Any,
    ] = _default_backend_factory

    # Debug
    check_input_addresses: bool = False

    def __post_init__(self) -> None:
        sizes = tuple(int(x) for x in self.capture_sizes)
        if not sizes or any(x <= 0 for x in sizes):
            raise ValueError("capture_sizes must be non-empty positive ints")
        if tuple(sorted(sizes)) != sizes:
            raise ValueError("capture_sizes must be sorted ascending")
        if len(set(sizes)) != len(sizes):
            raise ValueError("capture_sizes must not contain duplicates")
        if self.warmup_iters < 0:
            raise ValueError("warmup_iters must be >= 0")

    def _effective_piece_selector(self) -> Callable[[torch.nn.Module, str], PiecePolicy]:
        """Return the effective piece selector.

        If is_attention_module is explicitly set (not None), wrap it
        into a PieceSelector for backward compatibility. Otherwise,
        use piece_selector directly.
        """
        if self.is_attention_module is not None:
            attn_fn = self.is_attention_module
            def _compat_selector(mod: torch.nn.Module, qualname: str) -> PiecePolicy:
                if attn_fn(mod, qualname):
                    return PiecePolicy.EAGER
                return PiecePolicy.CAPTURE
            return _compat_selector
        return self.piece_selector

    @staticmethod
    def from_sizes(sizes: Iterable[int], **kwargs) -> PiecewiseHybridConfig:
        uniq = sorted({int(x) for x in sizes})
        return PiecewiseHybridConfig(capture_sizes=tuple(uniq), **kwargs)