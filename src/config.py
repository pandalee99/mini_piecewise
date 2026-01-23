from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import torch


def default_is_attention_module(mod: torch.nn.Module, qualname: str) -> bool:
    """Heuristic attention detector.

    This is intentionally simple:
    - Treat modules whose class name contains "Attention" as attention.
    - Treat names containing "attn"/"attention" as attention.
    - Treat torch.nn.MultiheadAttention as attention.

    Real projects typically provide a stricter allow/deny list.
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
    """Auto-detect attention modules for common architectures.

    This is a more comprehensive detector that combines heuristics
    for various model families. Use this when you don't know the
    specific model architecture.
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


def default_runtime_size_fn(args: tuple[object, ...], kwargs: dict[str, object]) -> int:
    """Infer runtime size from the first tensor argument's dim0."""

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


@dataclass(frozen=True)
class PiecewiseHybridConfig:
    """Configuration for the minimal piecewise hybrid runner."""

    capture_sizes: tuple[int, ...]

    # Capture behavior
    warmup_iters: int = 2
    zero_pad_inputs: bool = True

    # Dispatch
    runtime_size_fn: Callable[[tuple[object, ...], dict[str, object]], int] = default_runtime_size_fn

    # How to identify "attention" pieces to keep eager.
    is_attention_module: Callable[[torch.nn.Module, str], bool] = default_is_attention_module

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

    @staticmethod
    def from_sizes(sizes: Iterable[int], **kwargs) -> "PiecewiseHybridConfig":
        uniq = sorted({int(x) for x in sizes})
        return PiecewiseHybridConfig(capture_sizes=tuple(uniq), **kwargs)
