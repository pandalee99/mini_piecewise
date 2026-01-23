"""Mini Piecewise CUDA Graph Framework.

A minimal implementation of piecewise CUDA graph optimization for LLM inference.

Core API:
    - PiecewiseHybridConfig: Configuration for capture sizes and behavior
    - make_piecewise_hybrid_model: Build hybrid model from any nn.Module
    - PiecewiseHybridModel: The wrapped model with capture/replay

HuggingFace Integration:
    - piecewise_compile_hf: One-line API for HF CausalLM models
    - HFCausalLMWrapper: Wrapper for HF models
    - get_attention_modules: List attention modules in a model

Attention Detectors:
    - auto_attention_detector: Auto-detect attention for common architectures
    - qwen_attention_detector: Detect Qwen attention modules
    - llama_attention_detector: Detect LLaMA attention modules

Example:
    >>> from transformers import AutoModelForCausalLM
    >>> from src import piecewise_compile_hf
    >>>
    >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B-Base")
    >>> model = model.cuda().eval()
    >>>
    >>> hybrid = piecewise_compile_hf(model, [32, 64, 128, 256])
    >>> hybrid.capture()  # Capture CUDA graphs
    >>> output = hybrid(input_ids)  # Run inference
"""

from .config import (
    PiecewiseHybridConfig,
    default_is_attention_module,
    default_runtime_size_fn,
    qwen_attention_detector,
    llama_attention_detector,
    auto_attention_detector,
)
from .hybrid import (
    make_piecewise_hybrid_model,
    PiecewiseHybridModel,
)
from .hf_wrapper import (
    cudagraph_compile_hf,
    piecewise_compile_hf,  # Alias for backward compatibility
    HFCudaGraphRunner,
    HFCausalLMWrapper,
    get_attention_modules,
)

__all__ = [
    # Config
    "PiecewiseHybridConfig",
    "default_is_attention_module",
    "default_runtime_size_fn",
    "qwen_attention_detector",
    "llama_attention_detector",
    "auto_attention_detector",
    # Hybrid (for FX-traceable models)
    "make_piecewise_hybrid_model",
    "PiecewiseHybridModel",
    # HF CUDA Graph (for HuggingFace models)
    "cudagraph_compile_hf",
    "piecewise_compile_hf",  # Alias
    "HFCudaGraphRunner",
    "HFCausalLMWrapper",
    "get_attention_modules",
]
