"""Piecewise CUDA Graph Framework.

A general-purpose framework for piecewise CUDA graph optimization
in LLM inference. Supports arbitrary model signatures via adapters,
extensible backends via the CaptureBackend protocol, and flexible
piece selection policies.

Core API:
    - PiecewiseHybridConfig: Configuration for capture sizes, policies, and behavior
    - PiecePolicy: Enum for piece handling (CAPTURE, EAGER, SKIP)
    - PieceSelector: Callable that determines piece policy
    - attention_piece_selector: Default selector (keep attention eager, capture rest)
    - make_piecewise_hybrid_model: Build hybrid model from any nn.Module
    - PiecewiseHybridModel: The wrapped model with capture/replay

Backend System:
    - CaptureBackend: Protocol for piece capture/replay backends
    - CUDAGraphPiece: CUDA graph capture backend
    - cudagraph_backend_factory: Default factory for CUDA graph backends

HuggingFace Integration:
    - CudaGraphRunner: General CUDA graph runner with adapter support
    - cudagraph_compile_hf: One-line API for HF CausalLM models
    - get_attention_modules: List attention modules in a model

Attention Detectors:
    - auto_attention_detector: Auto-detect attention for common architectures
    - qwen_attention_detector: Detect Qwen attention modules
    - llama_attention_detector: Detect LLaMA attention modules

Diagnostics:
    - ModelInspector: Inspect captured model structure and statistics
    - setup_logging: Configure mini_piecewise logging

Example:
    >>> from transformers import AutoModelForCausalLM
    >>> from mini_piecewise import cudagraph_compile_hf
    >>>
    >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B-Base")
    >>> model = model.cuda().eval()
    >>>
    >>> runner = cudagraph_compile_hf(model, [32, 64, 128, 256])
    >>> runner.capture()  # Capture CUDA graphs
    >>> output = runner(input_ids)  # Run inference
"""

from .config import (
    PiecewiseHybridConfig,
    PiecePolicy,
    PieceSelector,
    attention_piece_selector,
    default_is_attention_module,
    default_runtime_size_fn,
    qwen_attention_detector,
    llama_attention_detector,
    auto_attention_detector,
)
from .backends import (
    CaptureBackend,
    cudagraph_backend_factory,
)
from .cudagraph_backend import CUDAGraphPiece
from .hybrid import (
    make_piecewise_hybrid_model,
    PiecewiseHybridModel,
)
from .hf_wrapper import (
    CudaGraphRunner,
    cudagraph_compile_hf,
    get_attention_modules,
)
from .diagnostics import ModelInspector, setup_logging

__all__ = [
    # Config
    "PiecewiseHybridConfig",
    "PiecePolicy",
    "PieceSelector",
    "attention_piece_selector",
    "default_is_attention_module",
    "default_runtime_size_fn",
    "qwen_attention_detector",
    "llama_attention_detector",
    "auto_attention_detector",
    # Backend
    "CaptureBackend",
    "CUDAGraphPiece",
    "cudagraph_backend_factory",
    # Hybrid (for FX-traceable models)
    "make_piecewise_hybrid_model",
    "PiecewiseHybridModel",
    # CUDA Graph Runner (general + HF)
    "CudaGraphRunner",
    "cudagraph_compile_hf",
    "get_attention_modules",
    # Diagnostics
    "ModelInspector",
    "setup_logging",
]