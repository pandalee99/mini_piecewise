# Mini Piecewise CUDA Graph

A general-purpose framework for CUDA graph optimization in LLM inference. Provides flexible approaches for different model types:

1. **CudaGraphRunner** (general): Works with any nn.Module via input/output adapters
2. **PiecewiseHybridModel** (FX-traceable): Splits model into pieces, captures non-attention parts
3. **cudagraph_compile_hf** (HuggingFace): Convenience wrapper for HF CausalLM models

## Features

- 2-4x inference speedup through CUDA graph capture/replay
- Flexible backend system (CUDAGraph, custom backends via protocol)
- General piece selection policies (PiecePolicy: CAPTURE, EAGER, SKIP)
- Arbitrary model signature support via input/output adapters
- Automatic attention module detection
- Sequence length bucketing for dynamic inputs
- Works with Qwen, LLaMA, and other transformer models
- Diagnostic tools and lifecycle management

## Installation

```bash
cd mini_piecewise
pip install -e .
```

## Quick Start

### For HuggingFace Models (Recommended)

```python
import torch
from transformers import AutoModelForCausalLM
from mini_piecewise import cudagraph_compile_hf

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B-Base",
    torch_dtype=torch.bfloat16,
).cuda().eval()

# Create CUDA graph runner
runner = cudagraph_compile_hf(model, capture_sizes=[32, 64, 128, 256])

# Capture CUDA graphs (do this once)
runner.capture()

# Run inference (uses CUDA graph replay)
input_ids = torch.randint(0, 1000, (64,), device="cuda", dtype=torch.long)
output = runner(input_ids)  # Fast!
```

### For Any Model (General CudaGraphRunner)

```python
import torch
from mini_piecewise import CudaGraphRunner

# Your model (any nn.Module)
model = MyModel().cuda().eval()

# Create runner with custom adapters
runner = CudaGraphRunner(
    model,
    capture_sizes=[32, 64, 128],
    input_adapter=lambda args, kwargs: (args, kwargs),  # Transform inputs
    output_adapter=lambda output: output,  # Transform outputs
)

runner.capture()
output = runner(*my_inputs)
```

### For FX-Traceable Models (Piecewise)

```python
import torch
from mini_piecewise import PiecewiseHybridConfig, PiecePolicy, make_piecewise_hybrid_model

# Your model (must be torch.fx traceable)
model = MyModel().cuda().eval()

# Create config with piece selection policy
config = PiecewiseHybridConfig.from_sizes(
    [32, 64, 128],
    warmup_iters=2,
)

# Build hybrid model
def example_inputs_fn(static_size):
    return (torch.zeros((static_size,), device="cuda", dtype=torch.long),)

hybrid = make_piecewise_hybrid_model(model, config, example_inputs_fn=example_inputs_fn)

# Capture and run
hybrid.capture()
output = hybrid(input_ids)
```

## Advanced Usage

### Custom Piece Selection Policy

```python
from mini_piecewise import PiecePolicy

def my_piece_selector(mod, qualname):
    """Custom policy: capture MLPs eagerly, use CUDA graph for everything else."""
    cls_name = mod.__class__.__name__.lower()
    if "mlp" in cls_name or "feedforward" in cls_name:
        return PiecePolicy.EAGER  # Keep MLPs eager
    if "attention" in cls_name or "attn" in cls_name:
        return PiecePolicy.EAGER  # Keep attention eager
    return PiecePolicy.CAPTURE  # Capture everything else

config = PiecewiseHybridConfig.from_sizes(
    [32, 64, 128],
    piece_selector=my_piece_selector,
)
```

### Custom Backend

```python
from mini_piecewise.backends import CaptureBackend
import torch.nn as nn

class MyCustomBackend(nn.Module):
    """Implement the CaptureBackend protocol."""

    def capture_from_recorded_inputs(self, *, static_size, recorded_args,
                                      recorded_kwargs, runtime_size):
        # Your capture logic
        pass

    def forward(self, *args, **kwargs):
        # Your forward logic
        return self.fn(*args, **kwargs)

def my_backend_factory(fn, config, *, graph_pool=None, device=None):
    return MyCustomBackend(fn)

config = PiecewiseHybridConfig.from_sizes(
    [32, 64],
    backend_factory=my_backend_factory,
)
```

### Diagnostics and Lifecycle

```python
from mini_piecewise import ModelInspector, setup_logging

# Enable logging
setup_logging(level=logging.DEBUG)

# Inspect model
summary = ModelInspector.piece_summary(hybrid)
print(ModelInspector.format_summary(summary))

# Re-capture with new sizes
hybrid.recapture([16, 32, 64, 128])

# Free resources
hybrid.free()
```

## API Reference

### Core

| Class/Function | Description |
|------|------------|
| `PiecewiseHybridConfig` | Configuration for capture sizes, policies, and behavior |
| `PiecePolicy` | Enum: CAPTURE, EAGER, SKIP |
| `PieceSelector` | Callable that returns PiecePolicy for each module |
| `attention_piece_selector` | Default selector (keep attention eager) |
| `make_piecewise_hybrid_model` | Build piecewise model from FX-traceable nn.Module |
| `PiecewiseHybridModel` | Hybrid model with capture/replay |

### Backends

| Class/Function | Description |
|------|------------|
| `CaptureBackend` | Protocol for piece capture/replay backends |
| `CUDAGraphPiece` | CUDA graph capture backend |
| `cudagraph_backend_factory` | Default factory for CUDAGraphPiece |

### CUDA Graph Runner

| Class/Function | Description |
|------|------------|
| `CudaGraphRunner` | General CUDA graph runner with adapter support |
| `cudagraph_compile_hf` | Convenience API for HF CausalLM models |
| `get_attention_modules` | List attention modules in a model |

### Diagnostics

| Class/Function | Description |
|------|------------|
| `ModelInspector` | Inspect captured model structure and stats |
| `setup_logging` | Configure mini_piecewise logging |

### Attention Detectors

```python
from mini_piecewise import (
    auto_attention_detector,    # Generic auto-detection
    qwen_attention_detector,    # Qwen-specific
    llama_attention_detector,   # LLaMA/Mistral/Gemma
)
```

## Examples

```bash
python examples/qwen3_example.py      # HuggingFace model
python examples/simple_llm.py          # Piecewise FX approach
python examples/multi_layer_llm.py     # Multi-layer piecewise
python examples/custom_backend.py      # Custom backend demo
```

## Testing

```bash
pytest tests/ -v                       # Run all tests
pytest tests/test_piecewise_hybrid.py  # Unit tests
pytest tests/test_qwen3_model.py       # Qwen3 tests
python run_qwen3_test.py --benchmark   # End-to-end benchmark
```

## Performance

Tested on NVIDIA L20 with Qwen3-0.6B-Base:

| Seq Len | Eager (ms) | CudaGraph (ms) | Speedup |
|---------|------------|----------------|---------|
| 32      | 4.8        | 1.1            | 4.48x   |
| 64      | 5.7        | 1.5            | 3.72x   |
| 128     | 7.8        | 2.8            | 2.78x   |

## Project Structure

```
mini_piecewise/
├── mini_piecewise/           # Main package
│   ├── __init__.py           # Public API exports
│   ├── backends.py           # Backend protocol & factory
│   ├── config.py             # PiecePolicy, PieceSelector, PiecewiseHybridConfig
│   ├── hybrid.py             # PiecewiseHybridModel
│   ├── cudagraph_backend.py  # CUDAGraphPiece implementation
│   ├── hf_wrapper.py         # CudaGraphRunner & HF convenience
│   ├── fx_split.py           # FX graph splitting
│   ├── diagnostics.py        # ModelInspector & logging
│   ├── tree_utils.py         # Tree structure utilities
│   └── errors.py             # Custom error types
├── tests/
│   ├── test_piecewise_hybrid.py  # Unit tests
│   └── test_qwen3_model.py       # Qwen3 integration tests
├── examples/
│   ├── qwen3_example.py          # HF model example
│   ├── simple_llm.py             # Simple piecewise example
│   ├── multi_layer_llm.py        # Multi-layer piecewise example
│   └── custom_backend.py         # Custom backend example
├── benchmarks/
│   ├── bench_llm_block.py
│   ├── bench_toy_model.py
│   └── utils.py
├── pyproject.toml
└── README.md
```

## Design Decisions

1. **Backend Abstraction**: Any object satisfying the CaptureBackend protocol can be used. CUDAGraphPiece is the default, but torch.compile, ONNX, or custom backends are possible.

2. **Piece Selection Policy**: PiecePolicy (CAPTURE/EAGER/SKIP) generalizes the old is_attention_module approach. Users can define custom selectors.

3. **Adapter Pattern**: CudaGraphRunner uses input/output adapters to support any model signature, not just CausalLM(input_ids)->logits.

4. **Bucketing**: Runtime sequences are padded to the nearest bucket size for fixed CUDA graphs with variable-length support.

5. **Inference Only**: This framework is designed for inference. KV cache is disabled for simplicity.

## Limitations

- Inference only (no training support)
- Single batch dimension (batch size = 1) for piecewise approach
- KV cache disabled
- Requires CUDA

## License

MIT

## Acknowledgments

Inspired by [SGLang](https://github.com/sgl-project/sglang) piecewise CUDA graph implementation.