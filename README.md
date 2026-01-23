# Mini Piecewise CUDA Graph

A minimal framework for CUDA graph optimization in LLM inference. Provides two approaches:

1. **Full-Model CUDA Graph** (for HuggingFace models): Captures the entire model as a single CUDA graph
2. **Piecewise CUDA Graph** (for FX-traceable models): Splits model into pieces, captures non-attention parts

## Features

- 2-4x inference speedup through CUDA graph capture/replay
- Simple one-line API for HuggingFace models
- Automatic attention module detection
- Sequence length bucketing for dynamic inputs
- Works with Qwen, LLaMA, and other transformer models

## Installation

```bash
cd /vllm-workspace/mini_piecewise
pip install -e .
```

## Quick Start

### For HuggingFace Models (Recommended)

```python
import torch
from transformers import AutoModelForCausalLM
from src import cudagraph_compile_hf

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

### For FX-Traceable Models

```python
import torch
from src import PiecewiseHybridConfig, make_piecewise_hybrid_model

# Your model (must be torch.fx traceable)
model = MyModel().cuda().eval()

# Create config
config = PiecewiseHybridConfig.from_sizes([32, 64, 128])

# Build hybrid model
def example_inputs_fn(static_size):
    return (torch.zeros((static_size,), device="cuda", dtype=torch.long),)

hybrid = make_piecewise_hybrid_model(model, config, example_inputs_fn=example_inputs_fn)

# Capture and run
hybrid.capture()
output = hybrid(input_ids)
```

## API Reference

### HuggingFace Integration

#### `cudagraph_compile_hf(model, capture_sizes, **kwargs)`

Wraps a HuggingFace CausalLM model with CUDA graph capture.

**Parameters:**
- `model`: HuggingFace model (must be on CUDA, in eval mode)
- `capture_sizes`: List of sequence lengths to capture (buckets)
- `warmup_iters`: Warmup iterations before capture (default: 2)
- `device`: Target device (default: infer from model)

**Returns:** `HFCudaGraphRunner` instance

#### `HFCudaGraphRunner`

- `capture()`: Capture CUDA graphs for all bucket sizes
- `forward(input_ids)`: Run inference using captured graphs

### Piecewise API

#### `PiecewiseHybridConfig.from_sizes(capture_sizes, **kwargs)`

Create configuration for piecewise capture.

**Parameters:**
- `capture_sizes`: List of sequence lengths to capture
- `warmup_iters`: Warmup iterations (default: 2)
- `zero_pad_inputs`: Zero-pad inputs to bucket size (default: True)
- `is_attention_module`: Custom attention detection function

#### `make_piecewise_hybrid_model(model, config, example_inputs_fn, **kwargs)`

Build a piecewise hybrid model using torch.fx.

**Parameters:**
- `model`: PyTorch model (must be FX-traceable)
- `config`: `PiecewiseHybridConfig` instance
- `example_inputs_fn`: Function that returns example inputs for a given size

**Returns:** `PiecewiseHybridModel` instance

### Attention Detectors

```python
from src import (
    auto_attention_detector,    # Generic auto-detection
    qwen_attention_detector,    # Qwen-specific
    llama_attention_detector,   # LLaMA/Mistral/Gemma
)
```

## Examples

### Qwen3 Example

```bash
python examples/qwen3_example.py
```

### Simple LLM (Piecewise)

```bash
python examples/simple_llm.py
```

### Multi-Layer LLM (Piecewise)

```bash
python examples/multi_layer_llm.py --num-layers 4 --hidden 256
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run Qwen3 tests (requires model)
pytest tests/test_qwen3_model.py -v

# Run end-to-end test
python run_qwen3_test.py --benchmark
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
├── src/
│   ├── __init__.py           # Public API exports
│   ├── hf_wrapper.py         # HuggingFace CUDA graph runner
│   ├── config.py             # Configuration and attention detectors
│   ├── hybrid.py             # Piecewise hybrid model
│   ├── cudagraph_backend.py  # CUDA graph piece implementation
│   ├── fx_split.py           # FX graph splitting
│   └── tree_utils.py         # Tree structure utilities
├── tests/
│   ├── test_qwen3_model.py   # Qwen3 integration tests
│   └── test_piecewise_hybrid.py  # Unit tests
├── examples/
│   ├── qwen3_example.py      # HuggingFace model example
│   ├── simple_llm.py         # Simple piecewise example
│   └── multi_layer_llm.py    # Multi-layer piecewise example
├── benchmarks/
│   └── ...                   # Benchmark scripts
├── run_qwen3_test.py         # End-to-end test script
└── README.md
```

## Design Decisions

1. **Full-Model vs Piecewise**: HuggingFace models use dynamic control flow that cannot be traced by torch.fx. We use full-model CUDA graph capture for these models, which is simpler and works universally.

2. **Bucketing**: Runtime sequences are padded to the nearest bucket size. This allows capturing a fixed set of graphs while supporting variable sequence lengths.

3. **Inference Only**: This framework is designed for inference. KV cache is disabled for simplicity.

4. **Correctness First**: Exact bucket matches produce identical output to eager mode. Non-exact sizes may have small numerical differences due to padding.

## Limitations

- Inference only (no training support)
- Single batch dimension (batch size = 1)
- KV cache disabled
- Requires CUDA

## License

MIT

## Acknowledgments

Inspired by [SGLang](https://github.com/sgl-project/sglang) piecewise CUDA graph implementation.
