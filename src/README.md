# min_piecewise

**Minimal Piecewise CUDA Graph Framework for LLM Inference**

A simple, portable implementation of piecewise CUDA graph optimization for LLM inference. This framework provides the core functionality of SGLang/vLLM-style piecewise compilation in a minimal, easy-to-understand package.

## Features

- **Automatic Graph Splitting**: Uses `torch.fx` to automatically split models at attention boundaries
- **Hybrid Execution**: Keeps attention modules eager while capturing CUDA graphs for other parts
- **Bucket-based Dispatch**: Supports multiple sequence lengths through bucket-based graph selection
- **Zero Dependencies**: Only requires PyTorch >= 2.0
- **Easy Integration**: Simple API for integrating into existing LLM projects

## Quick Start

### Installation

```bash
# Install in development mode
cd min_piecewise
pip install -e .

# Or install with dev dependencies
pip install -e ".[dev]"
```

### Basic Usage

```python
import torch
from min_piecewise import PiecewiseHybridConfig, make_piecewise_hybrid_model

# Your model with attention modules
model = YourLLM().to("cuda").eval()

# Step 1: Create configuration with bucket sizes
config = PiecewiseHybridConfig.from_sizes([32, 64, 128, 256])

# Step 2: Define how to create example inputs for each size
def example_inputs_fn(static_size: int):
    return (torch.zeros((static_size,), device="cuda", dtype=torch.long),)

# Step 3: Build hybrid model
hybrid = make_piecewise_hybrid_model(model, config, example_inputs_fn=example_inputs_fn)

# Step 4: Capture CUDA graphs
hybrid.capture()

# Step 5: Run inference (automatically uses CUDA graph replay)
output = hybrid(input_ids)
```

## How It Works

### 1. Graph Splitting

The framework traces your model using `torch.fx` and splits it at attention module boundaries:

```
Original Model:
[Embedding] -> [LayerNorm] -> [Attention] -> [LayerNorm] -> [MLP] -> [Output]

Split Graph:
[submod_0: Embedding+LayerNorm] -> [submod_1: Attention] -> [submod_2: LayerNorm+MLP+Output]
      (CUDA Graph)                    (Eager)                    (CUDA Graph)
```

### 2. Attention Detection

Modules are identified as "attention" by:
- Class name containing "Attention" or "attn"
- Module qualified name containing "attn" or "attention"
- Being a `torch.nn.MultiheadAttention` instance

You can customize this by providing your own `is_attention_module` function:

```python
def my_attention_detector(module, qualname):
    return "my_custom_attn" in qualname

config = PiecewiseHybridConfig.from_sizes(
    [32, 64, 128],
    is_attention_module=my_attention_detector
)
```

### 3. Bucket-based Dispatch

Different input sizes are routed to pre-captured graphs:

```
Input size 30 -> Bucket 32 (captures graph for size 32, slices output to 30)
Input size 50 -> Bucket 64
Input size 100 -> Bucket 128
```

## API Reference

### `PiecewiseHybridConfig`

Configuration for the hybrid model.

```python
PiecewiseHybridConfig(
    capture_sizes: tuple[int, ...],      # Bucket sizes (must be sorted, ascending)
    warmup_iters: int = 2,               # Warmup iterations before capture
    zero_pad_inputs: bool = True,        # Zero-pad smaller inputs
    runtime_size_fn: Callable = ...,     # Extract runtime size from inputs
    is_attention_module: Callable = ..., # Detect attention modules
    check_input_addresses: bool = False, # Debug: verify input addresses
)

# Convenience constructor
PiecewiseHybridConfig.from_sizes([32, 64, 128], warmup_iters=2)
```

### `make_piecewise_hybrid_model`

Build a hybrid model from an existing model.

```python
hybrid = make_piecewise_hybrid_model(
    model: nn.Module,                    # Your model
    config: PiecewiseHybridConfig,       # Configuration
    example_inputs_fn: Callable,         # Function to generate example inputs
    device: torch.device = None,         # Target device
    graph_pool: Any = None,              # CUDA graph memory pool
)
```

### `PiecewiseHybridModel`

The hybrid model wrapper.

```python
hybrid.capture()          # Capture CUDA graphs for all buckets
hybrid.items              # List of split pieces with metadata
hybrid.split_gm           # The underlying split FX GraphModule
output = hybrid(*args)    # Forward pass with automatic graph replay
```

## Examples

### Simple Example

```bash
python -m examples.llm_block
```

### Multi-Layer LLM

```bash
python -m examples.multi_layer_llm --num-layers 4 --hidden 256
```

## Benchmarks

### ToyModel Benchmark

```bash
python -m benchmarks.bench_toy_model
```

### LLM Block Benchmark

```bash
python -m benchmarks.bench_llm_block --hidden 512 --num-layers 2
```

### Expected Results

On a typical GPU, you can expect:
- **2-4x latency reduction** for small sequence lengths
- **1.5-2x latency reduction** for larger sequence lengths
- **~10-20% additional memory** for captured graphs

## Migration Guide

### Integrating into Your Project

1. **Identify Attention Modules**

   Ensure your attention modules have "Attention" or "attn" in their class names, or provide a custom detector.

2. **Choose Bucket Sizes**

   Select sizes that cover your expected input range. Common choices:
   - Decode: `[1, 2, 4, 8, 16, 32]`
   - Prefill: `[128, 256, 512, 1024, 2048]`

3. **Handle Dynamic Shapes**

   The framework assumes `dim0` is the dynamic dimension. For different layouts:
   ```python
   def custom_runtime_size_fn(args, kwargs):
       return args[0].shape[1]  # For [B, T, H] layout
   ```

4. **Memory Considerations**

   Each bucket captures a separate graph. Reduce bucket count if memory is constrained.

### Common Issues

**Q: FX tracing fails**

A: Some operations aren't traceable. Try:
- Making attention modules leaf modules (they're kept as `call_module` nodes)
- Avoiding dynamic control flow in non-attention parts

**Q: Results don't match eager mode**

A: Check that:
- Attention modules are correctly identified (use `hybrid.items` to verify)
- Input sizes don't exceed max bucket size

**Q: Out of memory during capture**

A: Capture happens from largest to smallest bucket. Reduce max bucket size or use fewer buckets.

## Project Structure

```
min_piecewise/
├── __init__.py           # Public API
├── config.py             # PiecewiseHybridConfig
├── hybrid.py             # PiecewiseHybridModel, make_piecewise_hybrid_model
├── cudagraph_backend.py  # CUDAGraphPiece (capture/replay logic)
├── fx_split.py           # FX graph splitting
├── tree_utils.py         # Nested structure utilities
├── errors.py             # Custom exceptions
├── examples/             # Usage examples
│   ├── llm_block.py
│   └── multi_layer_llm.py
├── benchmarks/           # Performance benchmarks
│   ├── utils.py
│   ├── bench_toy_model.py
│   └── bench_llm_block.py
├── test_min_piecewise_hybrid.py  # Unit tests
├── pyproject.toml        # Package configuration
└── README.md             # This file
```

## Running Tests

```bash
cd min_piecewise
pip install -e ".[dev]"
pytest -v
```

## License

Apache-2.0

## Acknowledgments

This implementation is inspired by:
- [SGLang](https://github.com/sgl-project/sglang) piecewise CUDA graph
- [vLLM](https://github.com/vllm-project/vllm) compilation framework

The goal is to provide a minimal, educational implementation that can be easily understood and adapted.
