#!/usr/bin/env python3
"""Multi-Layer LLM Example.

This example demonstrates using min_piecewise with a multi-layer LLM model,
showing how the framework handles multiple attention layers.

Key points:
- Each attention layer becomes a separate "eager" piece
- All other computations (embedding, MLP, norms) are grouped into CUDA graph pieces
- The framework automatically handles the stitching

Usage:
    python -m examples.multi_layer_llm
    python -m examples.multi_layer_llm --num-layers 4 --hidden 256
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn

# Support running from different locations
try:
    from min_piecewise import (
        PiecewiseHybridConfig,
        make_piecewise_hybrid_model,
    )
except ImportError:
    from .. import (
        PiecewiseHybridConfig,
        make_piecewise_hybrid_model,
    )


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return self.weight * x * torch.rsqrt(variance + self.eps)


class CausalSelfAttention(nn.Module):
    """Causal self-attention module.

    Contains 'Attention' in the name for automatic detection.
    """

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, H = x.shape

        q = self.q_proj(x).view(T, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(T, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(T, self.num_heads, self.head_dim)

        # [num_heads, T, head_dim]
        q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)

        # Attention with causal mask
        scale = self.head_dim ** -0.5
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))
        attn_weights = torch.softmax(attn_weights, dim=-1)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(0, 1).reshape(T, H)
        return self.out_proj(out)


class MLP(nn.Module):
    """Feed-forward network with SiLU activation."""

    def __init__(self, hidden_size: int, intermediate_size: int = None):
        super().__init__()
        if intermediate_size is None:
            intermediate_size = int(hidden_size * 8 / 3)
            intermediate_size = ((intermediate_size + 63) // 64) * 64

        self.gate = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(torch.nn.functional.silu(self.gate(x)) * self.up(x))


class DecoderLayer(nn.Module):
    """A single decoder layer."""

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.input_norm = RMSNorm(hidden_size)
        self.self_attn = CausalSelfAttention(hidden_size, num_heads)
        self.post_attn_norm = RMSNorm(hidden_size)
        self.mlp = MLP(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm with residual
        x = x + self.self_attn(self.input_norm(x))
        x = x + self.mlp(self.post_attn_norm(x))
        return x


class MultiLayerLLM(nn.Module):
    """Multi-layer LLM model."""

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            DecoderLayer(hidden_size, num_heads) for _ in range(num_layers)
        ])
        self.norm = RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)


def main():
    parser = argparse.ArgumentParser(description="Multi-Layer LLM Example")
    parser.add_argument("--vocab-size", type=int, default=10000, help="Vocabulary size")
    parser.add_argument("--hidden", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--num-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[16, 32, 64, 128],
                        help="Sequence lengths to test")
    args = parser.parse_args()

    print("=" * 60)
    print(" min_piecewise: Multi-Layer LLM Example")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("\nCUDA is not available. This example requires CUDA.")
        return

    device = torch.device("cuda")
    print(f"\nDevice: {torch.cuda.get_device_name(device)}")

    # Create model
    model = MultiLayerLLM(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
    ).to(device).eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: MultiLayerLLM")
    print(f"  vocab_size: {args.vocab_size}")
    print(f"  hidden_size: {args.hidden}")
    print(f"  num_heads: {args.num_heads}")
    print(f"  num_layers: {args.num_layers}")
    print(f"  total parameters: {num_params:,}")

    # Create config and hybrid model
    capture_sizes = sorted(args.seq_lens)
    config = PiecewiseHybridConfig.from_sizes(capture_sizes, warmup_iters=2)

    def example_inputs_fn(static_size: int):
        return (torch.zeros((static_size,), device=device, dtype=torch.long),)

    print(f"\nBuilding hybrid model with capture_sizes={capture_sizes}...")
    hybrid = make_piecewise_hybrid_model(
        model, config, example_inputs_fn=example_inputs_fn, device=device
    )

    # Show split structure
    print(f"\nFX split structure ({len(hybrid.items)} pieces):")
    attn_count = sum(1 for item in hybrid.items if item.is_attention_piece)
    cuda_count = len(hybrid.items) - attn_count
    print(f"  Attention pieces (eager): {attn_count}")
    print(f"  CUDA Graph pieces: {cuda_count}")

    for item in hybrid.items:
        mode = "EAGER" if item.is_attention_piece else "GRAPH"
        print(f"    {item.submod_name}: [{mode}]")

    # Capture
    print("\nCapturing CUDA graphs...")
    start = time.perf_counter()
    hybrid.capture()
    capture_time = time.perf_counter() - start
    print(f"Capture completed in {capture_time:.2f}s")

    # Test inference
    print("\n--- Testing Inference ---")
    print(f"{'Seq Len':>8} {'Bucket':>8} {'Status':>8} {'Max Diff':>12}")
    print("-" * 40)

    for seq_len in [10, 16, 25, 32, 50, 64, 100, 128]:
        if seq_len > max(capture_sizes):
            print(f"{seq_len:>8} {'N/A':>8} {'SKIP':>8} {'(too large)':>12}")
            continue

        input_ids = torch.randint(0, args.vocab_size, (seq_len,), device=device, dtype=torch.long)

        # Compare outputs
        with torch.inference_mode():
            out_hybrid = hybrid(input_ids)
            out_eager = model(input_ids)

        max_diff = (out_hybrid - out_eager).abs().max().item()
        is_correct = max_diff < 1e-3
        status = "PASS" if is_correct else "FAIL"

        bucket = min(s for s in capture_sizes if s >= seq_len)
        print(f"{seq_len:>8} {bucket:>8} {status:>8} {max_diff:>12.6f}")

    # Benchmark
    print("\n--- Quick Latency Comparison ---")

    # Warmup
    for _ in range(10):
        _ = model(torch.randint(0, args.vocab_size, (64,), device=device, dtype=torch.long))
        _ = hybrid(torch.randint(0, args.vocab_size, (64,), device=device, dtype=torch.long))
    torch.cuda.synchronize()

    for seq_len in capture_sizes:
        input_ids = torch.randint(0, args.vocab_size, (seq_len,), device=device, dtype=torch.long)

        # Measure eager
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(50):
            _ = model(input_ids)
        torch.cuda.synchronize()
        eager_time = (time.perf_counter() - start) / 50 * 1000

        # Measure hybrid
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(50):
            _ = hybrid(input_ids)
        torch.cuda.synchronize()
        hybrid_time = (time.perf_counter() - start) / 50 * 1000

        speedup = eager_time / hybrid_time if hybrid_time > 0 else 0
        print(f"  seq_len={seq_len:3d}: eager={eager_time:.3f}ms, hybrid={hybrid_time:.3f}ms, speedup={speedup:.2f}x")

    print("\nExample completed!")


if __name__ == "__main__":
    main()
