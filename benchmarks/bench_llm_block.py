#!/usr/bin/env python3
"""Benchmark LLM-style Transformer Block: eager vs piecewise CUDA graph.

This script benchmarks a more realistic LLM-style transformer block to show
the performance benefits of piecewise CUDA graph optimization.

Usage:
    python -m benchmarks.bench_llm_block
    python -m benchmarks.bench_llm_block --hidden 1024 --num-heads 16
    python -m benchmarks.bench_llm_block --num-layers 4
"""

from __future__ import annotations

import argparse
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# Support running from different locations
try:
    from min_piecewise import PiecewiseHybridConfig, make_piecewise_hybrid_model
    from min_piecewise.benchmarks.utils import (
        BenchmarkResult,
        BenchmarkTimer,
        format_comparison_table,
        get_gpu_memory_mb,
        measure_capture_time,
        warmup_cuda,
    )
except ImportError:
    from .. import PiecewiseHybridConfig, make_piecewise_hybrid_model
    from .utils import (
        BenchmarkResult,
        BenchmarkTimer,
        format_comparison_table,
        get_gpu_memory_mb,
        measure_capture_time,
        warmup_cuda,
    )


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (simplified)."""

    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply rotary position embedding to query and key tensors."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LLMAttention(nn.Module):
    """Multi-head attention with rotary embedding (simplified LLM style).

    This module is intentionally kept eager-friendly for piecewise CUDA graph.
    """

    def __init__(self, hidden_size: int, num_heads: int, max_seq_len: int = 2048):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.rotary_emb = RotaryEmbedding(self.head_dim, max_seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [T, H]
        seq_len = x.shape[0]

        q = self.q_proj(x).view(seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(seq_len, self.num_heads, self.head_dim)

        # Apply rotary embedding
        cos, sin = self.rotary_emb(x, seq_len)
        cos = cos.unsqueeze(1)  # [T, 1, head_dim]
        sin = sin.unsqueeze(1)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Transpose for attention: [num_heads, T, head_dim]
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape back: [T, H]
        attn_output = attn_output.transpose(0, 1).reshape(seq_len, self.hidden_size)
        return self.o_proj(attn_output)


class SwiGLU(nn.Module):
    """SwiGLU activation function used in modern LLMs."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LLMBlock(nn.Module):
    """A single LLM decoder block (simplified LLaMA-style)."""

    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int, max_seq_len: int = 2048):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size)
        self.self_attn = LLMAttention(hidden_size, num_heads, max_seq_len)
        self.post_attention_layernorm = RMSNorm(hidden_size)
        self.mlp = SwiGLU(hidden_size, intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm architecture
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x)
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return x


class SimpleLLM(nn.Module):
    """A simple multi-layer LLM for benchmarking."""

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        intermediate_size: int = None,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        if intermediate_size is None:
            intermediate_size = int(hidden_size * 8 / 3)
            # Round to nearest multiple of 64 for efficiency
            intermediate_size = ((intermediate_size + 63) // 64) * 64

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            LLMBlock(hidden_size, num_heads, intermediate_size, max_seq_len)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(hidden_size)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: [T]
        x = self.embed_tokens(input_ids)  # [T, H]

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        return x


def benchmark_eager(
    model: nn.Module,
    seq_lens: list[int],
    vocab_size: int,
    device: torch.device,
    timer: BenchmarkTimer,
) -> list[BenchmarkResult]:
    """Benchmark eager mode execution."""
    results = []

    for seq_len in seq_lens:
        ids = torch.randint(0, vocab_size, (seq_len,), device=device, dtype=torch.long)

        torch.cuda.reset_peak_memory_stats()
        latency_ms = timer.measure(model, ids)
        mem = get_gpu_memory_mb()
        throughput = seq_len / (latency_ms / 1000.0)

        results.append(
            BenchmarkResult(
                name="Eager",
                seq_len=seq_len,
                latency_ms=latency_ms,
                throughput_tokens_per_sec=throughput,
                memory_mb=mem,
                num_iterations=timer.measure_iters,
            )
        )

    return results


def benchmark_piecewise(
    model: nn.Module,
    seq_lens: list[int],
    vocab_size: int,
    device: torch.device,
    timer: BenchmarkTimer,
) -> tuple[list[BenchmarkResult], float]:
    """Benchmark piecewise CUDA graph mode execution."""
    config = PiecewiseHybridConfig.from_sizes(seq_lens, warmup_iters=2)

    def example_inputs_fn(static_size: int):
        ids = torch.zeros((static_size,), device=device, dtype=torch.long)
        return (ids,)

    hybrid = make_piecewise_hybrid_model(
        model, config, example_inputs_fn=example_inputs_fn, device=device
    )

    # Measure capture
    torch.cuda.reset_peak_memory_stats()
    mem_before = get_gpu_memory_mb()
    capture_time = measure_capture_time(hybrid.capture)
    mem_after = get_gpu_memory_mb()

    print(f"\nCapture Statistics:")
    print(f"  Capture time: {capture_time:.3f}s")
    print(f"  Memory before: {mem_before:.1f} MB")
    print(f"  Memory after: {mem_after:.1f} MB")
    print(f"  Memory delta: {mem_after - mem_before:+.1f} MB")
    print(f"  Number of pieces: {len(hybrid.items)}")
    for item in hybrid.items:
        status = "attention (eager)" if item.is_attention_piece else "cudagraph"
        print(f"    {item.submod_name}: {status}")

    results = []

    for seq_len in seq_lens:
        ids = torch.randint(0, vocab_size, (seq_len,), device=device, dtype=torch.long)

        latency_ms = timer.measure(hybrid, ids)
        mem = get_gpu_memory_mb()
        throughput = seq_len / (latency_ms / 1000.0)

        results.append(
            BenchmarkResult(
                name="Piece",
                seq_len=seq_len,
                latency_ms=latency_ms,
                throughput_tokens_per_sec=throughput,
                memory_mb=mem,
                num_iterations=timer.measure_iters,
            )
        )

    return results, capture_time


def main():
    parser = argparse.ArgumentParser(description="Benchmark LLM Block")
    parser.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=[32, 64, 128, 256],
        help="Sequence lengths to benchmark",
    )
    parser.add_argument("--vocab-size", type=int, default=32000, help="Vocabulary size")
    parser.add_argument("--hidden", type=int, default=512, help="Hidden dimension")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of transformer layers")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--iterations", type=int, default=100, help="Measurement iterations")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        sys.exit(1)

    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(device)}")

    warmup_cuda()
    torch.manual_seed(42)

    model = SimpleLLM(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
    ).to(device).eval()

    # Print model info
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: SimpleLLM")
    print(f"  vocab_size: {args.vocab_size}")
    print(f"  hidden_size: {args.hidden}")
    print(f"  num_heads: {args.num_heads}")
    print(f"  num_layers: {args.num_layers}")
    print(f"  parameters: {num_params:,}")
    print(f"\nBenchmark config:")
    print(f"  Sequence lengths: {args.seq_lens}")
    print(f"  Warmup iterations: {args.warmup}")
    print(f"  Measurement iterations: {args.iterations}")

    timer = BenchmarkTimer(warmup_iters=args.warmup, measure_iters=args.iterations)

    # Benchmark eager
    print("\n--- Benchmarking Eager Mode ---")
    eager_results = benchmark_eager(model, args.seq_lens, args.vocab_size, device, timer)

    # Benchmark piecewise
    print("\n--- Benchmarking Piecewise CUDA Graph Mode ---")
    piecewise_results, capture_time = benchmark_piecewise(
        model, args.seq_lens, args.vocab_size, device, timer
    )

    # Print comparison
    print("\n" + format_comparison_table(
        eager_results,
        piecewise_results,
        title=f"SimpleLLM Benchmark (hidden={args.hidden}, layers={args.num_layers})",
    ))

    # Verify correctness
    print("--- Verifying Correctness ---")
    all_correct = True

    config = PiecewiseHybridConfig.from_sizes(args.seq_lens, warmup_iters=2)
    hybrid = make_piecewise_hybrid_model(
        model, config,
        example_inputs_fn=lambda s: (torch.zeros((s,), device=device, dtype=torch.long),)
    )
    hybrid.capture()

    for seq_len in args.seq_lens:
        ids = torch.randint(0, args.vocab_size, (seq_len,), device=device, dtype=torch.long)

        with torch.inference_mode():
            y_eager = model(ids)
            y_piecewise = hybrid(ids)

        if torch.allclose(y_eager, y_piecewise, rtol=1e-3, atol=1e-3):
            print(f"  seq_len={seq_len}: PASS")
        else:
            max_diff = (y_eager - y_piecewise).abs().max().item()
            print(f"  seq_len={seq_len}: FAIL (max_diff={max_diff:.6f})")
            all_correct = False

    if all_correct:
        print("\nAll correctness checks passed!")
    else:
        print("\nSome correctness checks failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
