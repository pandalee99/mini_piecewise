#!/usr/bin/env python3
"""Benchmark ToyModel: eager vs piecewise CUDA graph.

This script benchmarks the basic ToyModel from the test suite to compare
eager execution against piecewise CUDA graph capture/replay.

Usage:
    python -m benchmarks.bench_toy_model
    python -m benchmarks.bench_toy_model --seq-lens 8 16 32 64
    python -m benchmarks.bench_toy_model --iterations 200
"""

from __future__ import annotations

import argparse
import sys

import torch
import torch.nn as nn

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


class ToyAttention(nn.Module):
    """A tiny attention-like module we want to keep eager."""

    def __init__(self, hidden: int):
        super().__init__()
        self.q = nn.Linear(hidden, hidden, bias=False)
        self.k = nn.Linear(hidden, hidden, bias=False)
        self.v = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [T, H]
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        attn = torch.softmax(q @ k.transpose(0, 1) / (q.shape[-1] ** 0.5), dim=-1)
        return attn @ v


class ToyModel(nn.Module):
    """A simple model with embedding, MLP, attention, and output projection."""

    def __init__(self, vocab: int = 256, hidden: int = 128):
        super().__init__()
        self.vocab = vocab
        self.hidden = hidden
        self.emb = nn.Embedding(vocab, hidden)
        self.mlp1 = nn.Linear(hidden, hidden * 4, bias=False)
        self.attn = ToyAttention(hidden * 4)
        self.mlp2 = nn.Linear(hidden * 4, hidden, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        # ids: [T]
        x = self.emb(ids)  # [T, H]
        x = torch.relu(self.mlp1(x))  # [T, H*4]
        x = self.attn(x)  # keep eager
        x = self.mlp2(x)  # [T, H]
        return x


def benchmark_eager(
    model: nn.Module,
    seq_lens: list[int],
    vocab: int,
    device: torch.device,
    timer: BenchmarkTimer,
) -> list[BenchmarkResult]:
    """Benchmark eager mode execution."""
    results = []

    for seq_len in seq_lens:
        ids = torch.randint(0, vocab, (seq_len,), device=device, dtype=torch.long)

        # Measure memory before
        torch.cuda.reset_peak_memory_stats()
        mem_before = get_gpu_memory_mb()

        # Benchmark
        latency_ms = timer.measure(model, ids)

        mem_after = get_gpu_memory_mb()
        throughput = seq_len / (latency_ms / 1000.0)

        results.append(
            BenchmarkResult(
                name="Eager",
                seq_len=seq_len,
                latency_ms=latency_ms,
                throughput_tokens_per_sec=throughput,
                memory_mb=mem_after,
                num_iterations=timer.measure_iters,
            )
        )

    return results


def benchmark_piecewise(
    model: nn.Module,
    seq_lens: list[int],
    vocab: int,
    device: torch.device,
    timer: BenchmarkTimer,
) -> tuple[list[BenchmarkResult], float]:
    """Benchmark piecewise CUDA graph mode execution."""
    # Create config with all seq_lens as capture sizes
    config = PiecewiseHybridConfig.from_sizes(seq_lens, warmup_iters=2)

    def example_inputs_fn(static_size: int):
        ids = torch.zeros((static_size,), device=device, dtype=torch.long)
        return (ids,)

    # Create hybrid model
    hybrid = make_piecewise_hybrid_model(
        model, config, example_inputs_fn=example_inputs_fn, device=device
    )

    # Measure capture time
    torch.cuda.reset_peak_memory_stats()
    mem_before_capture = get_gpu_memory_mb()
    capture_time = measure_capture_time(hybrid.capture)
    mem_after_capture = get_gpu_memory_mb()

    print(f"\nCapture Statistics:")
    print(f"  Capture time: {capture_time:.3f}s")
    print(f"  Memory before: {mem_before_capture:.1f} MB")
    print(f"  Memory after: {mem_after_capture:.1f} MB")
    print(f"  Memory delta: {mem_after_capture - mem_before_capture:+.1f} MB")

    results = []

    for seq_len in seq_lens:
        ids = torch.randint(0, vocab, (seq_len,), device=device, dtype=torch.long)

        # Benchmark
        latency_ms = timer.measure(hybrid, ids)

        mem_current = get_gpu_memory_mb()
        throughput = seq_len / (latency_ms / 1000.0)

        results.append(
            BenchmarkResult(
                name="Piece",
                seq_len=seq_len,
                latency_ms=latency_ms,
                throughput_tokens_per_sec=throughput,
                memory_mb=mem_current,
                num_iterations=timer.measure_iters,
            )
        )

    return results, capture_time


def main():
    parser = argparse.ArgumentParser(description="Benchmark ToyModel")
    parser.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=[8, 16, 32, 64],
        help="Sequence lengths to benchmark",
    )
    parser.add_argument(
        "--vocab", type=int, default=256, help="Vocabulary size"
    )
    parser.add_argument(
        "--hidden", type=int, default=128, help="Hidden dimension"
    )
    parser.add_argument(
        "--warmup", type=int, default=10, help="Warmup iterations"
    )
    parser.add_argument(
        "--iterations", type=int, default=100, help="Measurement iterations"
    )
    args = parser.parse_args()

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        sys.exit(1)

    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(device)}")

    # Initialize
    warmup_cuda()
    torch.manual_seed(42)

    # Create model
    model = ToyModel(vocab=args.vocab, hidden=args.hidden).to(device).eval()

    print(f"\nModel: ToyModel (vocab={args.vocab}, hidden={args.hidden})")
    print(f"Sequence lengths: {args.seq_lens}")
    print(f"Warmup iterations: {args.warmup}")
    print(f"Measurement iterations: {args.iterations}")

    timer = BenchmarkTimer(warmup_iters=args.warmup, measure_iters=args.iterations)

    # Benchmark eager mode
    print("\n--- Benchmarking Eager Mode ---")
    eager_results = benchmark_eager(
        model, args.seq_lens, args.vocab, device, timer
    )

    # Benchmark piecewise mode
    print("\n--- Benchmarking Piecewise CUDA Graph Mode ---")
    piecewise_results, capture_time = benchmark_piecewise(
        model, args.seq_lens, args.vocab, device, timer
    )

    # Print comparison
    print("\n" + format_comparison_table(
        eager_results,
        piecewise_results,
        title=f"ToyModel Benchmark (vocab={args.vocab}, hidden={args.hidden})",
    ))

    # Verify correctness
    print("--- Verifying Correctness ---")
    all_correct = True
    for seq_len in args.seq_lens:
        ids = torch.randint(0, args.vocab, (seq_len,), device=device, dtype=torch.long)

        # Recreate hybrid for verification
        config = PiecewiseHybridConfig.from_sizes(args.seq_lens, warmup_iters=2)
        hybrid = make_piecewise_hybrid_model(
            model, config, example_inputs_fn=lambda s: (torch.zeros((s,), device=device, dtype=torch.long),)
        )
        hybrid.capture()

        with torch.inference_mode():
            y_eager = model(ids)
            y_piecewise = hybrid(ids)

        if torch.allclose(y_eager, y_piecewise, rtol=1e-4, atol=1e-4):
            print(f"  seq_len={seq_len}: PASS")
        else:
            print(f"  seq_len={seq_len}: FAIL")
            max_diff = (y_eager - y_piecewise).abs().max().item()
            print(f"    Max difference: {max_diff}")
            all_correct = False

    if all_correct:
        print("\nAll correctness checks passed!")
    else:
        print("\nSome correctness checks failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
