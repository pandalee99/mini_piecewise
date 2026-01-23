"""Benchmarking utilities for measuring performance and memory usage."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    name: str
    seq_len: int
    latency_ms: float
    throughput_tokens_per_sec: float
    memory_mb: float
    num_iterations: int


@dataclass
class BenchmarkTimer:
    """Timer for measuring latency with CUDA synchronization."""

    warmup_iters: int = 10
    measure_iters: int = 100
    _start_events: list = field(default_factory=list, repr=False)
    _end_events: list = field(default_factory=list, repr=False)

    def measure(
        self,
        fn,
        *args,
        warmup_iters: Optional[int] = None,
        measure_iters: Optional[int] = None,
        **kwargs,
    ) -> float:
        """Measure the average latency of a function in milliseconds.

        Args:
            fn: The function to benchmark.
            *args: Positional arguments to pass to fn.
            warmup_iters: Number of warmup iterations (overrides default).
            measure_iters: Number of measurement iterations (overrides default).
            **kwargs: Keyword arguments to pass to fn.

        Returns:
            Average latency in milliseconds.
        """
        warmup = warmup_iters if warmup_iters is not None else self.warmup_iters
        measure = measure_iters if measure_iters is not None else self.measure_iters

        # Warmup
        for _ in range(warmup):
            fn(*args, **kwargs)
        torch.cuda.synchronize()

        # Measure using CUDA events for accuracy
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(measure)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(measure)]

        for i in range(measure):
            start_events[i].record()
            fn(*args, **kwargs)
            end_events[i].record()

        torch.cuda.synchronize()

        # Calculate average latency
        latencies = [
            start_events[i].elapsed_time(end_events[i]) for i in range(measure)
        ]
        return sum(latencies) / len(latencies)


def get_gpu_memory_mb() -> float:
    """Get current GPU memory usage in MB."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / (1024 * 1024)


def get_gpu_memory_reserved_mb() -> float:
    """Get current GPU memory reserved in MB."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_reserved() / (1024 * 1024)


def warmup_cuda():
    """Warmup CUDA to avoid cold-start latency."""
    if not torch.cuda.is_available():
        return
    # Run a simple operation to initialize CUDA context
    x = torch.randn(1024, 1024, device="cuda")
    y = torch.matmul(x, x)
    torch.cuda.synchronize()
    del x, y


def format_benchmark_table(results: list[BenchmarkResult], title: str = "Benchmark Results") -> str:
    """Format benchmark results as a markdown table.

    Args:
        results: List of benchmark results.
        title: Title for the table.

    Returns:
        Formatted markdown table string.
    """
    if not results:
        return "No results to display."

    lines = []
    lines.append(f"{'=' * 50}")
    lines.append(f" {title}")
    lines.append(f"{'=' * 50}")
    lines.append("")

    # Table header
    lines.append("| Mode | Seq Len | Latency (ms) | Throughput (tok/s) | Memory (MB) |")
    lines.append("|------|---------|--------------|--------------------| ------------|")

    # Table rows
    for r in results:
        lines.append(
            f"| {r.name:<4} | {r.seq_len:>7} | {r.latency_ms:>12.3f} | {r.throughput_tokens_per_sec:>18.1f} | {r.memory_mb:>11.1f} |"
        )

    lines.append("")
    return "\n".join(lines)


def format_comparison_table(
    eager_results: list[BenchmarkResult],
    piecewise_results: list[BenchmarkResult],
    title: str = "Performance Comparison",
) -> str:
    """Format a comparison table between eager and piecewise modes.

    Args:
        eager_results: Results from eager mode.
        piecewise_results: Results from piecewise cudagraph mode.
        title: Title for the table.

    Returns:
        Formatted comparison table string.
    """
    if not eager_results or not piecewise_results:
        return "Missing results for comparison."

    lines = []
    lines.append(f"{'=' * 70}")
    lines.append(f" {title}")
    lines.append(f"{'=' * 70}")
    lines.append("")

    # Table header
    lines.append("| Seq Len | Eager (ms) | Piecewise (ms) | Speedup | Memory Delta (MB) |")
    lines.append("|---------|------------|----------------|---------|-------------------|")

    # Create lookup by seq_len
    eager_by_seq = {r.seq_len: r for r in eager_results}
    piecewise_by_seq = {r.seq_len: r for r in piecewise_results}

    all_seq_lens = sorted(set(eager_by_seq.keys()) & set(piecewise_by_seq.keys()))

    for seq_len in all_seq_lens:
        eager = eager_by_seq[seq_len]
        piecewise = piecewise_by_seq[seq_len]
        speedup = eager.latency_ms / piecewise.latency_ms if piecewise.latency_ms > 0 else 0
        mem_delta = piecewise.memory_mb - eager.memory_mb

        lines.append(
            f"| {seq_len:>7} | {eager.latency_ms:>10.3f} | {piecewise.latency_ms:>14.3f} | {speedup:>6.2f}x | {mem_delta:>+17.1f} |"
        )

    lines.append("")
    return "\n".join(lines)


def measure_capture_time(capture_fn) -> float:
    """Measure the time taken to capture CUDA graphs.

    Args:
        capture_fn: A callable that performs the capture operation.

    Returns:
        Time in seconds.
    """
    torch.cuda.synchronize()
    start = time.perf_counter()
    capture_fn()
    torch.cuda.synchronize()
    end = time.perf_counter()
    return end - start
