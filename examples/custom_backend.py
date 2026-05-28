#!/usr/bin/env python3
"""Custom CaptureBackend implementation example.

Demonstrates how to implement a custom backend by satisfying the
CaptureBackend protocol. The EagerBackend shown here runs pieces
in standard eager mode without CUDA graph capture, useful for
benchmarking baseline performance or debugging.

Usage:
    cd /vllm-workspace/mini_piecewise
    python examples/custom_backend.py
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mini_piecewise import (
    PiecewiseHybridConfig,
    PiecePolicy,
    make_piecewise_hybrid_model,
)
from mini_piecewise.backends import CaptureBackend


class EagerBackend(nn.Module):
    """Passthrough backend that runs pieces in eager mode.

    Implements the CaptureBackend protocol without actual capture.
    Useful for establishing baseline performance, A/B comparison
    against CUDAGraphPiece, and debugging piece behavior.
    """

    def __init__(self, fn: nn.Module, **kwargs):
        super().__init__()
        self.fn = fn

    def capture_from_recorded_inputs(
        self,
        *,
        static_size: int,
        recorded_args: tuple,
        recorded_kwargs: dict,
        runtime_size: int,
    ) -> None:
        """No-op for eager backend."""
        pass

    def forward(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

    def info(self) -> dict:
        return {
            "backend": "EagerBackend",
            "fn_type": type(self.fn).__name__,
            "capture_sizes": [],
            "num_entries": 0,
        }


def eager_backend_factory(fn, config, *, graph_pool=None, device=None):
    """Factory that creates EagerBackend instances."""
    return EagerBackend(fn)


class SimpleAttention(nn.Module):
    """A simple self-attention module."""

    def __init__(self, hidden_size: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, H = x.shape
        qkv = self.qkv(x).view(T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(1)
        q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
        scale = self.head_dim ** -0.5
        attn = torch.softmax(q @ k.transpose(-2, -1) * scale, dim=-1)
        out = attn @ v
        out = out.transpose(0, 1).reshape(T, H)
        return self.out_proj(out)


class SimpleLLM(nn.Module):
    """A simple LLM model."""

    def __init__(self, vocab_size: int = 1000, hidden_size: int = 128, num_heads: int = 4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.attn = SimpleAttention(hidden_size, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size, bias=False),
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        x = self.attn(x)
        x = self.ffn(x)
        x = self.norm(x)
        return self.head(x)


def main():
    print("=" * 60)
    print(" Custom Backend Example: EagerBackend")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("\nCUDA is not available. This example requires CUDA.")
        return

    device = torch.device("cuda")
    model = SimpleLLM().to(device).eval()

    # --- Approach 1: Use CUDAGraph backend (default) ---
    print("\n--- Approach 1: CUDAGraph Backend (default) ---")
    config_cg = PiecewiseHybridConfig.from_sizes([8, 16, 32], warmup_iters=2)

    def example_inputs_fn(static_size: int):
        return (torch.zeros((static_size,), device=device, dtype=torch.long),)

    hybrid_cg = make_piecewise_hybrid_model(model, config_cg, example_inputs_fn=example_inputs_fn)
    hybrid_cg.capture()

    print("\nCUDAGraph model summary:")
    for item in hybrid_cg.items:
        mode = item.policy.value
        print(f"  {item.submod_name}: {mode}")

    # --- Approach 2: Use custom EagerBackend ---
    print("\n--- Approach 2: Custom EagerBackend ---")
    config_eager = PiecewiseHybridConfig.from_sizes(
        [8, 16, 32],
        warmup_iters=2,
        backend_factory=eager_backend_factory,
    )

    hybrid_eager = make_piecewise_hybrid_model(model, config_eager, example_inputs_fn=example_inputs_fn)
    hybrid_eager.capture()

    print("\nEagerBackend model summary:")
    summary = hybrid_eager.summary()
    for piece in summary["pieces"]:
        backend_info = piece.get("backend", {})
        backend_name = backend_info.get("backend", "none")
        print(f"  {piece['name']}: {piece['policy']} (backend={backend_name})")

    # --- Compare outputs ---
    print("\n--- Comparing Outputs ---")
    for seq_len in [5, 8, 16, 32]:
        input_ids = torch.randint(0, 1000, (seq_len,), device=device, dtype=torch.long)

        with torch.inference_mode():
            out_ref = model(input_ids)
            out_cg = hybrid_cg(input_ids)
            out_eager = hybrid_eager(input_ids)

        diff_cg = (out_ref - out_cg).abs().max().item()
        diff_eager = (out_ref - out_eager).abs().max().item()

        print(f"  seq_len={seq_len}: CUDAGraph diff={diff_cg:.6f}, EagerBackend diff={diff_eager:.6f}")

    print("\nExample completed!")


if __name__ == "__main__":
    main()