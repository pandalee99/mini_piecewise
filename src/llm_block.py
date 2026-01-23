#!/usr/bin/env python3
"""Simple LLM Block Example.

This example demonstrates the basic usage of min_piecewise with a simple
LLM-style transformer block. It shows how:

1. The framework automatically detects attention modules
2. Attention is kept eager while other parts use CUDA graphs
3. Different sequence lengths use different captured graphs (buckets)

Usage:
    python -m examples.llm_block
"""

from __future__ import annotations

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


class SimpleAttention(nn.Module):
    """A simple self-attention module.

    This module contains 'Attention' in its class name, so it will be
    automatically detected and kept eager.
    """

    def __init__(self, hidden_size: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [T, H]
        T, H = x.shape
        qkv = self.qkv(x).view(T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(1)  # Each: [T, num_heads, head_dim]

        # Transpose for attention: [num_heads, T, head_dim]
        q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = torch.softmax(q @ k.transpose(-2, -1) * scale, dim=-1)
        out = attn @ v

        # Reshape back: [T, H]
        out = out.transpose(0, 1).reshape(T, H)
        return self.out_proj(out)


class FeedForward(nn.Module):
    """Feed-forward network (MLP)."""

    def __init__(self, hidden_size: int, intermediate_size: int = None):
        super().__init__()
        if intermediate_size is None:
            intermediate_size = hidden_size * 4
        self.up = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(torch.relu(self.up(x)))


class TransformerBlock(nn.Module):
    """A simple transformer block with pre-norm architecture."""

    def __init__(self, hidden_size: int, num_heads: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = SimpleAttention(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = FeedForward(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm residual connections
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class SimpleLLM(nn.Module):
    """A simple LLM model for demonstration."""

    def __init__(self, vocab_size: int = 1000, hidden_size: int = 128, num_heads: int = 4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.block = TransformerBlock(hidden_size, num_heads)
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: [T]
        x = self.embed(input_ids)  # [T, H]
        x = self.block(x)
        x = self.norm(x)
        logits = self.head(x)  # [T, vocab_size]
        return logits


def main():
    print("=" * 60)
    print(" min_piecewise: Simple LLM Block Example")
    print("=" * 60)

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("\nCUDA is not available. This example requires CUDA.")
        print("Running CPU-only demo to show API usage...\n")
        device = torch.device("cpu")
        run_cpu_demo()
        return

    device = torch.device("cuda")
    print(f"\nDevice: {torch.cuda.get_device_name(device)}")

    # Create model
    model = SimpleLLM(vocab_size=1000, hidden_size=128, num_heads=4).to(device).eval()
    print(f"Model: SimpleLLM (vocab=1000, hidden=128, heads=4)")

    # Define capture sizes (sequence length buckets)
    capture_sizes = [8, 16, 32, 64]
    print(f"Capture sizes: {capture_sizes}")

    # Step 1: Create configuration
    config = PiecewiseHybridConfig.from_sizes(
        capture_sizes,
        warmup_iters=2,        # Number of warmup iterations before capture
        zero_pad_inputs=True,  # Zero-pad inputs smaller than bucket size
    )
    print(f"\nConfiguration:")
    print(f"  capture_sizes: {config.capture_sizes}")
    print(f"  warmup_iters: {config.warmup_iters}")
    print(f"  zero_pad_inputs: {config.zero_pad_inputs}")

    # Step 2: Create example inputs function
    # This function generates example inputs for each capture size
    def example_inputs_fn(static_size: int):
        """Generate example inputs for a given sequence length."""
        input_ids = torch.zeros((static_size,), device=device, dtype=torch.long)
        return (input_ids,)  # Return as tuple of args

    # Step 3: Build the hybrid model
    print("\nBuilding hybrid model...")
    hybrid = make_piecewise_hybrid_model(
        model,
        config,
        example_inputs_fn=example_inputs_fn,
        device=device,
    )

    # Step 4: Examine the split structure
    print(f"\nFX split structure ({len(hybrid.items)} pieces):")
    for item in hybrid.items:
        mode = "EAGER (attention)" if item.is_attention_piece else "CUDA Graph"
        print(f"  {item.submod_name}: {mode}")

    # Step 5: Capture CUDA graphs
    print("\nCapturing CUDA graphs...")
    hybrid.capture()
    print("Capture complete!")

    # Step 6: Run inference with different sequence lengths
    print("\n--- Running Inference ---")
    test_seq_lens = [5, 8, 12, 16, 30, 32, 50, 64]

    for seq_len in test_seq_lens:
        # Create random input
        input_ids = torch.randint(0, 1000, (seq_len,), device=device, dtype=torch.long)

        # Run with hybrid model (uses CUDA graph replay)
        with torch.inference_mode():
            output_hybrid = hybrid(input_ids)

        # Run with original model for comparison
        with torch.inference_mode():
            output_eager = model(input_ids)

        # Verify correctness
        is_correct = torch.allclose(output_hybrid, output_eager, rtol=1e-4, atol=1e-4)
        status = "PASS" if is_correct else "FAIL"

        # Find which bucket was used
        bucket_size = min(s for s in capture_sizes if s >= seq_len)
        print(f"  seq_len={seq_len:2d} -> bucket={bucket_size:2d}: {status}")

    print("\nExample completed successfully!")


def run_cpu_demo():
    """Run a CPU-only demo to show API usage."""
    print("Creating model on CPU...")

    model = SimpleLLM(vocab_size=100, hidden_size=32, num_heads=2)

    print("\nNote: Piecewise CUDA Graph requires CUDA.")
    print("On CPU, we can only demonstrate the FX tracing and splitting.\n")

    # Show model structure
    print("Model structure:")
    for name, module in model.named_modules():
        if name:
            is_attn = "Attention" in module.__class__.__name__
            marker = " <- will be kept EAGER" if is_attn else ""
            print(f"  {name}: {module.__class__.__name__}{marker}")

    print("\nAPI usage example (would require CUDA):")
    print("""
    # 1. Create config
    config = PiecewiseHybridConfig.from_sizes([8, 16, 32])

    # 2. Define example inputs function
    def example_inputs_fn(static_size):
        return (torch.zeros((static_size,), device="cuda", dtype=torch.long),)

    # 3. Build hybrid model
    hybrid = make_piecewise_hybrid_model(model, config, example_inputs_fn=example_inputs_fn)

    # 4. Capture CUDA graphs
    hybrid.capture()

    # 5. Run inference
    output = hybrid(input_ids)
    """)


if __name__ == "__main__":
    main()
