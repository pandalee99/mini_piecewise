"""Tests for CUDA graph optimization with the Qwen3 model.

Validates correctness, determinism, and shape consistency of
CudaGraphRunner with a real Qwen3-0.6B-Base model, plus
unit tests for the attention detection and PiecePolicy systems.

Usage:
    cd mini_piecewise
    pytest -v tests/test_qwen3_model.py

Or run directly:
    python tests/test_qwen3_model.py
"""

from __future__ import annotations

import sys
import torch

# Check dependencies
try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False

try:
    import transformers
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# Check CUDA
HAS_CUDA = torch.cuda.is_available()


class TestAttentionDetectors:
    """Test attention detection functions."""

    def test_qwen_attention_detector(self):
        """Test Qwen attention detector."""
        from mini_piecewise.config import qwen_attention_detector

        class Qwen3Attention(torch.nn.Module):
            pass

        class Qwen3MLP(torch.nn.Module):
            pass

        attn = Qwen3Attention()
        mlp = Qwen3MLP()

        assert qwen_attention_detector(attn, "model.layers.0.self_attn")
        assert not qwen_attention_detector(mlp, "model.layers.0.mlp")

    def test_llama_attention_detector(self):
        """Test LLaMA attention detector."""
        from mini_piecewise.config import llama_attention_detector

        class LlamaAttention(torch.nn.Module):
            pass

        class LlamaMLP(torch.nn.Module):
            pass

        attn = LlamaAttention()
        mlp = LlamaMLP()

        assert llama_attention_detector(attn, "model.layers.0.self_attn")
        assert not llama_attention_detector(mlp, "model.layers.0.mlp")

    def test_auto_attention_detector(self):
        """Test auto attention detector."""
        from mini_piecewise.config import auto_attention_detector

        class SomeAttention(torch.nn.Module):
            pass

        class SomeAttn(torch.nn.Module):
            pass

        class SomeMLP(torch.nn.Module):
            pass

        class AttentionMask(torch.nn.Module):
            pass

        assert auto_attention_detector(SomeAttention(), "block.attn")
        assert auto_attention_detector(SomeAttn(), "block.self_attn")
        assert not auto_attention_detector(SomeMLP(), "block.mlp")
        # AttentionMask should be excluded
        assert not auto_attention_detector(AttentionMask(), "block.attention_mask")

    def test_piece_policy_integration(self):
        """Test that attention_piece_selector returns PiecePolicy."""
        from mini_piecewise.config import attention_piece_selector, PiecePolicy

        class Qwen3Attention(torch.nn.Module):
            pass

        class MLPModule(torch.nn.Module):
            pass

        attn = Qwen3Attention()
        mlp = MLPModule()

        assert attention_piece_selector(attn, "model.layers.0.self_attn") == PiecePolicy.EAGER
        assert attention_piece_selector(mlp, "model.layers.0.mlp") == PiecePolicy.CAPTURE


class TestQwen3CudaGraph:
    """Test suite for Qwen3 with CUDA graph optimization."""

    @staticmethod
    def skip_if_no_deps():
        if not HAS_TRANSFORMERS:
            return "transformers not available"
        if not HAS_CUDA:
            return "CUDA not available"
        return None

    def test_attention_detection(self):
        """Test that attention modules are correctly detected."""
        skip = self.skip_if_no_deps()
        if skip:
            print(f"SKIP: {skip}")
            return

        from transformers import AutoModelForCausalLM
        from mini_piecewise import get_attention_modules

        model = AutoModelForCausalLM.from_pretrained(
            "/vllm-workspace/Qwen3-0.6B-Base",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).cuda().eval()

        attn_modules = get_attention_modules(model)

        # Qwen3-0.6B has 28 layers, each with one attention module
        assert len(attn_modules) >= 28, f"Expected at least 28 attention modules, got {len(attn_modules)}"

        # Check that detected modules contain "attn" in their name
        for name in attn_modules:
            assert "attn" in name.lower() or "attention" in name.lower(), \
                f"Module {name} doesn't look like attention"

        del model
        torch.cuda.empty_cache()

    def test_cudagraph_build_and_capture(self):
        """Test that CUDA graph runner can be built and captured."""
        skip = self.skip_if_no_deps()
        if skip:
            print(f"SKIP: {skip}")
            return

        from transformers import AutoModelForCausalLM
        from mini_piecewise import cudagraph_compile_hf, CudaGraphRunner

        model = AutoModelForCausalLM.from_pretrained(
            "/vllm-workspace/Qwen3-0.6B-Base",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).cuda().eval()

        capture_sizes = [32, 64]
        runner = cudagraph_compile_hf(model, capture_sizes)

        # Verify it's a CudaGraphRunner
        assert isinstance(runner, CudaGraphRunner)

        # Check initial state
        assert not runner._captured, "Should not be captured initially"

        # Capture
        runner.capture()

        # Check captured state
        assert runner._captured, "Should be captured after capture()"
        assert len(runner._entries) == len(capture_sizes), \
            f"Should have {len(capture_sizes)} entries"

        # Test summary
        summary = runner.summary()
        assert summary["captured"] is True
        assert summary["num_entries"] == 2

        del model, runner
        torch.cuda.empty_cache()

    def test_correctness_exact_bucket(self):
        """Test output matches eager mode for exact bucket sizes."""
        skip = self.skip_if_no_deps()
        if skip:
            print(f"SKIP: {skip}")
            return

        from transformers import AutoModelForCausalLM
        from mini_piecewise import cudagraph_compile_hf

        model = AutoModelForCausalLM.from_pretrained(
            "/vllm-workspace/Qwen3-0.6B-Base",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).cuda().eval()

        capture_sizes = [32, 64, 128]
        runner = cudagraph_compile_hf(model, capture_sizes)
        runner.capture()

        # Test exact bucket sizes - should have perfect accuracy
        for seq_len in capture_sizes:
            input_ids = torch.randint(
                0, model.config.vocab_size,
                (seq_len,),
                device="cuda",
                dtype=torch.long
            )
            input_ids_batched = input_ids.unsqueeze(0)

            # Eager output (reference)
            with torch.inference_mode():
                eager_out = model(input_ids_batched, use_cache=False).logits.squeeze(0)

            # Runner output
            runner_out = runner(input_ids)

            # Check shape
            assert runner_out.shape == eager_out.shape, \
                f"Shape mismatch: {runner_out.shape} vs {eager_out.shape}"

            # Strict correctness check for exact bucket sizes
            max_diff = (eager_out - runner_out).abs().max().item()
            assert max_diff < 1e-5, \
                f"seq_len={seq_len}: max_diff={max_diff} (should be ~0 for exact bucket)"

        del model, runner
        torch.cuda.empty_cache()

    def test_deterministic_output(self):
        """Test that runner produces deterministic output."""
        skip = self.skip_if_no_deps()
        if skip:
            print(f"SKIP: {skip}")
            return

        from transformers import AutoModelForCausalLM
        from mini_piecewise import cudagraph_compile_hf

        model = AutoModelForCausalLM.from_pretrained(
            "/vllm-workspace/Qwen3-0.6B-Base",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).cuda().eval()

        runner = cudagraph_compile_hf(model, [32, 64])
        runner.capture()

        # Same input
        input_ids = torch.randint(
            0, model.config.vocab_size,
            (64,),
            device="cuda",
            dtype=torch.long
        )

        # Run twice
        out1 = runner(input_ids.clone())
        out2 = runner(input_ids.clone())

        # Should be identical
        assert torch.equal(out1, out2), "Runner should produce deterministic output"

        del model, runner
        torch.cuda.empty_cache()

    def test_different_inputs_different_outputs(self):
        """Test that different inputs produce different outputs."""
        skip = self.skip_if_no_deps()
        if skip:
            print(f"SKIP: {skip}")
            return

        from transformers import AutoModelForCausalLM
        from mini_piecewise import cudagraph_compile_hf

        model = AutoModelForCausalLM.from_pretrained(
            "/vllm-workspace/Qwen3-0.6B-Base",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).cuda().eval()

        runner = cudagraph_compile_hf(model, [32, 64])
        runner.capture()

        # Different inputs
        input_ids1 = torch.randint(0, 1000, (64,), device="cuda", dtype=torch.long)
        input_ids2 = torch.randint(0, 1000, (64,), device="cuda", dtype=torch.long)

        # Ensure inputs are different
        while torch.equal(input_ids1, input_ids2):
            input_ids2 = torch.randint(0, 1000, (64,), device="cuda", dtype=torch.long)

        out1 = runner(input_ids1)
        out2 = runner(input_ids2)

        # Should be different
        assert not torch.equal(out1, out2), "Different inputs should produce different outputs"

        del model, runner
        torch.cuda.empty_cache()


def run_all_tests():
    """Run all tests without pytest."""
    print("=" * 60)
    print(" Running mini_piecewise tests")
    print("=" * 60)

    # Attention detector tests (no deps required)
    detector_tests = TestAttentionDetectors()
    tests = [
        ("test_qwen_attention_detector", detector_tests.test_qwen_attention_detector),
        ("test_llama_attention_detector", detector_tests.test_llama_attention_detector),
        ("test_auto_attention_detector", detector_tests.test_auto_attention_detector),
        ("test_piece_policy_integration", detector_tests.test_piece_policy_integration),
    ]

    # Qwen3 tests (require deps)
    qwen_tests = TestQwen3CudaGraph()
    tests.extend([
        ("test_attention_detection", qwen_tests.test_attention_detection),
        ("test_cudagraph_build_and_capture", qwen_tests.test_cudagraph_build_and_capture),
        ("test_correctness_exact_bucket", qwen_tests.test_correctness_exact_bucket),
        ("test_deterministic_output", qwen_tests.test_deterministic_output),
        ("test_different_inputs_different_outputs", qwen_tests.test_different_inputs_different_outputs),
    ])

    passed = 0
    failed = 0

    for name, test_fn in tests:
        print(f"\n--- {name} ---")
        try:
            test_fn()
            print("PASS")
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f" Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)