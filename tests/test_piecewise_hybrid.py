"""Tests for piecewise CUDA graph optimization.

Validates FX-based piecewise capture with simple traceable models,
PiecePolicy-based piece selection, backend abstraction, lifecycle
management, backward compatibility, diagnostics, and CudaGraphRunner.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mini_piecewise import (
    PiecewiseHybridConfig,
    PiecePolicy,
    PieceSelector,
    attention_piece_selector,
    default_runtime_size_fn,
    auto_attention_detector,
    CaptureBackend,
    CUDAGraphPiece,
    cudagraph_backend_factory,
    make_piecewise_hybrid_model,
    PiecewiseHybridModel,
    CudaGraphRunner,
    cudagraph_compile_hf,
    get_attention_modules,
    ModelInspector,
    setup_logging,
)
from mini_piecewise.fx_split import SplitItem
from mini_piecewise.errors import (
    PiecewiseHybridError,
    CudaNotAvailableError,
    CaptureNotPerformedError,
    ShapeOutOfRangeError,
    RecaptureError,
    FreeError,
)


# --- Test fixtures ---


class SimpleAttention(nn.Module):
    """Attention module for testing piece isolation."""

    def __init__(self, hidden: int):
        super().__init__()
        self.q = nn.Linear(hidden, hidden, bias=False)
        self.k = nn.Linear(hidden, hidden, bias=False)
        self.v = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.q(x), self.k(x), self.v(x)
        scale = x.shape[-1] ** -0.5
        attn = torch.softmax(q @ k.transpose(0, 1) * scale, dim=-1)
        return attn @ v


class TwoLayerModel(nn.Module):
    """Two-layer model with attention in the first layer for split testing."""

    def __init__(self, vocab: int = 64, hidden: int = 32):
        super().__init__()
        self.emb = nn.Embedding(vocab, hidden)
        self.mlp1 = nn.Linear(hidden, hidden, bias=False)
        self.attn = SimpleAttention(hidden)
        self.mlp2 = nn.Linear(hidden, hidden, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.emb(ids)
        x = torch.relu(self.mlp1(x))
        x = self.attn(x)
        x = self.mlp2(x)
        return x


class NoAttentionModel(nn.Module):
    """Model without any attention modules (all pieces should be CAPTURE)."""

    def __init__(self, vocab: int = 64, hidden: int = 32):
        super().__init__()
        self.emb = nn.Embedding(vocab, hidden)
        self.mlp = nn.Linear(hidden, hidden, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.emb(ids))


# --- PiecePolicy and config tests ---


def test_piece_policy_enum():
    assert PiecePolicy.CAPTURE.value == "capture"
    assert PiecePolicy.EAGER.value == "eager"
    assert PiecePolicy.SKIP.value == "skip"
    assert len(PiecePolicy) == 3


def test_attention_piece_selector_returns_eager_for_attention():
    cls = type("Qwen3Attention", (nn.Module,), {})
    result = attention_piece_selector(cls(), "model.layers.0.self_attn")
    assert result == PiecePolicy.EAGER


def test_attention_piece_selector_returns_capture_for_mlp():
    cls = type("Qwen3MLP", (nn.Module,), {})
    result = attention_piece_selector(cls(), "model.layers.0.mlp")
    assert result == PiecePolicy.CAPTURE


def test_attention_piece_selector_returns_eager_for_multihead_attention():
    mha = nn.MultiheadAttention(32, 4)
    result = attention_piece_selector(mha, "encoder.attn")
    assert result == PiecePolicy.EAGER


def test_config_validation_empty_sizes():
    with pytest.raises(ValueError, match="non-empty"):
        PiecewiseHybridConfig(capture_sizes=())


def test_config_validation_negative_sizes():
    with pytest.raises(ValueError, match="positive"):
        PiecewiseHybridConfig(capture_sizes=(8, -1))


def test_config_validation_unsorted_sizes():
    with pytest.raises(ValueError, match="sorted"):
        PiecewiseHybridConfig(capture_sizes=(16, 8))


def test_config_validation_duplicate_sizes():
    with pytest.raises(ValueError, match="duplicates"):
        PiecewiseHybridConfig(capture_sizes=(8, 8, 16))


def test_config_validation_negative_warmup():
    with pytest.raises(ValueError, match="warmup_iters"):
        PiecewiseHybridConfig(capture_sizes=(8, 16), warmup_iters=-1)


def test_config_from_sizes():
    config = PiecewiseHybridConfig.from_sizes([32, 16, 64])
    assert config.capture_sizes == (16, 32, 64)


def test_config_from_sizes_dedup_and_sort():
    config = PiecewiseHybridConfig.from_sizes([32, 16, 32, 64])
    assert config.capture_sizes == (16, 32, 64)


def test_config_default_piece_selector():
    config = PiecewiseHybridConfig.from_sizes([8, 16])
    assert config.piece_selector is attention_piece_selector
    assert config.is_attention_module is None


def test_config_effective_piece_selector_default():
    config = PiecewiseHybridConfig.from_sizes([8, 16])
    selector = config._effective_piece_selector()
    cls = type("Qwen3Attention", (nn.Module,), {})
    assert selector(cls(), "attn") == PiecePolicy.EAGER


def test_config_effective_piece_selector_with_is_attention_override():
    from mini_piecewise.config import default_is_attention_module
    config = PiecewiseHybridConfig.from_sizes(
        [8, 16],
        is_attention_module=default_is_attention_module,
    )
    selector = config._effective_piece_selector()
    cls = type("Qwen3Attention", (nn.Module,), {})
    assert selector(cls(), "attn") == PiecePolicy.EAGER

    # MLP should be CAPTURE
    cls2 = type("SomeMLP", (nn.Module,), {})
    assert selector(cls2(), "mlp") == PiecePolicy.CAPTURE


def test_runtime_size_fn_from_first_tensor():
    t = torch.zeros(32)
    assert default_runtime_size_fn((t,), {}) == 32


def test_runtime_size_fn_from_kwargs():
    t = torch.zeros(16)
    assert default_runtime_size_fn((), {"x": t}) == 16


def test_runtime_size_fn_raises_on_0dim():
    t = torch.tensor(5.0)
    with pytest.raises(ValueError, match="0-dim"):
        default_runtime_size_fn((t,), {})


def test_runtime_size_fn_raises_on_no_tensors():
    with pytest.raises(ValueError, match="no tensor"):
        default_runtime_size_fn((42,), {})


# --- SplitItem backward compat ---


def test_split_item_is_attention_piece():
    item_eager = SplitItem("submod_0", 0, PiecePolicy.EAGER)
    assert item_eager.is_attention_piece is True

    item_capture = SplitItem("submod_1", 1, PiecePolicy.CAPTURE)
    assert item_capture.is_attention_piece is False

    item_skip = SplitItem("submod_2", 2, PiecePolicy.SKIP)
    assert item_skip.is_attention_piece is False


# --- CaptureBackend protocol ---


def test_cudagraph_piece_satisfies_protocol():
    assert issubclass(CUDAGraphPiece, CaptureBackend)


def test_cudagraph_backend_factory_creates_piece():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    device = torch.device("cuda")
    model = TwoLayerModel().to(device).eval()
    config = PiecewiseHybridConfig.from_sizes([8, 16])

    backend = cudagraph_backend_factory(model, config, device=device)
    assert isinstance(backend, CUDAGraphPiece)
    assert backend.capture_sizes == config.capture_sizes


# --- PiecewiseHybridModel core tests ---


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestPiecewiseHybridModel:
    """Tests for PiecewiseHybridModel capture, correctness, and lifecycle."""

    def setup_method(self):
        self.device = torch.device("cuda")
        torch.manual_seed(42)
        self.model = TwoLayerModel().to(self.device).eval()
        self.config = PiecewiseHybridConfig.from_sizes([8, 16, 32], warmup_iters=2)
        self.example_inputs_fn = lambda s: (torch.zeros((s,), device=self.device, dtype=torch.long),)

    def test_split_structure(self):
        hybrid = make_piecewise_hybrid_model(
            self.model, self.config, example_inputs_fn=self.example_inputs_fn
        )
        items = hybrid.items
        assert len(items) >= 3
        assert any(it.policy == PiecePolicy.EAGER for it in items)
        assert any(it.policy == PiecePolicy.CAPTURE for it in items)

    def test_correctness_across_sizes(self):
        hybrid = make_piecewise_hybrid_model(
            self.model, self.config, example_inputs_fn=self.example_inputs_fn
        )
        hybrid.capture()

        for seq_len in [5, 8, 12, 16, 25, 32]:
            ids = torch.randint(0, 64, (seq_len,), device=self.device, dtype=torch.long)
            with torch.inference_mode():
                y_ref = self.model(ids)
            y_h = hybrid(ids)

            assert y_h.shape == y_ref.shape
            assert torch.allclose(y_h, y_ref, rtol=1e-4, atol=1e-4)

    def test_backward_compat_is_attention_module(self):
        from mini_piecewise.config import default_is_attention_module
        config = PiecewiseHybridConfig.from_sizes(
            [8, 16],
            warmup_iters=2,
            is_attention_module=default_is_attention_module,
        )
        hybrid = make_piecewise_hybrid_model(
            self.model, config, example_inputs_fn=self.example_inputs_fn
        )
        hybrid.capture()

        ids = torch.randint(0, 64, (7,), device=self.device, dtype=torch.long)
        with torch.inference_mode():
            y_ref = self.model(ids)
        y_h = hybrid(ids)
        assert torch.allclose(y_h, y_ref, rtol=1e-4, atol=1e-4)

    def test_summary(self):
        hybrid = make_piecewise_hybrid_model(
            self.model, self.config, example_inputs_fn=self.example_inputs_fn
        )
        hybrid.capture()

        summary = hybrid.summary()
        assert summary["installed"] is True
        assert summary["num_pieces"] > 0
        assert summary["capture_sizes"] == [8, 16, 32]
        assert len(summary["pieces"]) > 0
        for piece in summary["pieces"]:
            assert "name" in piece
            assert "policy" in piece

    def test_free_and_recapture(self):
        hybrid = make_piecewise_hybrid_model(
            self.model, self.config, example_inputs_fn=self.example_inputs_fn
        )
        hybrid.capture()

        # Free
        hybrid.free()
        assert hybrid._installed is False
        assert len(hybrid._backends) == 0

        # Recapture with same sizes
        hybrid.recapture()
        assert hybrid._installed is True

        # Verify still correct
        ids = torch.randint(0, 64, (8,), device=self.device, dtype=torch.long)
        with torch.inference_mode():
            y_ref = self.model(ids)
        y_h = hybrid(ids)
        assert torch.allclose(y_h, y_ref, rtol=1e-4, atol=1e-4)

    def test_recapture_with_new_sizes(self):
        hybrid = make_piecewise_hybrid_model(
            self.model, self.config, example_inputs_fn=self.example_inputs_fn
        )
        hybrid.capture()

        # Recapture with expanded sizes
        hybrid.recapture([8, 16, 32, 64])
        summary = hybrid.summary()
        assert summary["capture_sizes"] == [8, 16, 32, 64]

    def test_eager_execution_before_capture(self):
        hybrid = make_piecewise_hybrid_model(
            self.model, self.config, example_inputs_fn=self.example_inputs_fn
        )
        # Should work without capture (eager fallback)
        ids = torch.randint(0, 64, (8,), device=self.device, dtype=torch.long)
        y_h = hybrid(ids)
        with torch.inference_mode():
            y_ref = self.model(ids)
        assert torch.allclose(y_h, y_ref, rtol=1e-4, atol=1e-4)

    def test_no_attention_model(self):
        """All pieces should be CAPTURE when model has no attention."""
        model = NoAttentionModel().to(self.device).eval()
        config = PiecewiseHybridConfig.from_sizes([8, 16])
        hybrid = make_piecewise_hybrid_model(
            model, config, example_inputs_fn=self.example_inputs_fn
        )

        assert all(it.policy == PiecePolicy.CAPTURE for it in hybrid.items)

        hybrid.capture()
        ids = torch.randint(0, 64, (7,), device=self.device, dtype=torch.long)
        with torch.inference_mode():
            y_ref = model(ids)
        y_h = hybrid(ids)
        assert torch.allclose(y_h, y_ref, rtol=1e-4, atol=1e-4)


# --- CudaGraphRunner tests ---


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestCudaGraphRunner:
    """Tests for general CudaGraphRunner with adapter support."""

    def setup_method(self):
        self.device = torch.device("cuda")
        torch.manual_seed(42)

    def test_identity_adapter_passthrough(self):
        """CudaGraphRunner with default (identity) adapters should work."""
        model = nn.Sequential(
            nn.Linear(32, 64, bias=False),
            nn.ReLU(),
            nn.Linear(64, 32, bias=False),
        ).to(self.device).eval()

        runner = CudaGraphRunner(
            model, [16, 32],
            example_inputs_fn=lambda s: (torch.randn(s, 32, device=self.device),),
        )
        runner.capture()

        x = torch.randn(16, 32, device=self.device)
        with torch.inference_mode():
            y_ref = model(x)
        y_h = runner(x)
        assert y_h.shape == y_ref.shape
        assert torch.allclose(y_h, y_ref, rtol=1e-4, atol=1e-4)

    def test_custom_input_adapter(self):
        """Custom input adapter that adds batch dimension."""
        model = nn.Sequential(
            nn.Linear(32, 32, bias=False),
        ).to(self.device).eval()

        def input_adapter(args, kwargs):
            x = args[0]
            if x.dim() == 2 and x.shape[0] == 1:
                return (x,), kwargs
            return args, kwargs

        runner = CudaGraphRunner(
            model, [16, 32],
            input_adapter=input_adapter,
            example_inputs_fn=lambda s: (torch.randn(1, s, 32, device=self.device),),
        )
        runner.capture()

        x = torch.randn(1, 16, 32, device=self.device)
        with torch.inference_mode():
            y_ref = model(x)
        y_h = runner(x)
        assert y_h.shape == y_ref.shape
        assert torch.allclose(y_h, y_ref, rtol=1e-4, atol=1e-4)

    def test_summary(self):
        model = nn.Linear(32, 32).to(self.device).eval()
        runner = CudaGraphRunner(
            model, [8, 16],
            example_inputs_fn=lambda s: (torch.randn(s, 32, device=self.device),),
        )
        runner.capture()

        summary = runner.summary()
        assert summary["captured"] is True
        assert summary["capture_sizes"] == [8, 16]
        assert summary["num_entries"] == 2

    def test_free_and_recapture(self):
        model = nn.Linear(32, 32).to(self.device).eval()
        runner = CudaGraphRunner(
            model, [8, 16],
            example_inputs_fn=lambda s: (torch.randn(s, 32, device=self.device),),
        )
        runner.capture()

        runner.free()
        assert runner._captured is False
        assert len(runner._entries) == 0

        runner.recapture([8, 16, 32])
        assert runner._captured is True
        assert runner.capture_sizes == (8, 16, 32)

    def test_shape_out_of_range(self):
        model = nn.Linear(32, 32).to(self.device).eval()
        runner = CudaGraphRunner(
            model, [16],
            example_inputs_fn=lambda s: (torch.randn(s, 32, device=self.device),),
        )
        runner.capture()

        # Request a size larger than max capture size
        with pytest.raises(ValueError, match="exceeds"):
            x = torch.randn(32, 32, device=self.device)
            runner(x)

    def test_eager_fallback_before_capture(self):
        model = nn.Linear(32, 32).to(self.device).eval()
        runner = CudaGraphRunner(
            model, [16],
            example_inputs_fn=lambda s: (torch.randn(s, 32, device=self.device),),
        )

        # Should work in eager mode before capture
        x = torch.randn(16, 32, device=self.device)
        with torch.inference_mode():
            y_ref = model(x)
        y_h = runner(x)
        assert torch.allclose(y_h, y_ref, rtol=1e-4, atol=1e-4)


# --- Error hierarchy tests ---


def test_error_hierarchy():
    assert issubclass(CudaNotAvailableError, PiecewiseHybridError)
    assert issubclass(CaptureNotPerformedError, PiecewiseHybridError)
    assert issubclass(ShapeOutOfRangeError, PiecewiseHybridError)
    assert issubclass(RecaptureError, PiecewiseHybridError)
    assert issubclass(FreeError, PiecewiseHybridError)
    assert issubclass(PiecewiseHybridError, RuntimeError)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_capture_not_performed_error():
    """Forward on CUDAGraphPiece before capture raises error."""
    fn = nn.Linear(32, 32).to("cuda").eval()
    piece = CUDAGraphPiece(
        fn,
        capture_sizes=(8, 16),
        warmup_iters=2,
        zero_pad_inputs=True,
        runtime_size_fn=default_runtime_size_fn,
    )
    with pytest.raises(CaptureNotPerformedError):
        piece(torch.randn(8, 32, device="cuda"))


# --- Attention detector tests ---


def test_qwen_attention_detector():
    from mini_piecewise.config import qwen_attention_detector

    cls = type("Qwen3Attention", (nn.Module,), {})
    assert qwen_attention_detector(cls(), "attn")

    cls2 = type("Qwen3MLP", (nn.Module,), {})
    assert not qwen_attention_detector(cls2(), "mlp")


def test_llama_attention_detector():
    from mini_piecewise.config import llama_attention_detector

    cls = type("LlamaAttention", (nn.Module,), {})
    assert llama_attention_detector(cls(), "attn")

    cls2 = type("LlamaMLP", (nn.Module,), {})
    assert not llama_attention_detector(cls2(), "mlp")


def test_auto_attention_detector_specific_families():
    from mini_piecewise.config import auto_attention_detector

    for cls_name in ["Qwen3Attention", "LlamaAttention", "MistralAttention"]:
        cls = type(cls_name, (nn.Module,), {})
        assert auto_attention_detector(cls(), "attn")

    for cls_name in ["SomeMLP", "RMSNorm", "Linear"]:
        cls = type(cls_name, (nn.Module,), {})
        assert not auto_attention_detector(cls(), "block")

    # AttentionMask should be excluded by heuristic
    cls_mask = type("AttentionMask", (nn.Module,), {})
    assert not auto_attention_detector(cls_mask(), "mask")

    # MultiheadAttention built-in
    mha = nn.MultiheadAttention(32, 4)
    assert auto_attention_detector(mha, "encoder.attn")


# --- Diagnostics tests ---


def test_model_inspector_format_summary():
    inspector = ModelInspector()
    summary = {
        "model_type": "TwoLayerModel",
        "installed": True,
        "capture_sizes": [8, 16],
        "num_pieces": 3,
        "pieces": [
            {"name": "submod_0", "policy": "capture"},
            {"name": "submod_1", "policy": "eager"},
            {"name": "submod_2", "policy": "capture"},
        ],
    }
    formatted = inspector.format_summary(summary)
    assert "TwoLayerModel" in formatted
    assert "capture" in formatted
    assert "eager" in formatted


def test_setup_logging():
    import logging
    setup_logging(logging.DEBUG)
    mp_logger = logging.getLogger("mini_piecewise")
    assert mp_logger.level == logging.DEBUG


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_memory_summary():
    model = TwoLayerModel().to("cuda").eval()
    config = PiecewiseHybridConfig.from_sizes([8])
    hybrid = make_piecewise_hybrid_model(
        model, config,
        example_inputs_fn=lambda s: (torch.zeros((s,), device="cuda", dtype=torch.long),),
    )
    hybrid.capture()

    inspector = ModelInspector()
    mem = inspector.memory_summary(hybrid)
    assert "gpu_allocated_mb" in mem
    assert mem["gpu_allocated_mb"] > 0


# --- Export verification ---


def test_exports_no_piecewise_compile_hf():
    """piecewise_compile_hf should not be in public exports."""
    import mini_piecewise
    assert "piecewise_compile_hf" not in mini_piecewise.__all__


def test_exports_no_hf_causal_lm_wrapper():
    """HFCausalLMWrapper should not be in public exports."""
    import mini_piecewise
    assert "HFCausalLMWrapper" not in mini_piecewise.__all__


def test_exports_include_new_api():
    """All new API items should be in exports."""
    import mini_piecewise
    for name in [
        "PiecewiseHybridConfig", "PiecePolicy", "PieceSelector",
        "attention_piece_selector", "CaptureBackend", "CUDAGraphPiece",
        "cudagraph_backend_factory", "CudaGraphRunner", "cudagraph_compile_hf",
        "ModelInspector", "setup_logging",
    ]:
        assert name in mini_piecewise.__all__, f"{name} missing from __all__"


def test_internal_classes_available():
    """Internal classes should still be importable from submodules."""
    from mini_piecewise.hf_wrapper import HFCausalLMWrapper, HFCudaGraphRunner
    assert HFCudaGraphRunner is CudaGraphRunner