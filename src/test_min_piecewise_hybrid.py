import pytest
import torch

from sglang_min_piecewise import PiecewiseHybridConfig, make_piecewise_hybrid_model
from sglang_min_piecewise.cudagraph_backend import CUDAGraphPiece


class ToyAttention(torch.nn.Module):
    """A tiny attention-like module we want to keep eager."""

    def __init__(self, hidden: int):
        super().__init__()
        self.q = torch.nn.Linear(hidden, hidden, bias=False)
        self.k = torch.nn.Linear(hidden, hidden, bias=False)
        self.v = torch.nn.Linear(hidden, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [T, H]
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        attn = torch.softmax(q @ k.transpose(0, 1) / (q.shape[-1] ** 0.5), dim=-1)
        return attn @ v


class ToyModel(torch.nn.Module):
    def __init__(self, vocab: int = 64, hidden: int = 32):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, hidden)
        self.mlp1 = torch.nn.Linear(hidden, hidden, bias=False)
        self.attn = ToyAttention(hidden)
        self.mlp2 = torch.nn.Linear(hidden, hidden, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        # ids: [T]
        x = self.emb(ids)  # [T, H]
        x = torch.relu(self.mlp1(x))
        x = self.attn(x)  # keep eager
        x = self.mlp2(x)
        return x


@pytest.mark.optional
def test_fx_split_and_hybrid_capture_matches_eager():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    torch.manual_seed(0)

    device = torch.device("cuda")
    model = ToyModel().to(device).eval()

    config = PiecewiseHybridConfig.from_sizes([8, 16], warmup_iters=2)

    def example_inputs_fn(static_size: int):
        # ids: [T]
        ids = torch.zeros((static_size,), device=device, dtype=torch.long)
        return (ids,)

    hybrid = make_piecewise_hybrid_model(model, config, example_inputs_fn=example_inputs_fn)

    # Capture all non-attention pieces.
    hybrid.capture()

    # Ensure attention was actually isolated.
    assert any(it.is_attention_piece for it in hybrid.items)

    # Verify that at least one piece is replaced by CUDAGraphPiece.
    split_gm = hybrid.split_gm
    assert any(isinstance(split_gm.get_submodule(it.submod_name), CUDAGraphPiece) for it in hybrid.items if not it.is_attention_piece)

    # Attention piece must remain eager.
    for it in hybrid.items:
        if it.is_attention_piece:
            assert not isinstance(split_gm.get_submodule(it.submod_name), CUDAGraphPiece)

    ids_rt = torch.randint(0, 64, (7,), device=device, dtype=torch.long)
    y_ref = model(ids_rt)
    y_h = hybrid(ids_rt)

    assert y_h.shape == y_ref.shape
    assert torch.allclose(y_h, y_ref, rtol=1e-4, atol=1e-4)
