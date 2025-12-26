from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.fx as fx


@dataclass(frozen=True)
class SplitItem:
    """Metadata about a split subgraph."""

    submod_name: str
    graph_id: int
    is_attention_piece: bool


def split_graph_by_attention(
    gm: fx.GraphModule,
    *,
    is_attention_module: Callable[[torch.nn.Module, str], bool],
) -> tuple[fx.GraphModule, list[SplitItem]]:
    """Split an FX graph module into pieces, isolating attention calls.

    Policy (minimal, SGLang-like):
    - Traverse nodes in order.
    - When encountering an attention *call_module* node, put it in its own piece.
    - Other contiguous nodes are grouped.

    Returns:
        split_gm: a stitched GraphModule with submodules named submod_0, submod_1, ...
        items: metadata for each submodule piece.
    """

    # Identify which nodes are attention calls.
    attention_nodes: set[fx.Node] = set()
    for node in gm.graph.nodes:
        if node.op == "call_module":
            qualname = str(node.target)
            try:
                mod = gm.get_submodule(qualname)
            except AttributeError:
                continue
            if is_attention_module(mod, qualname):
                attention_nodes.add(node)

        # Fallback for traces that decompose attention into call_function.
        if node.op == "call_function":
            tgt = str(node.target)
            if "scaled_dot_product_attention" in tgt:
                attention_nodes.add(node)

    # Assign a piece id to each node.
    node_to_piece: dict[fx.Node, int] = {}
    attention_piece_ids: set[int] = set()

    piece_id = 0
    for node in gm.graph.nodes:
        if node.op in ("placeholder", "output"):
            continue

        if node in attention_nodes:
            # Isolate attention: bump id, put node alone, bump id for subsequent nodes.
            piece_id += 1
            node_to_piece[node] = piece_id
            attention_piece_ids.add(piece_id)
            piece_id += 1
        else:
            node_to_piece[node] = piece_id

    split_gm = fx.passes.split_module.split_module(
        gm,
        None,
        lambda n: node_to_piece[n],
        keep_original_order=True,
    )

    # Build SplitItem list (one per submodule).
    items: list[SplitItem] = []
    # NOTE: split_gm.named_modules() includes nested names like "submod_0.emb".
    # We only want the top-level pieces: "submod_<int>".
    for name, _m in split_gm.named_children():
        if not name.startswith("submod_"):
            continue
        suffix = name[len("submod_") :]
        if not suffix.isdigit():
            # Be conservative: ignore unexpected names.
            continue
        idx = int(suffix)
        items.append(
            SplitItem(
                submod_name=name,
                graph_id=idx,
                is_attention_piece=idx in attention_piece_ids,
            )
        )

    # Keep in submod order.
    items.sort(key=lambda x: x.graph_id)
    return split_gm, items
