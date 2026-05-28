from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.fx as fx

from .config import PiecePolicy, PieceSelector


@dataclass(frozen=True)
class SplitItem:
    """Metadata about a split subgraph."""

    submod_name: str
    graph_id: int
    policy: PiecePolicy

    @property
    def is_attention_piece(self) -> bool:
        """Backward compatibility: True if this piece should run eagerly."""
        return self.policy == PiecePolicy.EAGER


def split_graph_by_attention(
    gm: fx.GraphModule,
    *,
    piece_selector: PieceSelector,
) -> tuple[fx.GraphModule, list[SplitItem]]:
    """Split an FX GraphModule into pieces according to piece_selector policy.

    EAGER pieces are isolated into separate submodules so they remain
    in eager mode during execution. CAPTURE pieces are grouped together
    for backend optimization. SKIP pieces are isolated but left unchanged.

    Returns:
        split_gm: a stitched GraphModule with submodules named submod_0, submod_1, ...
        items: metadata for each submodule piece.
    """
    # Classify each call_module node using the piece selector.
    node_policy: dict[fx.Node, PiecePolicy] = {}
    for node in gm.graph.nodes:
        if node.op == "call_module":
            qualname = str(node.target)
            try:
                mod = gm.get_submodule(qualname)
            except AttributeError:
                continue
            node_policy[node] = piece_selector(mod, qualname)

        # Fallback for traces that decompose attention into call_function.
        if node.op == "call_function":
            tgt = str(node.target)
            if "scaled_dot_product_attention" in tgt:
                node_policy[node] = PiecePolicy.EAGER

    # Assign a piece id to each node.
    node_to_piece: dict[fx.Node, int] = {}
    eager_piece_ids: set[int] = set()
    skip_piece_ids: set[int] = set()

    piece_id = 0
    for node in gm.graph.nodes:
        if node.op in ("placeholder", "output"):
            continue

        policy = node_policy.get(node, PiecePolicy.CAPTURE)

        if policy == PiecePolicy.SKIP:
            # SKIP nodes get their own piece so they can be removed later.
            piece_id += 1
            node_to_piece[node] = piece_id
            skip_piece_ids.add(piece_id)
            piece_id += 1
        elif policy == PiecePolicy.EAGER:
            # Isolate eager: bump id, put node alone, bump id for subsequent nodes.
            piece_id += 1
            node_to_piece[node] = piece_id
            eager_piece_ids.add(piece_id)
            piece_id += 1
        else:
            # CAPTURE nodes are grouped together.
            node_to_piece[node] = piece_id

    split_gm = fx.passes.split_module.split_module(
        gm,
        None,
        lambda n: node_to_piece[n],
        keep_original_order=True,
    )

    # Build SplitItem list (one per submodule).
    items: list[SplitItem] = []
    for name, _m in split_gm.named_children():
        if not name.startswith("submod_"):
            continue
        suffix = name[len("submod_") :]
        if not suffix.isdigit():
            continue
        idx = int(suffix)

        if idx in eager_piece_ids:
            policy = PiecePolicy.EAGER
        elif idx in skip_piece_ids:
            policy = PiecePolicy.SKIP
        else:
            policy = PiecePolicy.CAPTURE

        items.append(
            SplitItem(
                submod_name=name,
                graph_id=idx,
                policy=policy,
            )
        )

    # Keep in submod order.
    items.sort(key=lambda x: x.graph_id)

    # For backward compatibility, add is_attention_piece property.
    # This is done by making it accessible via the dataclass.
    return split_gm, items