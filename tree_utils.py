from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch


def _is_tensor(x: Any) -> bool:
    return isinstance(x, torch.Tensor)


def tree_map(fn: Callable[[Any], Any], tree: Any) -> Any:
    """Apply fn to each leaf in a nested structure."""
    if isinstance(tree, dict):
        return {k: tree_map(fn, v) for k, v in tree.items()}
    if isinstance(tree, list):
        return [tree_map(fn, v) for v in tree]
    if isinstance(tree, tuple):
        return tuple(tree_map(fn, v) for v in tree)
    return fn(tree)


def tree_flatten(tree: Any) -> list[Any]:
    """Flatten leaves in a deterministic order."""
    leaves: list[Any] = []

    def _walk(x: Any) -> None:
        if isinstance(x, dict):
            for k in x.keys():
                _walk(x[k])
        elif isinstance(x, (list, tuple)):
            for v in x:
                _walk(v)
        else:
            leaves.append(x)

    _walk(tree)
    return leaves


def tree_tensor_addresses(tree: Any) -> list[int]:
    addrs: list[int] = []
    for leaf in tree_flatten(tree):
        if _is_tensor(leaf):
            addrs.append(int(leaf.data_ptr()))
    return addrs


def tree_make_static_like(
    sample: Any,
    *,
    static_size: int,
    runtime_size: int,
) -> Any:
    """Create a static-buffer tree from a sample input tree.

    Heuristic:
    - If a tensor has dim0 == runtime_size, we expand dim0 to static_size.
    - Otherwise, keep shape unchanged.

    This matches common token-major layouts: [T, ...].
    """

    def _mk(x: Any) -> Any:
        if not _is_tensor(x):
            return x
        if x.ndim >= 1 and int(x.shape[0]) == int(runtime_size):
            shape = (static_size,) + tuple(x.shape[1:])
            return torch.empty(shape, device=x.device, dtype=x.dtype)
        return torch.empty_like(x)

    return tree_map(_mk, sample)


def tree_copy_into(
    dst: Any,
    src: Any,
    *,
    runtime_size: int,
    static_size: int,
    zero_pad: bool,
) -> None:
    """Copy src tensors into dst tensors in-place."""

    if _is_tensor(dst) and _is_tensor(src):
        if dst.ndim >= 1 and int(dst.shape[0]) == int(static_size) and src.ndim >= 1:
            # Copy valid prefix.
            if runtime_size > 0:
                dst[:runtime_size].copy_(src[:runtime_size])
            # Zero-pad tail to avoid stale values.
            if zero_pad and runtime_size < static_size:
                dst[runtime_size:static_size].zero_()
        else:
            dst.copy_(src)
        return

    if isinstance(dst, dict) and isinstance(src, dict):
        if dst.keys() != src.keys():
            raise ValueError("Mismatched dict keys between dst and src")
        for k in dst.keys():
            tree_copy_into(
                dst[k],
                src[k],
                runtime_size=runtime_size,
                static_size=static_size,
                zero_pad=zero_pad,
            )
        return

    if isinstance(dst, list) and isinstance(src, list):
        if len(dst) != len(src):
            raise ValueError("Mismatched list length between dst and src")
        for a, b in zip(dst, src):
            tree_copy_into(
                a,
                b,
                runtime_size=runtime_size,
                static_size=static_size,
                zero_pad=zero_pad,
            )
        return

    if isinstance(dst, tuple) and isinstance(src, tuple):
        if len(dst) != len(src):
            raise ValueError("Mismatched tuple length between dst and src")
        for a, b in zip(dst, src):
            tree_copy_into(
                a,
                b,
                runtime_size=runtime_size,
                static_size=static_size,
                zero_pad=zero_pad,
            )
        return

    # Non-tensor leaves ignored.
    return


def tree_slice_dim0(tree: Any, *, runtime_size: int, static_size: int) -> Any:
    """Slice tensor leaves on dim0 when they match static_size."""

    def _slice(x: Any) -> Any:
        if _is_tensor(x) and x.ndim >= 1 and int(x.shape[0]) == int(static_size):
            return x[:runtime_size]
        return x

    return tree_map(_slice, tree)
