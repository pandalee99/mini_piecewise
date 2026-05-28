"""Diagnostic and inspection tools for captured models.

Provides utilities to inspect model structure, measure memory usage,
generate timing reports, and configure framework logging.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import torch

from .config import PiecePolicy

logger = logging.getLogger("mini_piecewise")


def _get_logger() -> logging.Logger:
    """Get the mini_piecewise logger.

    Configure with:
        import logging
        logging.getLogger("mini_piecewise").setLevel(logging.DEBUG)
    """
    return logger


def setup_logging(level: int = logging.INFO) -> None:
    """Configure mini_piecewise logging.

    Args:
        level: Logging level (default: INFO)
    """
    mp_logger = logging.getLogger("mini_piecewise")
    mp_logger.setLevel(level)
    if not mp_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "[%(name)s] %(levelname)s: %(message)s"
        ))
        mp_logger.addHandler(handler)


class ModelInspector:
    """Inspect captured model structure and statistics."""

    @staticmethod
    def piece_summary(model: Any) -> dict[str, Any]:
        """Return summary of pieces: names, policies, backends, sizes.

        Works with both PiecewiseHybridModel and CudaGraphRunner.
        """
        if hasattr(model, "summary"):
            return model.summary()

        # Fallback for raw models
        return {
            "model_type": type(model).__name__,
            "note": "No capture information available (model not wrapped)",
        }

    @staticmethod
    def memory_summary(model: Any) -> dict[str, Any]:
        """Return memory usage stats for a captured model.

        Returns GPU memory allocated and reserved, plus estimated
        capture overhead.
        """
        result = {
            "gpu_allocated_mb": torch.cuda.memory_allocated() / (1024 * 1024),
            "gpu_reserved_mb": torch.cuda.memory_reserved() / (1024 * 1024),
        }

        if hasattr(model, "summary"):
            summary = model.summary()
            result["capture_info"] = summary

            # Estimate memory from backends
            total_backend_mem = 0
            if "pieces" in summary:
                for piece in summary["pieces"]:
                    if "backend" in piece:
                        mem = piece["backend"].get("memory_estimate_bytes", 0)
                        total_backend_mem += mem

            result["backend_memory_estimate_mb"] = total_backend_mem / (1024 * 1024)

        return result

    @staticmethod
    def timing_report(
        model: Any,
        *args: Any,
        eager_model: Any = None,
        num_iters: int = 20,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Measure timing for a captured model vs eager baseline.

        Args:
            model: Captured (hybrid) model
            args: Input arguments
            eager_model: Eager model for comparison (optional)
            num_iters: Number of measurement iterations
            kwargs: Input keyword arguments

        Returns:
            Dict with timing information
        """
        result = {}

        # Warmup
        for _ in range(5):
            _ = model(*args, **kwargs)
        torch.cuda.synchronize()

        # Measure captured model
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_iters):
            _ = model(*args, **kwargs)
        torch.cuda.synchronize()
        captured_time = (time.perf_counter() - start) / num_iters * 1000

        result["captured_ms"] = captured_time

        if eager_model is not None:
            # Warmup eager
            for _ in range(5):
                with torch.inference_mode():
                    _ = eager_model(*args, **kwargs)
            torch.cuda.synchronize()

            # Measure eager
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(num_iters):
                with torch.inference_mode():
                    _ = eager_model(*args, **kwargs)
            torch.cuda.synchronize()
            eager_time = (time.perf_counter() - start) / num_iters * 1000

            result["eager_ms"] = eager_time
            result["speedup"] = eager_time / captured_time if captured_time > 0 else 0

        return result

    @staticmethod
    def format_summary(summary: dict[str, Any]) -> str:
        """Format a summary dict as a human-readable string."""
        lines = []
        lines.append(f"Model type: {summary.get('model_type', 'unknown')}")
        lines.append(f"Installed: {summary.get('installed', False)}")
        lines.append(f"Capture sizes: {summary.get('capture_sizes', [])}")

        if "pieces" in summary:
            lines.append(f"Number of pieces: {summary.get('num_pieces', 0)}")
            for piece in summary["pieces"]:
                policy = piece.get("policy", "unknown")
                name = piece.get("name", "unknown")
                backend_info = piece.get("backend", {})
                if backend_info:
                    backend_name = backend_info.get("backend", "unknown")
                    sizes = backend_info.get("capture_sizes", [])
                    lines.append(f"  {name}: {policy} ({backend_name}, sizes={sizes})")
                else:
                    lines.append(f"  {name}: {policy}")

        if "captured" in summary:
            lines.append(f"Captured: {summary.get('captured', False)}")
            lines.append(f"Num entries: {summary.get('num_entries', 0)}")

        return "\n".join(lines)