class PiecewiseHybridError(RuntimeError):
    """Base error for sglang_min_piecewise."""


class CudaNotAvailableError(PiecewiseHybridError):
    """Raised when CUDA is required but not available."""


class CaptureNotPerformedError(PiecewiseHybridError):
    """Raised when replay is attempted before capture."""


class ShapeOutOfRangeError(PiecewiseHybridError):
    """Raised when runtime shape is larger than max capture size."""
