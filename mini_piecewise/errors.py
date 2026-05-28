class PiecewiseHybridError(RuntimeError):
    """Base exception for the mini_piecewise framework."""


class CudaNotAvailableError(PiecewiseHybridError):
    """Raised when CUDA is required but not available."""


class CaptureNotPerformedError(PiecewiseHybridError):
    """Raised when replay is attempted before capture."""


class ShapeOutOfRangeError(PiecewiseHybridError):
    """Raised when runtime shape is larger than max capture size."""


class RecaptureError(PiecewiseHybridError):
    """Raised when re-capture fails or is not supported."""


class FreeError(PiecewiseHybridError):
    """Raised when freeing captured resources fails."""
