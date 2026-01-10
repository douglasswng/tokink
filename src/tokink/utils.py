import warnings


def warn(message: str, stacklevel: int = 2) -> None:
    """
    Issue a warning with consistent formatting.

    Args:
        message: The warning message to display
        stacklevel: Number of stack frames to go up (default: 2, which points to the caller)
    """
    warnings.warn(message, UserWarning, stacklevel=stacklevel)


def clean_round(x):
    """Round a number to the nearest integer using traditional rounding (0.5 rounds up)."""
    return int(x + 0.5) if x > 0 else int(x - 0.5)
