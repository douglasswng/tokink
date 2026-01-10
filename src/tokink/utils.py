import warnings


def warn(message: str, stacklevel: int = 2) -> None:
    """
    Issue a warning with consistent formatting.

    Args:
        message: The warning message to display
        stacklevel: Number of stack frames to go up (default: 2, which points to the caller)
    """
    warnings.warn(message, UserWarning, stacklevel=stacklevel)
