import warnings
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.axes import Axes


def warn(message: str, stacklevel: int = 2) -> None:
    """
    Issue a warning with consistent formatting.

    Args:
        message: The warning message to display
        stacklevel: Number of stack frames to go up (default: 2, which points to the caller)
    """
    warnings.warn(message, UserWarning, stacklevel=stacklevel)


def create_axes(figsize: tuple[int, int] = (12, 8)) -> Axes:
    """
    Create a matplotlib axes configured for displaying ink strokes.

    Args:
        figsize: The figure size as (width, height) in inches.

    Returns:
        A configured matplotlib Axes object.
    """
    ax = plt.subplots(figsize=figsize)[1]
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    return ax


def math_round(x: float) -> int:
    """
    Round a number to the nearest integer using traditional rounding (0.5 rounds up).

    Args:
        x: The number to round.

    Returns:
        The rounded integer.
    """
    return int(x + 0.5) if x > 0 else int(x - 0.5)


def get_timestamp() -> str:
    """
    Generate a timestamp string in the format YYYYMMDD_HHMMSS.

    Returns:
        A timestamp string.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")
