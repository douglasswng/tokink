import numpy as np
from scipy.signal import savgol_filter

from tokink.ink import Ink, Point, Stroke

__all__ = ["scale", "to_int", "resample", "smooth"]


def scale(ink: Ink, factor: float | int = 0.2) -> Ink:
    """
    Scale the coordinates of all points in the ink drawing.

    Args:
        ink: The ink drawing to scale.
        factor: The scaling factor to apply.

    Returns:
        A new Ink instance with scaled coordinates.
    """
    return ink * factor


def to_int(ink: Ink) -> Ink[int]:
    """
    Convert all point coordinates in the ink drawing to integers.

    Args:
        ink: The ink drawing to convert.

    Returns:
        A new Ink instance with integer coordinates.
    """
    return ink.to_int()


def resample(ink: Ink, sample_every: int = 3) -> Ink:
    """
    Resample the ink strokes by taking every Nth point.

    Args:
        ink: The ink drawing to resample.
        sample_every: The interval at which to sample points.

    Returns:
        A new Ink instance with resampled strokes.
    """

    def resample_stroke(stroke: Stroke, sample_every: int) -> Stroke:
        return stroke.model_copy(update={"points": stroke.points[::sample_every]})

    return Ink(strokes=[resample_stroke(stroke, sample_every) for stroke in ink.strokes])


def smooth(ink: Ink, window_length: int = 7, polyorder: int = 3) -> Ink:
    """
    Smooth the ink strokes using a Savitzky-Golay filter.

    Args:
        ink: The ink drawing to smooth.
        window_length: The length of the filter window (must be odd).
        polyorder: The order of the polynomial used to fit the samples.

    Returns:
        A new Ink instance with smoothed strokes.
    """

    def smooth_stroke(stroke: Stroke, window_length: int, polyorder: int) -> "Stroke":
        if len(stroke.points) < window_length:
            return Stroke(points=stroke.points[:])

        x_coords = np.array([point.x for point in stroke.points])
        y_coords = np.array([point.y for point in stroke.points])

        smoothed_x = savgol_filter(x_coords, window_length, polyorder)
        smoothed_y = savgol_filter(y_coords, window_length, polyorder)

        assert isinstance(smoothed_x, np.ndarray)
        assert isinstance(smoothed_y, np.ndarray)

        smoothed_points = [Point(x=float(x), y=float(y)) for x, y in zip(smoothed_x, smoothed_y)]
        return Stroke(points=smoothed_points)

    return Ink(strokes=[smooth_stroke(stroke, window_length, polyorder) for stroke in ink.strokes])
