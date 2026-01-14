import math

import numpy as np
from scipy.signal import savgol_filter

from tokink.ink import Ink, Point, Stroke

__all__ = ["scale", "to_int", "resample", "smooth", "jitter", "rotate", "shear"]


def scale(ink: Ink, factor: float | int = 1 / 16) -> Ink:
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


def resample(ink: Ink, sample_every: int = 2) -> Ink:
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
            return stroke.model_copy()

        x_coords = np.array([point.x for point in stroke.points])
        y_coords = np.array([point.y for point in stroke.points])

        smoothed_x = savgol_filter(x_coords, window_length, polyorder)
        smoothed_y = savgol_filter(y_coords, window_length, polyorder)

        assert isinstance(smoothed_x, np.ndarray)
        assert isinstance(smoothed_y, np.ndarray)

        smoothed_points = [Point(x=float(x), y=float(y)) for x, y in zip(smoothed_x, smoothed_y)]
        return Stroke(points=smoothed_points)

    return Ink(strokes=[smooth_stroke(stroke, window_length, polyorder) for stroke in ink.strokes])


def jitter(ink: Ink, sigma: float = 0.5) -> Ink:
    """
    Apply random Gaussian noise to each point in the ink drawing.

    Args:
        ink: The ink drawing to jitter.
        sigma: The standard deviation of the Gaussian noise.

    Returns:
        A new Ink instance with jittered points.
    """

    def jitter_stroke(stroke: Stroke, sigma: float) -> Stroke:
        points = []
        for point in stroke.points:
            jitter_x = np.random.normal(0, sigma)
            jitter_y = np.random.normal(0, sigma)
            points.append(Point(x=point.x + jitter_x, y=point.y + jitter_y))
        return Stroke(points=points)

    return Ink(strokes=[jitter_stroke(stroke, sigma) for stroke in ink.strokes])


def rotate(ink: Ink, angle_degrees: float) -> Ink:
    """
    Rotate the ink drawing by a given angle in degrees around the origin (0, 0).

    Args:
        ink: The ink drawing to rotate.
        angle_degrees: The rotation angle in degrees.

    Returns:
        A new Ink instance with rotated points.
    """
    angle_rad = math.radians(angle_degrees)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    def rotate_stroke(stroke: Stroke) -> Stroke:
        points = []
        for point in stroke.points:
            new_x = point.x * cos_a - point.y * sin_a
            new_y = point.x * sin_a + point.y * cos_a
            points.append(Point(x=new_x, y=new_y))
        return Stroke(points=points)

    return Ink(strokes=[rotate_stroke(stroke) for stroke in ink.strokes])


def shear(ink: Ink, shear_factor: float) -> Ink:
    """
    Apply a horizontal shear transformation to the ink drawing.

    Args:
        ink: The ink drawing to shear.
        shear_factor: The factor by which to shear the x-coordinates based on y.

    Returns:
        A new Ink instance with sheared points.
    """

    def shear_stroke(stroke: Stroke, factor: float) -> Stroke:
        points = [Point(x=p.x + factor * p.y, y=p.y) for p in stroke.points]
        return Stroke(points=points)

    return Ink(strokes=[shear_stroke(stroke, shear_factor) for stroke in ink.strokes])
