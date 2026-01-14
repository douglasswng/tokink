import numpy as np
from scipy.signal import savgol_filter

from tokink.ink import Ink, Point, Stroke


def scale(ink: Ink, factor: float | int = 0.2) -> Ink:
    return ink * factor


def to_int(ink: Ink) -> Ink[int]:
    return ink.to_int()


def resample(ink: Ink, sample_every: int = 3) -> Ink:
    def resample_stroke(stroke: Stroke, sample_every: int) -> Stroke:
        return stroke.model_copy(update={"points": stroke.points[::sample_every]})

    return Ink(strokes=[resample_stroke(stroke, sample_every) for stroke in ink.strokes])


def smooth(ink: Ink, window_length: int = 7, polyorder: int = 3) -> Ink:
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
