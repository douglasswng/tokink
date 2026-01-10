import numpy as np
import pytest

from tokink.ink import Ink, Point, Stroke
from tokink.processor import resample, scale, smooth, to_int


class TestScale:
    def test_scale_default_factor(self):
        """Test scaling with default factor (0.2)"""
        ink = Ink.from_coords([[(10, 20), (30, 40)]])
        scaled = scale(ink)
        assert scaled.strokes[0].points[0].x == 2.0
        assert scaled.strokes[0].points[0].y == 4.0
        assert scaled.strokes[0].points[1].x == 6.0
        assert scaled.strokes[0].points[1].y == 8.0

    def test_scale_custom_factor(self):
        """Test scaling with custom factor"""
        ink = Ink.from_coords([[(5, 10)]])
        scaled = scale(ink, factor=2)
        assert scaled.strokes[0].points[0].x == 10
        assert scaled.strokes[0].points[0].y == 20

    def test_scale_zero_factor(self):
        """Test scaling with zero factor"""
        ink = Ink.from_coords([[(5, 10), (15, 20)]])
        scaled = scale(ink, factor=0)
        assert scaled.strokes[0].points[0].x == 0
        assert scaled.strokes[0].points[0].y == 0
        assert scaled.strokes[0].points[1].x == 0
        assert scaled.strokes[0].points[1].y == 0

    def test_scale_negative_factor(self):
        """Test scaling with negative factor (flips coordinates)"""
        ink = Ink.from_coords([[(5, 10)]])
        scaled = scale(ink, factor=-1)
        assert scaled.strokes[0].points[0].x == -5
        assert scaled.strokes[0].points[0].y == -10

    def test_scale_multiple_strokes(self):
        """Test scaling ink with multiple strokes"""
        ink = Ink.from_coords([[(0, 0), (10, 10)], [(20, 20), (30, 30)]])
        scaled = scale(ink, factor=0.5)
        assert len(scaled.strokes) == 2
        assert scaled.strokes[0].points[0].x == 0
        assert scaled.strokes[0].points[1].x == 5.0
        assert scaled.strokes[1].points[0].x == 10.0
        assert scaled.strokes[1].points[1].x == 15.0

    def test_scale_empty_ink(self):
        """Test scaling empty ink"""
        ink = Ink(strokes=[])
        scaled = scale(ink, factor=2)
        assert len(scaled.strokes) == 0


class TestToInt:
    def test_to_int_float_coordinates(self):
        """Test converting float coordinates to integers"""
        ink = Ink.from_coords([[(1.4, 2.6), (3.2, 4.8)]])
        ink_int = to_int(ink)
        assert ink_int.strokes[0].points[0].x == 1
        assert ink_int.strokes[0].points[0].y == 3
        assert ink_int.strokes[0].points[1].x == 3
        assert ink_int.strokes[0].points[1].y == 5
        assert isinstance(ink_int.strokes[0].points[0].x, int)

    def test_to_int_already_int(self):
        """Test converting already integer coordinates"""
        ink = Ink[int](strokes=[Stroke[int](points=[Point[int](x=1, y=2)])])
        ink_int = to_int(ink)
        assert ink_int.strokes[0].points[0].x == 1
        assert ink_int.strokes[0].points[0].y == 2

    def test_to_int_rounding(self):
        """Test that to_int uses traditional rounding (0.5 rounds up)"""
        ink = Ink.from_coords([[(0.4, 0.6), (1.5, 2.5)]])
        ink_int = to_int(ink)
        assert ink_int.strokes[0].points[0].x == 0  # 0.4 rounds down
        assert ink_int.strokes[0].points[0].y == 1  # 0.6 rounds up
        assert ink_int.strokes[0].points[1].x == 2  # 1.5 rounds up (traditional)
        assert ink_int.strokes[0].points[1].y == 3  # 2.5 rounds up (traditional)

    def test_to_int_negative_coordinates(self):
        """Test converting negative float coordinates"""
        ink = Ink.from_coords([[(-1.4, -2.6), (-3.2, -4.8)]])
        ink_int = to_int(ink)
        assert ink_int.strokes[0].points[0].x == -1
        assert ink_int.strokes[0].points[0].y == -3
        assert ink_int.strokes[0].points[1].x == -3
        assert ink_int.strokes[0].points[1].y == -5

    def test_to_int_multiple_strokes(self):
        """Test converting multiple strokes to integers"""
        ink = Ink.from_coords([[(1.1, 2.2)], [(3.3, 4.4)]])
        ink_int = to_int(ink)
        assert len(ink_int.strokes) == 2
        assert all(
            isinstance(p.x, int) and isinstance(p.y, int)
            for stroke in ink_int.strokes
            for p in stroke.points
        )

    def test_to_int_empty_ink(self):
        """Test converting empty ink"""
        ink = Ink(strokes=[])
        ink_int = to_int(ink)
        assert len(ink_int.strokes) == 0


class TestResample:
    def test_resample_default(self):
        """Test resampling with default sample_every (3)"""
        points = [(i, i * 2) for i in range(10)]
        ink = Ink.from_coords([points])
        resampled = resample(ink)
        # Should keep points at indices 0, 3, 6, 9
        assert len(resampled.strokes[0].points) == 4
        assert resampled.strokes[0].points[0].x == 0
        assert resampled.strokes[0].points[1].x == 3
        assert resampled.strokes[0].points[2].x == 6
        assert resampled.strokes[0].points[3].x == 9

    def test_resample_custom_interval(self):
        """Test resampling with custom interval"""
        points = [(i, i) for i in range(10)]
        ink = Ink.from_coords([points])
        resampled = resample(ink, sample_every=2)
        # Should keep points at indices 0, 2, 4, 6, 8
        assert len(resampled.strokes[0].points) == 5
        assert resampled.strokes[0].points[0].x == 0
        assert resampled.strokes[0].points[1].x == 2
        assert resampled.strokes[0].points[4].x == 8

    def test_resample_every_point(self):
        """Test resampling with sample_every=1 (no change)"""
        points = [(i, i) for i in range(5)]
        ink = Ink.from_coords([points])
        resampled = resample(ink, sample_every=1)
        assert len(resampled.strokes[0].points) == 5

    def test_resample_large_interval(self):
        """Test resampling with interval larger than stroke length"""
        points = [(i, i) for i in range(5)]
        ink = Ink.from_coords([points])
        resampled = resample(ink, sample_every=10)
        # Should only keep first point
        assert len(resampled.strokes[0].points) == 1
        assert resampled.strokes[0].points[0].x == 0

    def test_resample_multiple_strokes(self):
        """Test resampling ink with multiple strokes"""
        ink = Ink.from_coords([[(i, i) for i in range(10)], [(i * 2, i * 2) for i in range(6)]])
        resampled = resample(ink, sample_every=2)
        assert len(resampled.strokes) == 2
        assert len(resampled.strokes[0].points) == 5  # 0, 2, 4, 6, 8
        assert len(resampled.strokes[1].points) == 3  # 0, 2, 4

    def test_resample_single_point_stroke(self):
        """Test resampling stroke with single point"""
        ink = Ink.from_coords([[(5, 5)]])
        resampled = resample(ink, sample_every=3)
        assert len(resampled.strokes[0].points) == 1
        assert resampled.strokes[0].points[0].x == 5

    def test_resample_empty_ink(self):
        """Test resampling empty ink"""
        ink = Ink(strokes=[])
        resampled = resample(ink, sample_every=3)
        assert len(resampled.strokes) == 0

    def test_resample_preserves_original(self):
        """Test that resampling doesn't modify original ink"""
        points = [(i, i) for i in range(10)]
        ink = Ink.from_coords([points])
        original_len = len(ink.strokes[0].points)
        resample(ink, sample_every=3)
        assert len(ink.strokes[0].points) == original_len  # Original unchanged


class TestSmooth:
    def test_smooth_default_parameters(self):
        """Test smoothing with default parameters"""
        # Create a stroke with some noise
        points = [(i, i + (i % 2)) for i in range(20)]
        ink = Ink.from_coords([points])
        smoothed = smooth(ink)

        # Should have same number of points
        assert len(smoothed.strokes[0].points) == 20

        # Smoothed values should be floats
        assert isinstance(smoothed.strokes[0].points[0].x, float)

    def test_smooth_custom_parameters(self):
        """Test smoothing with custom window length and polynomial order"""
        points = [(i, i + (i % 2)) for i in range(20)]
        ink = Ink.from_coords([points])
        smoothed = smooth(ink, window_length=5, polyorder=2)
        assert len(smoothed.strokes[0].points) == 20

    def test_smooth_short_stroke(self):
        """Test smoothing stroke shorter than window length"""
        points = [(i, i) for i in range(5)]
        ink = Ink.from_coords([points])
        smoothed = smooth(ink, window_length=7, polyorder=3)

        # Should return unchanged stroke
        assert len(smoothed.strokes[0].points) == 5
        assert smoothed.strokes[0].points[0].x == 0
        assert smoothed.strokes[0].points[4].x == 4

    def test_smooth_reduces_noise(self):
        """Test that smoothing actually reduces noise"""
        # Create a noisy signal
        np.random.seed(42)
        x_coords = list(range(20))
        y_coords = [i + np.random.normal(0, 0.5) for i in range(20)]
        points = list(zip(x_coords, y_coords))
        ink = Ink.from_coords([points])

        smoothed = smooth(ink, window_length=7, polyorder=3)

        # Calculate variance of differences (measure of noise)
        original_diffs = [
            abs(ink.strokes[0].points[i + 1].y - ink.strokes[0].points[i].y)
            for i in range(len(ink.strokes[0].points) - 1)
        ]
        smoothed_diffs = [
            abs(smoothed.strokes[0].points[i + 1].y - smoothed.strokes[0].points[i].y)
            for i in range(len(smoothed.strokes[0].points) - 1)
        ]

        # Smoothed should have lower variance (less noise)
        assert np.var(smoothed_diffs) < np.var(original_diffs)

    def test_smooth_multiple_strokes(self):
        """Test smoothing ink with multiple strokes"""
        ink = Ink.from_coords(
            [[(i, i + (i % 2)) for i in range(20)], [(i * 2, i * 2 + (i % 2)) for i in range(15)]]
        )
        smoothed = smooth(ink)

        assert len(smoothed.strokes) == 2
        assert len(smoothed.strokes[0].points) == 20
        assert len(smoothed.strokes[1].points) == 15

    def test_smooth_straight_line(self):
        """Test smoothing a straight line (should remain mostly unchanged)"""
        points = [(i, i) for i in range(20)]
        ink = Ink.from_coords([points])
        smoothed = smooth(ink)

        # Points should be very close to original
        for i in range(20):
            assert abs(smoothed.strokes[0].points[i].x - i) < 0.1
            assert abs(smoothed.strokes[0].points[i].y - i) < 0.1

    def test_smooth_empty_ink(self):
        """Test smoothing empty ink"""
        ink = Ink(strokes=[])
        smoothed = smooth(ink)
        assert len(smoothed.strokes) == 0

    def test_smooth_single_point_stroke(self):
        """Test smoothing stroke with single point"""
        ink = Ink.from_coords([[(5, 5)]])
        smoothed = smooth(ink, window_length=7, polyorder=3)
        assert len(smoothed.strokes[0].points) == 1
        assert smoothed.strokes[0].points[0].x == 5

    def test_smooth_preserves_original(self):
        """Test that smoothing doesn't modify original ink"""
        points = [(i, i + (i % 2)) for i in range(20)]
        ink = Ink.from_coords([points])
        original_y = ink.strokes[0].points[5].y
        smooth(ink)
        assert ink.strokes[0].points[5].y == original_y  # Original unchanged

    def test_smooth_polyorder_less_than_window(self):
        """Test that polyorder must be less than window length"""
        points = [(i, i) for i in range(20)]
        ink = Ink.from_coords([points])

        with pytest.raises(ValueError):
            smooth(ink, window_length=5, polyorder=5)

    def test_smooth_with_even_window_length(self):
        """Test smoothing with even window length (scipy handles this)"""
        points = [(i, i + (i % 2)) for i in range(20)]
        ink = Ink.from_coords([points])
        # Even window lengths are now handled by scipy
        smoothed = smooth(ink, window_length=8, polyorder=3)
        assert len(smoothed.strokes[0].points) == 20


class TestProcessorIntegration:
    def test_scale_then_to_int(self):
        """Test scaling followed by conversion to int"""
        ink = Ink.from_coords([[(10, 20), (30, 40)]])
        processed = to_int(scale(ink, factor=0.5))
        assert processed.strokes[0].points[0].x == 5
        assert processed.strokes[0].points[0].y == 10
        assert isinstance(processed.strokes[0].points[0].x, int)

    def test_resample_then_smooth(self):
        """Test resampling followed by smoothing"""
        points = [(i, i + (i % 2)) for i in range(30)]
        ink = Ink.from_coords([points])
        processed = smooth(resample(ink, sample_every=2))
        assert len(processed.strokes[0].points) == 15  # 30 points resampled every 2

    def test_full_pipeline(self):
        """Test full processing pipeline: scale -> to_int -> resample -> smooth"""
        ink = Ink.from_coords([[(i * 10, i * 10) for i in range(30)]])

        # Scale down
        processed = scale(ink, factor=0.1)
        # Convert to int
        processed = to_int(processed)
        # Resample
        processed = resample(processed, sample_every=2)
        # Smooth
        processed = smooth(processed)

        assert len(processed.strokes) == 1
        assert len(processed.strokes[0].points) == 15
