from pathlib import Path

import matplotlib
import pytest

from tokink.ink import Ink, Point, Stroke


class TestPoint:
    def test_from_coords_tuple(self):
        """Test Point creation from tuple"""
        p = Point.from_coords((1, 2))
        assert p.x == 1
        assert p.y == 2

    def test_from_coords_list(self):
        """Test Point creation from list"""
        p = Point.from_coords([3, 4])
        assert p.x == 3
        assert p.y == 4

    def test_from_coords_float(self):
        """Test Point with float coordinates"""
        p = Point.from_coords((1.5, 2.5))
        assert p.x == 1.5
        assert p.y == 2.5

    def test_point_addition(self):
        """Test Point addition operator"""
        p1 = Point(x=1, y=2)
        p2 = Point(x=3, y=4)
        p3 = p1 + p2
        assert p3.x == 4
        assert p3.y == 6

    def test_point_subtraction(self):
        """Test Point subtraction operator"""
        p1 = Point(x=5, y=8)
        p2 = Point(x=2, y=3)
        p3 = p1 - p2
        assert p3.x == 3
        assert p3.y == 5

    def test_invalid_coords(self):
        """Test that invalid coordinates raise ValueError"""
        with pytest.raises(ValueError, match="Invalid coordinates"):
            Point.from_coords([1, 2, 3])


class TestStroke:
    def test_from_coords_tuples(self):
        """Test Stroke creation from list of tuples"""
        s = Stroke.from_coords([(1, 2), (3, 4), (5, 6)])
        assert len(s.points) == 3
        assert s.points[0].x == 1
        assert s.points[0].y == 2
        assert s.points[2].x == 5

    def test_from_coords_lists(self):
        """Test Stroke creation from list of lists"""
        s = Stroke.from_coords([[0, 0], [10, 10]])
        assert len(s.points) == 2
        assert s.points[1].x == 10
        assert s.points[1].y == 10

    def test_single_point_stroke(self):
        """Test stroke with single point"""
        s = Stroke.from_coords([(5, 5)])
        assert len(s.points) == 1
        assert s.points[0].x == 5

    def test_stroke_str(self):
        """Test string representation of stroke"""
        s = Stroke.from_coords([(1, 2), (3, 4)])
        result = str(s)
        assert "(1, 2)" in result
        assert "(3, 4)" in result
        assert "→" in result


class TestInk:
    def test_from_coords_basic(self):
        """Test Ink creation from nested coordinate lists"""
        ink = Ink.from_coords([[(0, 0), (1, 1), (2, 2)], [(5, 5), (6, 6)]])
        assert len(ink.strokes) == 2
        assert len(ink.strokes[0].points) == 3
        assert len(ink.strokes[1].points) == 2

    def test_from_coords_mixed_lists_tuples(self):
        """Test Ink with mixed list and tuple coordinates"""
        ink = Ink.from_coords([[[0, 0], [1, 1]], [(2, 2), (3, 3)]])
        assert len(ink.strokes) == 2
        assert ink.strokes[0].points[0].x == 0
        assert ink.strokes[1].points[1].y == 3

    def test_example_loads(self):
        """Test that example JSON loads successfully"""
        ink = Ink.example()
        assert len(ink.strokes) > 0
        assert all(len(stroke.points) > 0 for stroke in ink.strokes)

    def test_ink_str(self):
        """Test string representation of ink"""
        ink = Ink.from_coords([[(1, 2), (3, 4)]])
        result = str(ink)
        assert "DigitalInk" in result
        assert "stroke1" in result

    def test_empty_ink(self):
        """Test ink with no strokes"""
        ink = Ink(strokes=[])
        assert len(ink.strokes) == 0

    def test_float_coordinates(self):
        """Test Ink with float coordinates"""
        ink = Ink.from_coords(
            [
                [(1.5, 2.5), (3.7, 4.2)],
            ]
        )
        assert ink.strokes[0].points[0].x == 1.5
        assert ink.strokes[0].points[1].y == 4.2

    def test_point_multiplication(self):
        """Test Point multiplication by scalar"""
        p = Point(x=2, y=3)
        p_scaled = p * 2.5
        assert p_scaled.x == 5.0
        assert p_scaled.y == 7.5

    def test_point_multiplication_int(self):
        """Test Point multiplication by integer"""
        p = Point(x=4.5, y=6.5)
        p_scaled = p * 2
        assert p_scaled.x == 9.0
        assert p_scaled.y == 13.0

    def test_point_to_int(self):
        """Test Point conversion to integer coordinates with traditional rounding"""
        p = Point(x=1.4, y=2.6)
        p_int = p.to_int()
        assert p_int.x == 1  # 1.4 rounds down
        assert p_int.y == 3  # 2.6 rounds up
        assert isinstance(p_int.x, int)
        assert isinstance(p_int.y, int)

    def test_point_to_int_half_values(self):
        """Test Point conversion with .5 values (traditional rounding, not banker's)"""
        # Traditional rounding: 0.5 always rounds up
        p1 = Point(x=0.5, y=1.5)
        p1_int = p1.to_int()
        assert p1_int.x == 1  # 0.5 → 1 (not 0 like banker's rounding)
        assert p1_int.y == 2  # 1.5 → 2

        p2 = Point(x=2.5, y=3.5)
        p2_int = p2.to_int()
        assert p2_int.x == 3  # 2.5 → 3 (not 2 like banker's rounding)
        assert p2_int.y == 4  # 3.5 → 4

    def test_point_to_int_negative(self):
        """Test Point conversion to int with negative values"""
        p = Point(x=-1.6, y=-2.4)
        p_int = p.to_int()
        assert p_int.x == -2
        assert p_int.y == -2

    def test_stroke_multiplication(self):
        """Test Stroke multiplication by scalar"""
        s = Stroke.from_coords([(1, 2), (3, 4)])
        s_scaled = s * 2
        assert s_scaled.points[0].x == 2
        assert s_scaled.points[0].y == 4
        assert s_scaled.points[1].x == 6
        assert s_scaled.points[1].y == 8

    def test_stroke_multiplication_float(self):
        """Test Stroke multiplication by float"""
        s = Stroke.from_coords([(2, 4), (6, 8)])
        s_scaled = s * 0.5
        assert s_scaled.points[0].x == 1.0
        assert s_scaled.points[0].y == 2.0
        assert s_scaled.points[1].x == 3.0
        assert s_scaled.points[1].y == 4.0

    def test_stroke_to_int(self):
        """Test Stroke conversion to integer coordinates"""
        s = Stroke.from_coords([(1.4, 2.6), (3.2, 4.8)])
        s_int = s.to_int()
        assert s_int.points[0].x == 1
        assert s_int.points[0].y == 3
        assert s_int.points[1].x == 3
        assert s_int.points[1].y == 5
        assert all(isinstance(p.x, int) and isinstance(p.y, int) for p in s_int.points)

    def test_ink_multiplication(self):
        """Test Ink multiplication by scalar"""
        ink = Ink.from_coords([[(1, 2), (3, 4)], [(5, 6)]])
        ink_scaled = ink * 3
        assert ink_scaled.strokes[0].points[0].x == 3
        assert ink_scaled.strokes[0].points[0].y == 6
        assert ink_scaled.strokes[0].points[1].x == 9
        assert ink_scaled.strokes[0].points[1].y == 12
        assert ink_scaled.strokes[1].points[0].x == 15
        assert ink_scaled.strokes[1].points[0].y == 18

    def test_ink_multiplication_float(self):
        """Test Ink multiplication by float"""
        ink = Ink.from_coords([[(10, 20)]])
        ink_scaled = ink * 0.1
        assert ink_scaled.strokes[0].points[0].x == 1.0
        assert ink_scaled.strokes[0].points[0].y == 2.0

    def test_ink_to_int(self):
        """Test Ink conversion to integer coordinates with traditional rounding"""
        ink = Ink.from_coords([[(1.4, 2.6), (3.2, 4.8)], [(5.5, 6.5)]])
        ink_int = ink.to_int()
        assert ink_int.strokes[0].points[0].x == 1  # 1.4 rounds down
        assert ink_int.strokes[0].points[0].y == 3  # 2.6 rounds up
        assert ink_int.strokes[0].points[1].x == 3  # 3.2 rounds down
        assert ink_int.strokes[0].points[1].y == 5  # 4.8 rounds up
        assert ink_int.strokes[1].points[0].x == 6  # 5.5 rounds up (traditional)
        assert ink_int.strokes[1].points[0].y == 7  # 6.5 rounds up (traditional)
        # Verify all coordinates are integers
        for stroke in ink_int.strokes:
            for point in stroke.points:
                assert isinstance(point.x, int)
                assert isinstance(point.y, int)

    def test_ink_len(self):
        """Test Ink length returns total number of points"""
        ink = Ink.from_coords([[(0, 0), (1, 1), (2, 2)], [(5, 5), (6, 6)]])
        assert len(ink) == 5

    def test_ink_len_empty(self):
        """Test Ink length with no strokes"""
        ink = Ink(strokes=[])
        assert len(ink) == 0

    def test_ink_save_plot_with_path(self, tmp_path):
        """Test saving ink plot to a specified path"""
        matplotlib.use("Agg")  # Use non-interactive backend

        ink = Ink.from_coords([[(0, 0), (10, 10)]])
        save_path = tmp_path / "test_ink.png"
        ink.save_plot(save_path)
        assert save_path.exists()

    def test_ink_save_plot_default_path(self):
        """Test saving ink with auto-generated filename"""
        matplotlib.use("Agg")  # Use non-interactive backend

        ink = Ink.from_coords([[(0, 0), (10, 10)]])
        ink.save_plot()

        # Check that a file was created in the current working directory.
        cwd = Path.cwd()
        png_files = list(cwd.glob("*.png"))
        assert len(png_files) > 0

        # Clean up the generated file.
        for png_file in png_files:
            png_file.unlink()

    def test_ink_save_and_load(self, tmp_path):
        """Test saving and loading ink as JSON"""
        ink = Ink.from_coords([[(0, 0), (1, 1)]])
        save_path = tmp_path / "test_ink.json"

        # Test save.
        ink.save(save_path)
        assert save_path.exists()

        # Test load.
        loaded_ink = Ink.load(save_path)
        assert len(loaded_ink.strokes) == 1
        assert len(loaded_ink.strokes[0].points) == 2
        assert loaded_ink.strokes[0].points[1].x == 1
