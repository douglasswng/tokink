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
