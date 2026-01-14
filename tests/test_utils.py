import re
from datetime import datetime

import matplotlib.pyplot as plt
import pytest
from matplotlib.figure import Figure

from tokink.utils import create_axes, get_timestamp, math_round, warn


class TestMathRound:
    def test_round_positive_up(self):
        """Test rounding up for positive numbers"""
        assert math_round(1.6) == 2
        assert math_round(2.9) == 3

    def test_round_positive_down(self):
        """Test rounding down for positive numbers"""
        assert math_round(1.4) == 1
        assert math_round(2.1) == 2

    def test_round_positive_half(self):
        """Test rounding .5 for positive numbers (should round up)"""
        # (should round up).
        assert math_round(0.5) == 1
        assert math_round(1.5) == 2
        assert math_round(2.5) == 3

    def test_round_negative_away_from_zero(self):
        """Test rounding for negative numbers"""
        assert math_round(-1.6) == -2
        assert math_round(-1.4) == -1

    def test_round_negative_half(self):
        """Test rounding .5 for negative numbers (should round away from zero)"""
        # (should round away from zero).
        assert math_round(-0.5) == -1
        assert math_round(-1.5) == -2
        assert math_round(-2.5) == -3

    def test_round_zero(self):
        """Test rounding zero"""
        assert math_round(0) == 0
        assert math_round(0.0) == 0


class TestGetTimestamp:
    def test_timestamp_format(self):
        """Test that timestamp matches YYYYMMDD_HHMMSS format"""
        timestamp = get_timestamp()
        assert isinstance(timestamp, str)
        # Regex for YYYYMMDD_HHMMSS.
        assert re.match(r"^\d{8}_\d{6}$", timestamp)

    def test_timestamp_current(self):
        """Test that timestamp reflects current time (at least the year/month/day)"""
        timestamp = get_timestamp()
        now = datetime.now()
        # (at least the year/month/day).
        expected_prefix = now.strftime("%Y%m%d")
        assert timestamp.startswith(expected_prefix)


class TestWarn:
    def test_warn_issues_warning(self):
        """Test that warn() issues a UserWarning"""
        with pytest.warns(UserWarning, match="test warning message"):
            warn("test warning message")

    def test_warn_stacklevel(self):
        """Test that warn() accepts stacklevel (internal check that it doesn't crash)"""
        # (internal check that it doesn't crash).
        with pytest.warns(UserWarning):
            warn("test message", stacklevel=3)


class TestCreateAxes:
    def test_creates_axes(self):
        """Test that create_ink_axes returns a matplotlib Axes object"""
        ax = create_axes()
        assert ax is not None
        assert hasattr(ax, "plot")
        plt.close()

    def test_default_figsize(self):
        """Test that default figsize is (12, 8)"""
        ax = create_axes()
        fig = ax.get_figure()
        assert isinstance(fig, Figure)

        assert fig.get_figwidth() == 12
        assert fig.get_figheight() == 8
        plt.close()

    def test_custom_figsize(self):
        """Test that custom figsize is applied"""
        ax = create_axes(figsize=(10, 6))
        fig = ax.get_figure()
        assert isinstance(fig, Figure)

        assert fig.get_figwidth() == 10
        assert fig.get_figheight() == 6
        plt.close()

    def test_aspect_ratio_equal(self):
        """Test that aspect ratio is set to equal"""
        ax = create_axes()
        # When aspect is "equal", get_aspect() returns 1.0
        assert ax.get_aspect() == 1.0
        plt.close()

    def test_yaxis_inverted(self):
        """Test that y-axis is inverted"""
        ax = create_axes()
        # Check if y-axis is inverted by comparing limits
        ylim = ax.get_ylim()
        assert ylim[0] > ylim[1]  # Top should be greater than bottom when inverted
        plt.close()
