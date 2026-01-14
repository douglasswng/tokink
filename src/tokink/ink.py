from collections.abc import Sequence
from pathlib import Path
from typing import Self, Type

import matplotlib.pyplot as plt
from pydantic import BaseModel

from tokink.utils import get_timestamp, math_round

__all__ = ["Ink", "Stroke", "Point"]

type Coords[T: (int, float)] = tuple[T, T] | list[T]


class Point[T: (int, float)](BaseModel):
    """
    A representation of a single point in 2D space.

    Attributes:
        x: The x-coordinate.
        y: The y-coordinate.
    """

    x: T
    y: T

    @classmethod
    def from_coords[U: (int, float)](cls: Type, coords: Coords[U]) -> "Point[U]":
        """
        Create a Point from a tuple or list of coordinates.

        Args:
            coords: A sequence (tuple or list) of two numbers [x, y] or (x, y).

        Returns:
            A Point instance.

        Raises:
            ValueError: If the input sequence does not have exactly two elements.
        """
        match coords:
            case (x, y):
                return cls(x=x, y=y)
            case [x, y]:
                return cls(x=x, y=y)
            case _:
                raise ValueError(f"Invalid coordinates: {coords}")

    def __str__(self) -> str:
        """Return a string representation of the point: '(x, y)'."""
        return f"({self.x}, {self.y})"

    def __add__(self, other: "Point[T]") -> "Point[T]":
        """Add another point to this point."""
        return Point(x=self.x + other.x, y=self.y + other.y)

    def __sub__(self, other: "Point[T]") -> "Point[T]":
        """Subtract another point from this point."""
        return Point(x=self.x - other.x, y=self.y - other.y)

    def __mul__(self, other: float | int) -> "Point":
        """Scale the point coordinates by a factor."""
        return Point(x=self.x * other, y=self.y * other)

    def to_int(self) -> "Point[int]":
        """Convert point coordinates to integers using traditional rounding."""
        return Point(x=math_round(self.x), y=math_round(self.y))


class Stroke[T: (int, float)](BaseModel):
    """
    A representation of a single ink stroke consisting of multiple points.

    Attributes:
        points: A list of Point instances representing the stroke path.
    """

    points: list[Point[T]]

    @classmethod
    def from_coords[U: (int, float)](cls: Type, coords: Sequence[Coords[U]]) -> "Stroke[U]":
        """
        Create a Stroke from a sequence of coordinate pairs.

        Args:
            coords: A sequence of (x, y) coordinate pairs.

        Returns:
            A Stroke instance.
        """
        points = [Point.from_coords(coord) for coord in coords]
        return cls(points=points)

    def __str__(self) -> str:
        """Return a string representation of the stroke showing point flow."""
        points_str = " → ".join(str(point) for point in self.points)
        return points_str

    def __len__(self) -> int:
        """Return the number of points in the stroke."""
        return len(self.points)

    def __mul__(self, other: float | int) -> "Stroke":
        """Scale all points in the stroke by a factor."""
        return Stroke(points=[point * other for point in self.points])

    def to_int(self) -> "Stroke[int]":
        """Convert all points in the stroke to integer coordinates."""
        return Stroke(points=[point.to_int() for point in self.points])


class Ink[T: (int, float)](BaseModel):
    """
    A representation of a complete ink drawing consisting of multiple strokes.

    Attributes:
        strokes: A list of Stroke instances.
    """

    strokes: list[Stroke[T]]

    @classmethod
    def from_coords[U: (int, float)](
        cls: Type,
        coords: Sequence[Sequence[Coords[U]]],
    ) -> "Ink[U]":
        """
        Create an Ink object from a nested sequence of coordinates.

        Args:
            coords: A nested sequence where each inner sequence represents a stroke
                   and contains (x, y) coordinate pairs.

        Returns:
            An Ink instance.
        """
        strokes = [Stroke.from_coords(stroke_coords) for stroke_coords in coords]
        return cls(strokes=strokes)

    @classmethod
    def load(cls, path: Path | str) -> Self:
        """
        Load ink strokes from a JSON file.

        Args:
            path: Path to the JSON file to load.

        Returns:
            An Ink instance.
        """
        with open(path, "r") as f:
            return cls.model_validate_json(f.read())

    @classmethod
    def example(cls) -> Self:
        """
        Load an example ink drawing from the package data.

        Returns:
            An example Ink instance.
        """
        example_path = Path(__file__).parent / "data" / "ink_example.json"
        with open(example_path, "r") as f:
            return cls.model_validate_json(f.read())

    def __str__(self) -> str:
        """Return a formatted string representation of the Ink object."""
        line = "-" * 100
        strokes_str = f"{line}\nDigitalInk:\n"
        strokes_str += "\n\n".join(
            f"  stroke{i + 1}: {str(stroke)}" for i, stroke in enumerate(self.strokes)
        )
        strokes_str += f"\n{line}"
        return strokes_str

    def __len__(self) -> int:
        """Return the total number of points across all strokes."""
        return sum(len(stroke) for stroke in self.strokes)

    def __mul__(self, other: float | int) -> "Ink":
        """Scale all strokes in the ink drawing by a factor."""
        return Ink(strokes=[stroke * other for stroke in self.strokes])

    def to_int(self) -> "Ink[int]":
        """Convert all points in all strokes to integer coordinates."""
        return Ink(strokes=[stroke.to_int() for stroke in self.strokes])

    def save(self, save_path: Path | str | None = None) -> None:
        """Save the ink strokes as a JSON file.

        Args:
            save_path: Path where the JSON should be saved. If None, generates a timestamped
                      filename in the current working directory.
        """
        save_path = Path(save_path or Path.cwd() / f"{get_timestamp()}.json")
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "w") as f:
            f.write(self.model_dump_json(indent=2))

    def save_plot(self, save_path: Path | str | None = None) -> None:
        """Save the ink strokes as an image file.

        Args:
            save_path: Path where the image should be saved. If None, generates a timestamped
                      filename in the current working directory.
        """
        self._create_plot()

        save_path = Path(save_path or Path.cwd() / f"{get_timestamp()}.png")
        save_path.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(save_path)
        plt.close()

    def plot(self) -> None:
        """Display the ink strokes in a matplotlib window."""
        self._create_plot()
        plt.show()

    def _create_plot(self) -> None:
        """Internal helper to set up a matplotlib plot of the ink strokes."""
        ax = plt.subplots(figsize=(12, 8))[1]
        ax.set_aspect("equal", adjustable="box")
        ax.invert_yaxis()

        for stroke in self.strokes:
            x, y = zip(*((p.x, p.y) for p in stroke.points))
            if len(stroke.points) == 1:
                ax.plot(x, y, "ko", markersize=1.5)
            else:
                ax.plot(x, y, "-k", linewidth=1.5)
