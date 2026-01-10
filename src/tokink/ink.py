from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Self, Type

import matplotlib.pyplot as plt
from pydantic import BaseModel

from tokink.utils import clean_round

type Coord[T: (int, float)] = tuple[T, T] | list[T]


class Point[T: (int, float)](BaseModel):
    x: T
    y: T

    @classmethod
    def from_coords[U: (int, float)](cls: Type, coords: Coord[U]) -> "Point[U]":
        match coords:
            case (x, y):
                return cls(x=x, y=y)
            case [x, y]:
                return cls(x=x, y=y)
            case _:
                raise ValueError(f"Invalid coordinates: {coords}")

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"

    def __add__(self, other: "Point[T]") -> "Point[T]":
        return Point(x=self.x + other.x, y=self.y + other.y)

    def __sub__(self, other: "Point[T]") -> "Point[T]":
        return Point(x=self.x - other.x, y=self.y - other.y)

    def __mul__(self, other: float | int) -> "Point":
        return Point(x=self.x * other, y=self.y * other)

    def to_int(self) -> "Point[int]":
        return Point(x=clean_round(self.x), y=clean_round(self.y))


class Stroke[T: (int, float)](BaseModel):
    points: list[Point[T]]

    @classmethod
    def from_coords[U: (int, float)](cls: Type, coords: Sequence[Coord[U]]) -> "Stroke[U]":
        points = [Point.from_coords(coord) for coord in coords]
        return cls(points=points)

    def __str__(self) -> str:
        points_str = " → ".join(str(point) for point in self.points)
        return points_str

    def __len__(self) -> int:
        return len(self.points)

    def __mul__(self, other: float | int) -> "Stroke":
        return Stroke(points=[point * other for point in self.points])

    def to_int(self) -> "Stroke[int]":
        return Stroke(points=[point.to_int() for point in self.points])


class Ink[T: (int, float)](BaseModel):
    strokes: list[Stroke[T]]

    @classmethod
    def from_coords[U: (int, float)](
        cls: Type,
        coords: Sequence[Sequence[Coord[U]]],
    ) -> "Ink[U]":
        strokes = [Stroke.from_coords(stroke_coords) for stroke_coords in coords]
        return cls(strokes=strokes)

    @classmethod
    def example(cls) -> Self:
        example_path = Path(__file__).parent / "data" / "ink_example.json"
        with open(example_path, "r") as f:
            return cls.model_validate_json(f.read())

    def __str__(self) -> str:
        line = "-" * 100
        strokes_str = f"{line}\nDigitalInk:\n"
        strokes_str += "\n\n".join(
            f"  stroke{i + 1}: {str(stroke)}" for i, stroke in enumerate(self.strokes)
        )
        strokes_str += f"\n{line}"
        return strokes_str

    def __len__(self) -> int:
        return sum(len(stroke) for stroke in self.strokes)

    def __mul__(self, other: float | int) -> "Ink":
        return Ink(strokes=[stroke * other for stroke in self.strokes])

    def to_int(self) -> "Ink[int]":
        return Ink(strokes=[stroke.to_int() for stroke in self.strokes])

    def _create_plot(self) -> None:
        ax = plt.subplots(figsize=(12, 8))[1]
        ax.set_aspect("equal", adjustable="box")
        ax.invert_yaxis()

        for stroke in self.strokes:
            x, y = zip(*((p.x, p.y) for p in stroke.points))
            if len(stroke.points) == 1:
                ax.plot(x, y, "ko", markersize=1.5)
            else:
                ax.plot(x, y, "-k", linewidth=1.5)

    def plot(self) -> None:
        """Display the ink strokes in a matplotlib window."""
        self._create_plot()
        plt.show()

    def save(self, save_path: Path | str | None = None) -> None:
        """Save the ink strokes as an image file.

        Args:
            save_path: Path where the image should be saved. If None, generates a timestamped
                      filename in the package directory.
        """
        self._create_plot()

        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = Path(__file__).parent / f"{timestamp}.png"
        else:
            save_path = Path(save_path)

        plt.savefig(save_path)
        plt.close()
