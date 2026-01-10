from pathlib import Path
from typing import Self

import matplotlib.pyplot as plt
from pydantic import BaseModel


class Point[T: (int, float)](BaseModel):
    x: T
    y: T

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"

    def __add__(self, other: "Point[T]") -> "Point[T]":
        return Point(x=self.x + other.x, y=self.y + other.y)

    def __sub__(self, other: "Point[T]") -> "Point[T]":
        return Point(x=self.x - other.x, y=self.y - other.y)


class Stroke[T: (int, float)](BaseModel):
    points: list[Point[T]]

    def __str__(self) -> str:
        points_str = " → ".join(str(point) for point in self.points)
        return points_str


class Ink[T: (int, float)](BaseModel):
    strokes: list[Stroke[T]]

    def __str__(self) -> str:
        line = "-" * 100
        strokes_str = f"{line}\nDigitalInk:\n"
        strokes_str += "\n\n".join(
            f"  stroke{i + 1}: {str(stroke)}" for i, stroke in enumerate(self.strokes)
        )
        strokes_str += f"\n{line}"
        return strokes_str

    @classmethod
    def example(cls) -> Self:
        example_path = Path(__file__).parent / "data" / "ink_example.json"
        with open(example_path, "r") as f:
            return cls.model_validate_json(f.read())

    @classmethod
    def load_test(cls) -> "Ink[int]":
        raw_strokes = [[(0, 0), (1, 0)], [(2, 1), (4, -1)], [(10, 10)]]
        strokes: list[Stroke[int]] = [
            Stroke[int](points=[Point[int](x=x, y=y) for x, y in stroke]) for stroke in raw_strokes
        ]
        return Ink[int](strokes=strokes)

    def plot(self) -> None:
        ax = plt.subplots(figsize=(12, 8))[1]
        ax.set_aspect("equal", adjustable="box")
        ax.invert_yaxis()

        for stroke in self.strokes:
            x, y = zip(*((p.x, p.y) for p in stroke.points))
            if len(stroke.points) == 1:
                ax.plot(x, y, "ko", markersize=1.5)
            else:
                ax.plot(x, y, "-k", linewidth=1.5)

        plt.show()
