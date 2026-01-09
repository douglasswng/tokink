import json
import warnings
from itertools import groupby
from pathlib import Path
from typing import Self

from tokenizers.models import BPE

from tokink.ink import Ink, Point, Stroke


class Tokinkizer:
    BOS = "[BOS]"
    EOS = "[EOS]"
    UP = "[UP]"
    DOWN = "[DOWN]"

    COORD_TO_TOKEN = {
        (0, 1): "[↑]",
        (0, -1): "[↓]",
        (-1, 0): "[←]",
        (1, 0): "[→]",
        (-1, 1): "[↖]",
        (1, 1): "[↗]",
        (-1, -1): "[↙]",
        (1, -1): "[↘]",
    }
    TOKEN_TO_COORD = {v: k for k, v in COORD_TO_TOKEN.items()}

    def __init__(self, vocab: dict[str, int], merges: list[tuple[str, str]]):
        self._vocab = vocab
        self._merges = merges

        self._bpe = self._init_bpe(vocab, merges)

    @classmethod
    def from_pretrained(cls, path: Path | str | None = None) -> Self:
        if path is None:
            path = Path(__file__).parent / "data"

        vocab_path = Path(path) / "vocab.json"
        merges_path = Path(path) / "merges.txt"

        with open(vocab_path, "r") as f:
            vocab = json.load(f)
        with open(merges_path, "r") as f:
            merges = [
                (parts[0], parts[1])
                for line in f
                if len(parts := line.strip().split()) == 2
            ]
        return cls(vocab=vocab, merges=merges)

    def _init_bpe(self, vocab: dict[str, int], merges: list[tuple[str, str]]) -> BPE:
        hf_vocab = {
            self._strip_token(token): id
            for token, id in vocab.items()
            if token not in [self.BOS, self.EOS, self.UP, self.DOWN]
        }
        hf_merges = [
            (self._strip_token(merge[0]), self._strip_token(merge[1]))
            for merge in merges
        ]
        return BPE(vocab=hf_vocab, merges=hf_merges)

    def _strip_token(self, token: str) -> str:
        if not token.startswith("[") or not token.endswith("]"):
            raise ValueError(
                f"Token must start with '[' and end with ']', got: {token}"
            )
        return token[1:-1]

    def _bres_line(self, x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
        if not all(isinstance(v, int) for v in (x0, y0, x1, y1)):
            raise TypeError(
                f"All coordinates must be integers, got {x0}, {y0}, {x1}, {y1} instead."
            )

        coords: list[tuple[int, int]] = []
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            coords.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

        return coords

    def _point_to_tokens(self, point: Point[int]) -> list[str]:
        bres_line = self._bres_line(0, 0, point.x, point.y)
        tokens: list[str] = []
        for p1, p2 in zip(bres_line, bres_line[1:]):
            coord = (p2[0] - p1[0], p2[1] - p1[1])
            tokens.append(self.COORD_TO_TOKEN[coord])
        return tokens

    def _merge_move_tokens(self, tokens: list[str]) -> list[str]:
        return tokens

    def _merge_tokens(self, tokens: list[str]) -> list[str]:
        merged_tokens = []
        for is_move, group in groupby(tokens, self._is_move_token):
            if is_move:
                merged_tokens.extend(self._merge_move_tokens(list(group)))
            else:
                merged_tokens.extend(group)
        return merged_tokens

    def tokenize(self, ink: Ink[int]) -> list[str]:
        tokens: list[str] = []
        prev_point: Point[int] = Point(x=0, y=0)
        for stroke in ink.strokes:
            for i, point in enumerate(stroke.points):
                delta = point - prev_point
                tokens.extend(self._point_to_tokens(delta))
                prev_point = point
                if i == 0:
                    tokens.append(self.DOWN)
            tokens.append(self.UP)
        return self._merge_tokens([self.BOS, *tokens, self.EOS])

    def _is_move_token(self, token: str) -> bool:
        if not all(arrow in "↑↓←→↖↗↙↘" for arrow in self._strip_token(token)):
            return False
        return True

    def _token_to_points(self, token: str) -> list[Point[int]]:
        if not self._is_move_token(token):
            raise ValueError(f"Invalid move token: {token}, expected format: '[↑↖←]'")

        coords = [
            self.TOKEN_TO_COORD[f"[{arrow}]"] for arrow in self._strip_token(token)
        ]
        curr_point = Point(x=0, y=0)
        points = [curr_point]
        for coord in coords:
            curr_point += Point(x=coord[0], y=coord[1])
            points.append(curr_point)
        return points

    def detokenize(self, tokens: list[str]) -> Ink[int]:
        try:
            start = next(i for i, t in enumerate(tokens) if t == self.BOS)
            tokens = tokens[start + 1 :]
        except StopIteration:
            warnings.warn(
                f"No {self.BOS} token found. This may lead to unexpected results.",
                UserWarning,
                stacklevel=2,
            )
            pass

        try:
            end = next(i for i, t in enumerate(tokens) if t == self.EOS)
            tokens = tokens[:end]
        except StopIteration:
            warnings.warn(
                f"No {self.EOS} token found. This may lead to unexpected results.",
                UserWarning,
                stacklevel=2,
            )
            pass

        curr_state = None
        curr_point = Point(x=0, y=0)
        curr_stroke = Stroke(points=[])
        ink = Ink(strokes=[])
        for token in tokens:
            match token:
                case self.BOS | self.EOS:
                    warnings.warn(
                        f"Unexpected token: {token}. Ignoring...",
                        UserWarning,
                        stacklevel=2,
                    )
                    continue
                case self.DOWN:
                    curr_state = self.DOWN
                case self.UP:
                    curr_state = self.UP
                    if curr_stroke.points:
                        ink.strokes.append(curr_stroke)
                        curr_stroke = Stroke(points=[])
                case _:  # Should be a move token like "[↑←←↓]"
                    if not self._is_move_token(token):
                        raise ValueError(f"Unexpected token: {token}")

                    points = [p + curr_point for p in self._token_to_points(token)]
                    curr_point = points[-1]
                    if curr_state == self.DOWN:
                        curr_stroke.points.extend(points)
        return ink

    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        return [self._vocab[token] for token in tokens]

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        reverse_vocab = {v: k for k, v in self._vocab.items()}
        return [reverse_vocab[id] for id in ids]

    def encode(self, ink: Ink[int]) -> list[int]:
        tokens = self.tokenize(ink)
        return self.convert_tokens_to_ids(tokens)

    def decode(self, ids: list[int]) -> Ink[int]:
        tokens = self.convert_ids_to_tokens(ids)
        return self.detokenize(tokens)


if __name__ == "__main__":
    ink = Ink.example()
    tokinkizer = Tokinkizer.from_pretrained()

    tokens = tokinkizer.tokenize(ink)
    for token in tokens:
        print(token)
    print(len(tokens))
    ink = tokinkizer.detokenize(tokens)
    ink.plot()
