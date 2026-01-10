import json
from functools import lru_cache
from itertools import groupby
from pathlib import Path
from typing import Self

from tokenizers.models import BPE

from tokink.ink import Ink, Point, Stroke
from tokink.utils import warn


class Tokinkizer:
    _BOS = "[BOS]"
    _EOS = "[EOS]"
    _UP = "[UP]"
    _DOWN = "[DOWN]"

    _COORD_TO_TOKEN = {
        (0, 1): "[↑]",
        (0, -1): "[↓]",
        (-1, 0): "[←]",
        (1, 0): "[→]",
        (-1, 1): "[↖]",
        (1, 1): "[↗]",
        (-1, -1): "[↙]",
        (1, -1): "[↘]",
    }
    _TOKEN_TO_COORD = {v: k for k, v in _COORD_TO_TOKEN.items()}
    _ARROWS = "↑↓←→↖↗↙↘"

    def __init__(self, vocab: dict[str, int], merges: list[tuple[str, str]]):
        self._vocab = vocab
        self._merges = merges

        self._reverse_vocab = {v: k for k, v in vocab.items()}
        self._bpe = self._init_bpe(vocab, merges)

    @classmethod
    @lru_cache(maxsize=1)
    def from_pretrained(
        cls, path: Path | str | None = None, *, vocab_size: int | None = None
    ) -> Self:
        if path is None:
            path = Path(__file__).parent / "data"

        vocab_path = Path(path) / "vocab.json"
        merges_path = Path(path) / "merges.txt"

        with open(vocab_path, "r") as f:
            vocab = json.load(f)
        with open(merges_path, "r") as f:
            merges = []
            for line in f:
                if not len(parts := line.strip().split()) == 2:
                    raise ValueError(
                        f"Invalid merge line: {line}, "
                        "Expected exactly 2 whitespace-separated tokens."
                    )

                merges.append((parts[0], parts[1]))

        if vocab_size is None:
            return cls(vocab=vocab, merges=merges)

        if (reduce_count := len(vocab) - vocab_size) < 0:
            raise ValueError(f"Target vocab size {vocab_size} larger than train size {len(vocab)}")

        if reduce_count > 0:
            vocab = {k: v for i, (k, v) in enumerate(vocab.items()) if i < vocab_size}
            merges = merges[:-reduce_count]
        return cls(vocab=vocab, merges=merges)

    def _init_bpe(self, vocab: dict[str, int], merges: list[tuple[str, str]]) -> BPE:
        hf_vocab = {
            self._strip_token(token): id
            for token, id in vocab.items()
            if self._is_move_token(token)
        }
        hf_merges = [(self._strip_token(merge[0]), self._strip_token(merge[1])) for merge in merges]
        return BPE(vocab=hf_vocab, merges=hf_merges)

    def _strip_token(self, token: str) -> str:
        if not token.startswith("[") or not token.endswith("]"):
            raise ValueError(f"Token must start with '[' and end with ']', got: {token}")
        return token[1:-1]

    def _wrap_token(self, token: str) -> str:
        return f"[{token}]"

    def _bresenham_line(self, x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
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
        bres_line = self._bresenham_line(0, 0, point.x, point.y)
        tokens: list[str] = []
        for p1, p2 in zip(bres_line, bres_line[1:]):
            coord = (p2[0] - p1[0], p2[1] - p1[1])
            tokens.append(self._COORD_TO_TOKEN[coord])
        return tokens

    def _merge_move_tokens(self, tokens: list[str]) -> list[str]:
        hf_tokens = self._bpe.tokenize("".join(tokens))
        return [self._wrap_token(token.value) for token in hf_tokens]

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
                    tokens.append(self._DOWN)
            tokens.append(self._UP)
        return self._merge_tokens([self._BOS, *tokens, self._EOS])

    def _is_move_token(self, token: str) -> bool:
        return all(arrow in self._ARROWS for arrow in self._strip_token(token))

    def _token_to_points(self, token: str) -> list[Point[int]]:
        if not self._is_move_token(token):
            raise ValueError(f"Invalid move token: {token}, expected format: '[↑↖←]'")

        coords = [self._TOKEN_TO_COORD[f"[{arrow}]"] for arrow in self._strip_token(token)]
        points = []
        curr_point = Point(x=0, y=0)
        for coord in coords:
            curr_point += Point(x=coord[0], y=coord[1])
            points.append(curr_point)
        return points

    def detokenize(self, tokens: list[str]) -> Ink[int]:
        if not tokens:
            raise ValueError("No tokens provided")

        if tokens[0] == self._BOS:
            tokens = tokens[1:]
        else:
            warn(f"First token {tokens[0]} is not {self._BOS}. Ignoring...")

        curr_state = self._UP
        curr_point = Point(x=0, y=0)
        curr_stroke = Stroke(points=[])
        ink = Ink(strokes=[])
        for token in tokens:
            match token:
                case self._BOS:
                    warn(f"Unexpected token: {token}. Ignoring...")
                case self._EOS:
                    return ink
                case self._DOWN:
                    curr_state = self._DOWN
                    curr_stroke.points.append(curr_point)
                case self._UP:
                    curr_state = self._UP
                    if curr_stroke.points:
                        ink.strokes.append(curr_stroke)
                        curr_stroke = Stroke(points=[])
                    else:
                        warn("No points in stroke. This may lead to unexpected results.")
                case _:  # Should be a move token like "[↑←←↓]"
                    if not self._is_move_token(token):
                        raise ValueError(f"Unexpected token: {token}")

                    points = [p + curr_point for p in self._token_to_points(token)]
                    curr_point = points[-1]
                    if curr_state == self._DOWN:
                        curr_stroke.points.extend(points)
        return ink

    def token_to_id(self, token: str) -> int:
        if token not in self._vocab:
            raise ValueError(f"Token '{token}' not found in vocabulary")
        return self._vocab[token]

    def id_to_token(self, id: int) -> str:
        if id not in self._reverse_vocab:
            raise ValueError(f"ID {id} not found in vocabulary")
        return self._reverse_vocab[id]

    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        return [self.token_to_id(token) for token in tokens]

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        return [self.id_to_token(id) for id in ids]

    def encode(self, ink: Ink[int]) -> list[int]:
        tokens = self.tokenize(ink)
        return self.convert_tokens_to_ids(tokens)

    def decode(self, ids: list[int]) -> Ink[int]:
        tokens = self.convert_ids_to_tokens(ids)
        return self.detokenize(tokens)
