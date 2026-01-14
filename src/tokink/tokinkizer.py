import json
from collections.abc import Iterator
from functools import lru_cache
from itertools import groupby
from pathlib import Path
from typing import Self

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

from tokink.ink import Ink, Point, Stroke
from tokink.utils import get_timestamp, warn


class Tokinkizer:
    # Class constants - special tokens with brackets
    _BOS = "[BOS]"
    _EOS = "[EOS]"
    _UP = "[UP]"
    _DOWN = "[DOWN]"

    # Arrow tokens - stored without brackets
    _COORD_TO_ARROW = {
        (0, 1): "↑",
        (0, -1): "↓",
        (-1, 0): "←",
        (1, 0): "→",
        (-1, 1): "↖",
        (1, 1): "↗",
        (-1, -1): "↙",
        (1, -1): "↘",
    }
    _ARROW_TO_COORD = {v: k for k, v in _COORD_TO_ARROW.items()}

    # Constructor
    def __init__(self, vocab: dict[str, int], merges: list[tuple[str, str]]):
        self._vocab = vocab
        self._merges = merges

        self._reverse_vocab = {v: k for k, v in vocab.items()}
        self._bpe = self._init_bpe(vocab, merges)

    # Class methods (factory methods)
    @classmethod
    def from_pretrained(
        cls, path: Path | str | None = None, *, vocab_size: int | None = 32_000
    ) -> Self:
        """
        Load a pretrained tokinkizer from the given path.

        Args:
            path: Path to the directory containing vocab.json and merges.txt.
                  If None, uses the default data directory.
            vocab_size: Optional target vocabulary size. If provided, the vocab
                       will be truncated to this size.

        Returns:
            A Tokinkizer instance loaded from the pretrained files.
        """
        return cls._from_pretrained_cached(path, vocab_size=vocab_size)

    @classmethod
    @lru_cache(maxsize=1)
    def _from_pretrained_cached(
        cls, path: Path | str | None = None, *, vocab_size: int | None = None
    ) -> Self:
        """Internal cached implementation of from_pretrained.

        This method is separated from from_pretrained to allow for LRU caching
        while maintaining proper type hinting on the public API.

        Args:
            path: Path to the directory containing vocab.json and merges.txt.
            vocab_size: Optional target vocabulary size for truncation.

        Returns:
            A cached Tokinkizer instance.
        """
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

    @classmethod
    def train(cls, inks: Iterator[Ink[int]], *, vocab_size: int = 100_000) -> Self:
        """
        Train a new tokinkizer from an iterator of ink samples using BPE.

        Args:
            inks: Iterator of Ink objects to train on.
            vocab_size: Target vocabulary size for BPE training.

        Returns:
            A trained Tokinkizer instance.
        """

        def get_token_iterator() -> Iterator[str]:
            """Generate token strings from inks for BPE training."""
            for ink in inks:
                base_tokens = cls._tokenize_base(ink)
                # Extract only move tokens (arrow tokens) for BPE training
                for is_move, group in groupby(base_tokens, cls._is_move_token):
                    if is_move:
                        yield "".join(group)

        # Initialize HuggingFace tokenizer with BPE model
        hf_tokenizer = Tokenizer(BPE())
        trainer = BpeTrainer(vocab_size=vocab_size, show_progress=True)
        hf_tokenizer.train_from_iterator(get_token_iterator(), trainer=trainer)

        # Extract vocab and merges from trained tokenizer
        tokenizer_data = json.loads(hf_tokenizer.to_str())
        hf_vocab = tokenizer_data["model"]["vocab"]
        hf_merges = tokenizer_data["model"]["merges"]

        # Build the full vocabulary including special tokens (save id 0 for padding)
        vocab: dict[str, int] = {}
        vocab_id = 1

        # Add special tokens first (these keep their brackets)
        for token in [cls._BOS, cls._EOS, cls._UP, cls._DOWN]:
            vocab[token] = vocab_id
            vocab_id += 1

        # Add base arrow tokens (no brackets)
        for token in cls._COORD_TO_ARROW.values():
            vocab[token] = vocab_id
            vocab_id += 1

        # Add BPE-learned tokens (no brackets)
        for token in sorted(hf_vocab, key=lambda x: hf_vocab[x]):
            if token not in vocab:
                vocab[token] = vocab_id
                vocab_id += 1

        # Convert merges from trained tokenizer
        # HuggingFace returns merges as a list of lists of strings (e.g., [["↑", "↑"], ...])
        merges = [tuple(merge) for merge in hf_merges]

        return cls(vocab=vocab, merges=merges)

    @classmethod
    def _tokenize_base(cls, ink: Ink[int]) -> list[str]:
        """
        Tokenize ink into base tokens without applying BPE merges.

        Converts ink strokes into a sequence of directional arrow tokens (no brackets),
        along with special tokens for pen state ([DOWN], [UP]) and sequence
        boundaries ([BOS], [EOS]).

        Args:
            ink: The ink drawing to tokenize.

        Returns:
            List of base tokens without any BPE merges applied.
        """
        tokens: list[str] = []
        prev_point: Point[int] = Point(x=0, y=0)
        for stroke in ink.strokes:
            for i, point in enumerate(stroke.points):
                delta = point - prev_point
                tokens.extend(cls._point_to_tokens(delta))
                prev_point = point
                if i == 0:
                    tokens.append(cls._DOWN)
            tokens.append(cls._UP)
        return [cls._BOS, *tokens, cls._EOS]

    @classmethod
    def _point_to_tokens(cls, point: Point[int]) -> list[str]:
        bres_line = cls._bresenham_line(0, 0, point.x, point.y)
        tokens: list[str] = []
        for p1, p2 in zip(bres_line, bres_line[1:]):
            coord = (p2[0] - p1[0], p2[1] - p1[1])
            tokens.append(cls._COORD_TO_ARROW[coord])
        return tokens

    @classmethod
    def _is_move_token(cls, token: str) -> bool:
        """Check if a token is a move token (contains only arrow characters, no brackets)."""
        return all(char in cls._ARROW_TO_COORD.keys() for char in token)

    @staticmethod
    def _bresenham_line(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
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

    # Public instance methods (main API)
    def tokenize(self, ink: Ink[int]) -> list[str]:
        base_tokens = self._tokenize_base(ink)
        return self._merge_tokens(base_tokens)

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
                case _:  # Should be a move token like "↑←←↓"
                    if not self._is_move_token(token):
                        raise ValueError(f"Unexpected token: {token}")

                    points = [p + curr_point for p in self._token_to_points(token)]
                    curr_point = points[-1]
                    if curr_state == self._DOWN:
                        curr_stroke.points.extend(points)
        return ink

    def encode(self, ink: Ink[int]) -> list[int]:
        tokens = self.tokenize(ink)
        return self.convert_tokens_to_ids(tokens)

    def decode(self, ids: list[int]) -> Ink[int]:
        tokens = self.convert_ids_to_tokens(ids)
        return self.detokenize(tokens)

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

    def save(self, save_path: Path | str | None = None) -> None:
        """Save the tokinkizer vocabulary and merges to files.

        Args:
            save_path: Directory path where vocab.json and merges.txt should be saved.
                      If None, generates a timestamped directory in the current working directory.
        """
        save_dir = Path(save_path or Path.cwd() / get_timestamp())
        save_dir.mkdir(parents=True, exist_ok=True)

        vocab_path = save_dir / "vocab.json"
        merges_path = save_dir / "merges.txt"

        # Save vocabulary
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(self._vocab, f, indent=2, ensure_ascii=False)

        # Save merges
        with open(merges_path, "w", encoding="utf-8") as f:
            for merge in self._merges:
                f.write(f"{merge[0]} {merge[1]}\n")

    # Private instance methods (helpers)
    def _init_bpe(self, vocab: dict[str, int], merges: list[tuple[str, str]]) -> BPE:
        # Filter vocab to only include move tokens (no special tokens)
        hf_vocab = {token: id for token, id in vocab.items() if self._is_move_token(token)}
        # Merges are already in the correct format (no brackets), just need to ensure they're tuples
        hf_merges = [tuple(merge) for merge in merges]
        return BPE(vocab=hf_vocab, merges=hf_merges)

    def _merge_tokens(self, tokens: list[str]) -> list[str]:
        merged_tokens = []
        for is_move, group in groupby(tokens, self._is_move_token):
            if is_move:
                merged_tokens.extend(self._merge_move_tokens(list(group)))
            else:
                merged_tokens.extend(group)
        return merged_tokens

    def _merge_move_tokens(self, tokens: list[str]) -> list[str]:
        # Tokens are already in the correct format (no brackets for move tokens)
        hf_tokens = self._bpe.tokenize("".join(tokens))
        return [token.value for token in hf_tokens]

    def _token_to_points(self, token: str) -> list[Point[int]]:
        if not self._is_move_token(token):
            raise ValueError(f"Invalid move token: {token}, expected format: '↑↖←'")

        # Each character in the token is an arrow
        coords = [self._ARROW_TO_COORD[arrow] for arrow in token]
        points = []
        curr_point = Point(x=0, y=0)
        for coord in coords:
            curr_point += Point.from_coords(coord)
            points.append(curr_point)
        return points
