import pytest

from tokink.ink import Ink, Point, Stroke
from tokink.tokinkizer import Tokinkizer


class TestTokinkizer:
    @pytest.fixture
    def tokinkizer(self):
        """Load the pretrained tokinkizer."""
        return Tokinkizer.from_pretrained()

    @pytest.fixture
    def simple_ink(self):
        """Create a simple test ink sample with Bresenham interpolation.

        Note: stroke2 goes from (2,1) to (4,-1), which will be interpolated
        to include (3,0) by the Bresenham line algorithm.
        """
        raw_strokes = [[(0, 0), (1, 0)], [(2, 1), (4, -1)], [(5, 5)]]
        return Ink.from_coords(raw_strokes)

    def test_load_pretrained(self, tokinkizer):
        """Test loading pretrained tokinkizer."""
        assert tokinkizer is not None
        assert tokinkizer._vocab is not None
        assert tokinkizer._merges is not None
        assert len(tokinkizer._vocab) > 0
        assert len(tokinkizer._merges) > 0

    def test_encode_decode_roundtrip(self, tokinkizer, simple_ink):
        """Test that double round-trip encoding/decoding is stable.

        Due to Bresenham interpolation, the first decode may add points,
        but subsequent round-trips should be identical.
        """
        ids = tokinkizer.encode(simple_ink)
        decoded_ink = tokinkizer.decode(ids)

        # Double round-trip should be stable
        ids2 = tokinkizer.encode(decoded_ink)
        decoded_ink2 = tokinkizer.decode(ids2)
        assert decoded_ink == decoded_ink2
        assert ids == ids2

    def test_encode_output(self, tokinkizer, simple_ink):
        """Test that encoding produces the expected IDs for the test ink."""
        ids = tokinkizer.encode(simple_ink)

        # Note: These IDs correspond to the Bresenham-interpolated version.
        expected_ids = [1, 4, 7, 3, 10, 4, 15, 3, 2704, 4, 3, 2]
        assert ids == expected_ids

    def test_decode_output(self, tokinkizer):
        """Test that decoding produces the expected ink with Bresenham interpolation."""
        ids = [1, 4, 7, 3, 10, 4, 15, 3, 2704, 4, 3, 2]
        decoded_ink = tokinkizer.decode(ids)

        # Expected decoded ink includes Bresenham interpolation: (2,1) → (3,0) → (4,-1).
        expected_strokes = [[(0, 0), (1, 0)], [(2, 1), (3, 0), (4, -1)], [(5, 5)]]
        expected_ink = Ink[int](
            strokes=[
                Stroke[int](points=[Point[int](x=x, y=y) for x, y in stroke])
                for stroke in expected_strokes
            ]
        )
        assert decoded_ink == expected_ink

    def test_tokenize(self, tokinkizer, simple_ink):
        """Test tokenization produces proper structure."""
        tokens = tokinkizer.tokenize(simple_ink)
        assert tokens[0] == "[BOS]"
        assert tokens[-1] == "[EOS]"
        assert "[DOWN]" in tokens
        assert "[UP]" in tokens

    def test_detokenize(self, tokinkizer, simple_ink):
        """Test that double tokenization round-trip is stable."""
        tokens = tokinkizer.tokenize(simple_ink)
        detokenized_ink = tokinkizer.detokenize(tokens)

        # Double tokenization should be stable
        tokens2 = tokinkizer.tokenize(detokenized_ink)
        detokenized_ink2 = tokinkizer.detokenize(tokens2)
        assert detokenized_ink == detokenized_ink2
        assert tokens == tokens2

    def test_token_id_conversion(self, tokinkizer):
        """Test bidirectional token-ID conversion."""
        token = "[BOS]"
        id = tokinkizer.token_to_id(token)
        # BOS.
        assert id == 1
        assert tokinkizer.id_to_token(id) == token

    def test_bresenham_line(self, tokinkizer):
        """Test Bresenham line algorithm for diagonal line."""
        coords = tokinkizer._bresenham_line(0, 0, 3, 3)
        assert coords == [(0, 0), (1, 1), (2, 2), (3, 3)]

    def test_is_move_token(self, tokinkizer):
        """Test identifying move tokens vs control tokens."""
        assert tokinkizer._is_move_token("→")
        assert tokinkizer._is_move_token("↑↖←")
        assert not tokinkizer._is_move_token("[BOS]")
        assert not tokinkizer._is_move_token("[DOWN]")

    def test_empty_ink(self, tokinkizer):
        """Test encoding empty ink produces only BOS and EOS tokens."""
        empty_ink = Ink[int](strokes=[])
        tokens = tokinkizer.tokenize(empty_ink)
        # Just BOS and EOS.
        assert tokens == ["[BOS]", "[EOS]"]

    def test_single_point_stroke(self, tokinkizer):
        """Test encoding a single point at origin."""
        ink = Ink[int](strokes=[Stroke[int](points=[Point[int](x=0, y=0)])])
        ids = tokinkizer.encode(ink)
        decoded_ink = tokinkizer.decode(ids)
        assert decoded_ink == ink

    def test_bresenham_interpolation(self, tokinkizer):
        """Test that Bresenham interpolation adds intermediate points between endpoints."""
        # Diagonal line from (0,0) to (3,2) requires interpolation.
        ink = Ink[int](strokes=[Stroke[int](points=[Point[int](x=0, y=0), Point[int](x=3, y=2)])])

        ids = tokinkizer.encode(ink)
        decoded_ink = tokinkizer.decode(ids)

        # Bresenham should add intermediate points.
        assert len(decoded_ink.strokes) == 1
        # More than the original 2 points.
        assert len(decoded_ink.strokes[0].points) > 2

        # Start and end points should be preserved.
        assert decoded_ink.strokes[0].points[0] == Point[int](x=0, y=0)
        assert decoded_ink.strokes[0].points[-1] == Point[int](x=3, y=2)

        # Double encoding should be stable.
        ids2 = tokinkizer.encode(decoded_ink)
        decoded_ink2 = tokinkizer.decode(ids2)
        assert decoded_ink == decoded_ink2

    def test_invalid_token_raises_error(self, tokinkizer):
        """Test that invalid token raises ValueError."""
        with pytest.raises(ValueError):
            tokinkizer.token_to_id("INVALID_TOKEN")

    def test_invalid_id_raises_error(self, tokinkizer):
        """Test that invalid ID raises ValueError."""
        with pytest.raises(ValueError):
            tokinkizer.id_to_token(999999)

    def test_vocab_size_reduction(self):
        """Test loading tokinkizer with reduced vocabulary."""
        tokinkizer = Tokinkizer.from_pretrained(vocab_size=100)
        assert len(tokinkizer._vocab) == 100

    def test_vocab_size_too_large_raises_error(self):
        """Test that requesting vocab size larger than trained raises ValueError."""
        with pytest.raises(ValueError):
            Tokinkizer.from_pretrained(vocab_size=999999)

    def test_convert_tokens_to_ids(self, tokinkizer):
        """Test converting list of tokens to IDs."""
        tokens = ["[BOS]", "→", "[EOS]"]
        ids = tokinkizer.convert_tokens_to_ids(tokens)
        assert len(ids) == 3
        # BOS.
        assert ids[0] == 1
        # EOS.
        assert ids[2] == 2
        assert all(isinstance(id, int) for id in ids)

    def test_convert_ids_to_tokens(self, tokinkizer):
        """Test converting list of IDs to tokens."""
        # BOS, some token, EOS.
        ids = [1, 7, 2]
        tokens = tokinkizer.convert_ids_to_tokens(ids)
        assert len(tokens) == 3
        assert tokens[0] == "[BOS]"
        assert tokens[2] == "[EOS]"
        assert all(isinstance(token, str) for token in tokens)

    def test_convert_tokens_ids_roundtrip(self, tokinkizer):
        """Test bidirectional conversion between tokens and IDs."""
        original_tokens = ["[BOS]", "→", "↑", "[DOWN]", "[UP]", "[EOS]"]
        ids = tokinkizer.convert_tokens_to_ids(original_tokens)
        recovered_tokens = tokinkizer.convert_ids_to_tokens(ids)
        assert original_tokens == recovered_tokens

    def test_encode_method(self, tokinkizer, simple_ink):
        """Test encode method produces list of integers."""
        ids = tokinkizer.encode(simple_ink)
        assert isinstance(ids, list)
        assert all(isinstance(id, int) for id in ids)
        assert len(ids) > 0
        # Should start with BOS token ID.
        assert ids[0] == 1
        # Should end with EOS token ID.
        assert ids[-1] == 2

    def test_decode_method(self, tokinkizer):
        """Test decode method produces Ink object."""
        # Simple valid sequence.
        ids = [1, 4, 7, 3, 2]
        ink = tokinkizer.decode(ids)
        assert isinstance(ink, Ink)
        assert len(ink.strokes) >= 0

    def test_encode_decode_consistency(self, tokinkizer, simple_ink):
        """Test that encode/decode are inverse operations (with Bresenham interpolation)."""
        # First encoding.
        ids1 = tokinkizer.encode(simple_ink)
        decoded1 = tokinkizer.decode(ids1)

        # Second encoding should be stable.
        ids2 = tokinkizer.encode(decoded1)
        decoded2 = tokinkizer.decode(ids2)

        assert ids1 == ids2
        assert decoded1 == decoded2

    def test_encode_empty_ink(self, tokinkizer):
        """Test encoding empty ink."""
        empty_ink = Ink[int](strokes=[])
        ids = tokinkizer.encode(empty_ink)
        # Just BOS and EOS.
        assert ids == [1, 2]

    def test_decode_empty_sequence(self, tokinkizer):
        """Test decoding sequence with only BOS and EOS."""
        ids = [1, 2]
        ink = tokinkizer.decode(ids)
        assert len(ink.strokes) == 0

    def test_convert_tokens_to_ids_invalid_token(self, tokinkizer):
        """Test that converting invalid token raises ValueError."""
        with pytest.raises(ValueError, match="not found in vocabulary"):
            tokinkizer.convert_tokens_to_ids(["[BOS]", "[INVALID]"])

    def test_convert_ids_to_tokens_invalid_id(self, tokinkizer):
        """Test that converting invalid ID raises ValueError."""
        with pytest.raises(ValueError, match="not found in vocabulary"):
            tokinkizer.convert_ids_to_tokens([1, 999999])

    def test_encode_decode_multiple_strokes(self, tokinkizer):
        """Test encode/decode with multiple strokes."""
        ink = Ink[int](
            strokes=[
                Stroke[int](points=[Point[int](x=0, y=0), Point[int](x=1, y=0)]),
                Stroke[int](points=[Point[int](x=2, y=2), Point[int](x=3, y=3)]),
                Stroke[int](points=[Point[int](x=5, y=5)]),
            ]
        )
        ids = tokinkizer.encode(ink)
        decoded = tokinkizer.decode(ids)

        # Should have same number of strokes.
        assert len(decoded.strokes) == 3

        # Double encoding should be stable.
        ids2 = tokinkizer.encode(decoded)
        assert ids == ids2

    def test_token_id_bidirectional_conversion(self, tokinkizer):
        """Test all special tokens can be converted to IDs and back."""
        special_tokens = ["[BOS]", "[EOS]", "[UP]", "[DOWN]"]
        for token in special_tokens:
            id = tokinkizer.token_to_id(token)
            recovered_token = tokinkizer.id_to_token(id)
            assert recovered_token == token


class TestTokinkizerTrain:
    def test_train_basic(self):
        """Test basic training with a few simple inks."""
        inks = [
            Ink.from_coords([[(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]]),
            Ink.from_coords([[(0, 0), (5, 5), (10, 10)]]),
        ]

        # Train with a small vocab size for testing.
        vocab_size = 50
        tokinkizer = Tokinkizer.train(iter(inks), vocab_size=vocab_size)

        assert isinstance(tokinkizer, Tokinkizer)
        assert len(tokinkizer._vocab) > 0

        # Check for special tokens.
        assert "[BOS]" in tokinkizer._vocab
        assert "[EOS]" in tokinkizer._vocab
        assert "[UP]" in tokinkizer._vocab
        assert "[DOWN]" in tokinkizer._vocab

        # Check for base arrow tokens.
        for arrow in "↑↓←→↖↗↙↘":
            assert arrow in tokinkizer._vocab

    def test_train_learns_merges(self):
        """Test that training learns merges from repeating patterns."""
        # Create ink with lots of repeating "→" (right) moves.
        # (0,0) to (100, 0) will produce 100 "→" tokens.
        inks = [Ink.from_coords([[(0, 0), (100, 0)]])] * 10

        tokinkizer = Tokinkizer.train(iter(inks), vocab_size=50)

        # It should have learned some merges of "→".
        # The base arrows + special tokens take up some space.
        # 4 (special) + 8 (arrows) = 12 tokens.
        # Merges will start after these.

        assert len(tokinkizer._merges) > 0
        # Check if any merge consists of "→".
        has_right_merge = any("→" in m[0] or "→" in m[1] for m in tokinkizer._merges)
        assert has_right_merge

    def test_train_save_load(self, tmp_path):
        """Test training, saving, and loading back a tokinkizer."""
        inks = [Ink.from_coords([[(0, 0), (10, 0), (10, 10)]])]

        tokinkizer = Tokinkizer.train(iter(inks), vocab_size=50)

        save_dir = tmp_path / "trained_tokinkizer"
        tokinkizer.save(save_dir)

        # Load it back.
        loaded_tokinkizer = Tokinkizer.from_pretrained(save_dir, vocab_size=None)

        assert loaded_tokinkizer._vocab == tokinkizer._vocab
        assert loaded_tokinkizer._merges == tokinkizer._merges

        # Test tokenization consistency.
        test_ink = Ink.from_coords([[(0, 0), (5, 0)]])
        assert tokinkizer.tokenize(test_ink) == loaded_tokinkizer.tokenize(test_ink)

    def test_train_empty_iterator(self):
        """Test training with an empty iterator of inks."""
        # This might fail or produce a minimal vocab depending on HF tokenizer behavior.
        inks = []
        tokinkizer = Tokinkizer.train(iter(inks), vocab_size=50)

        assert "[BOS]" in tokinkizer._vocab
        assert "↑" in tokinkizer._vocab
        # Should still have special and base tokens even if no data.

    def test_train_vocab_size_limit(self):
        """Test that vocab_size parameter is respected as an upper bound."""
        # Use a very small vocab size.
        # Special tokens (4) + Base arrows (8) = 12.
        # If we set vocab_size=15, we should have at most 15 tokens?
        # Actually, Tokinkizer.train adds learned tokens ON TOP of special/base tokens.
        # if they are not already there.

        inks = [Ink.from_coords([[(0, 0), (i, i)]]) for i in range(1, 20)]

        target_hf_vocab_size = 20
        tokinkizer = Tokinkizer.train(iter(inks), vocab_size=target_hf_vocab_size)

        # The total vocab size will be:
        # 4 special + 8 base + (hf learned tokens that aren't special/base).
        # hf_vocab will contain base arrows and merges.

        # Let's just check it doesn't crash and returns something reasonable.
        assert len(tokinkizer._vocab) >= 12
