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
        strokes: list[Stroke[int]] = [
            Stroke[int](points=[Point[int](x=x, y=y) for x, y in stroke]) for stroke in raw_strokes
        ]
        return Ink[int](strokes=strokes)

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
        print(ids)
        # Note: These IDs correspond to the Bresenham-interpolated version
        expected_ids = [1, 4, 7, 3, 10, 4, 15, 3, 2704, 4, 3, 2]
        assert ids == expected_ids

    def test_decode_output(self, tokinkizer):
        """Test that decoding produces the expected ink with Bresenham interpolation."""
        ids = [1, 4, 7, 3, 10, 4, 15, 3, 2704, 4, 3, 2]
        decoded_ink = tokinkizer.decode(ids)
        print(ids)
        print(decoded_ink)
        
        # Expected decoded ink includes Bresenham interpolation: (2,1) → (3,0) → (4,-1)
        expected_strokes = [
            [(0, 0), (1, 0)],
            [(2, 1), (3, 0), (4, -1)],
            [(5, 5)]
        ]
        expected_ink = Ink[int](strokes=[
            Stroke[int](points=[Point[int](x=x, y=y) for x, y in stroke])
            for stroke in expected_strokes
        ])
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
        assert id == 1
        assert tokinkizer.id_to_token(id) == token

    def test_bresenham_line(self, tokinkizer):
        """Test Bresenham line algorithm for diagonal line."""
        coords = tokinkizer._bresenham_line(0, 0, 3, 3)
        assert coords == [(0, 0), (1, 1), (2, 2), (3, 3)]

    def test_is_move_token(self, tokinkizer):
        """Test identifying move tokens vs control tokens."""
        assert tokinkizer._is_move_token("[→]")
        assert tokinkizer._is_move_token("[↑↖←]")
        assert not tokinkizer._is_move_token("[BOS]")
        assert not tokinkizer._is_move_token("[DOWN]")

    def test_empty_ink(self, tokinkizer):
        """Test encoding empty ink produces only BOS and EOS tokens."""
        empty_ink = Ink[int](strokes=[])
        tokens = tokinkizer.tokenize(empty_ink)
        assert tokens == ["[BOS]", "[EOS]"]

    def test_single_point_stroke(self, tokinkizer):
        """Test encoding a single point at origin."""
        ink = Ink[int](strokes=[Stroke[int](points=[Point[int](x=0, y=0)])])
        ids = tokinkizer.encode(ink)
        decoded_ink = tokinkizer.decode(ids)
        assert decoded_ink == ink
    
    def test_bresenham_interpolation(self, tokinkizer):
        """Test that Bresenham interpolation adds intermediate points between endpoints."""
        # Diagonal line from (0,0) to (3,2) requires interpolation
        ink = Ink[int](strokes=[
            Stroke[int](points=[Point[int](x=0, y=0), Point[int](x=3, y=2)])
        ])
        
        ids = tokinkizer.encode(ink)
        decoded_ink = tokinkizer.decode(ids)
        
        # Bresenham should add intermediate points
        assert len(decoded_ink.strokes) == 1
        assert len(decoded_ink.strokes[0].points) > 2  # More than the original 2 points
        
        # Start and end points should be preserved
        assert decoded_ink.strokes[0].points[0] == Point[int](x=0, y=0)
        assert decoded_ink.strokes[0].points[-1] == Point[int](x=3, y=2)
        
        # Double encoding should be stable
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
