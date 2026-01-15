"""
Training a Custom Tokinkizer
=============================

This example demonstrates how to train your own BPE tokenizer for digital ink.
You might want to train a custom tokenizer when:

1. Your ink has unique characteristics (specific writing styles, languages, symbols)
2. You need a different vocabulary size for your model architecture
3. You want to optimize compression for your specific dataset

This script shows:
- Loading and preprocessing training data
- Training tokenizers with different vocabulary sizes
- Saving and loading custom tokenizers
- Evaluating compression efficiency
- Comparing with the pretrained tokenizer
"""

from pathlib import Path

from tokink import Ink, Tokinkizer
from tokink.processor import scale, to_int

# ==============================================================================
# Configuration
# ==============================================================================

SCALE_FACTOR = 1 / 16  # Scale down coordinates for better compression
VOCAB_SIZE = 100  # Target vocabulary size
OUTPUT_DIR = Path("trained_tokenizers")


# ==============================================================================
# Helper Functions
# ==============================================================================


def preprocess_ink(ink: Ink) -> Ink:
    """
    Preprocess ink for tokenization.

    Scaling down coordinates reduces the sequence length, making BPE more
    effective at learning common patterns.
    """
    return to_int(scale(ink, SCALE_FACTOR))


def evaluate_tokenizer(tokenizer: Tokinkizer, test_inks: list[Ink[int]]) -> dict:
    """
    Evaluate tokenizer compression efficiency.

    Args:
        tokenizer: The tokenizer to evaluate
        test_inks: List of preprocessed ink samples to test on

    Returns:
        Dictionary with evaluation metrics
    """
    total_points = 0
    total_tokens = 0

    for ink in test_inks:
        total_points += len(ink)
        tokens = tokenizer.encode(ink)
        total_tokens += len(tokens)

    compression_ratio = total_points / total_tokens if total_tokens > 0 else 0

    return {
        "total_points": total_points,
        "total_tokens": total_tokens,
        "compression_ratio": compression_ratio,
    }


# ==============================================================================
# Main Training Pipeline
# ==============================================================================

print("Custom Tokenizer Training Example")
print("=" * 70)

# 1. Load training dataset
# Replace this with your actual dataset loading logic
print("\n[1/5] Loading training dataset...")

# For demonstration, we'll use the example ink multiple times
# In practice, load your actual dataset here:
#   dataset = [Ink.from_json(path) for path in Path("data/").glob("*.json")]
dataset = [Ink.example() for _ in range(100)]  # Mock dataset

print(f"Loaded {len(dataset)} ink samples")

# 2. Preprocess the dataset
print("\n[2/5] Preprocessing dataset...")
preprocessed_dataset = [preprocess_ink(ink) for ink in dataset]

# Calculate dataset statistics
total_strokes = sum(len(ink.strokes) for ink in preprocessed_dataset)
total_points = sum(len(ink) for ink in preprocessed_dataset)
avg_points_per_sample = total_points / len(preprocessed_dataset)

print(f"Total strokes: {total_strokes:,}")
print(f"Total points: {total_points:,}")
print(f"Average points per sample: {avg_points_per_sample:.1f}")

# 3. Train tokenizer
print(f"\n[3/4] Training tokenizer with vocab_size={VOCAB_SIZE:,}...")
OUTPUT_DIR.mkdir(exist_ok=True)

# Train the tokenizer
# Note: We pass an iterator to avoid loading all data into memory at once
tokenizer = Tokinkizer.train(iter(preprocessed_dataset), vocab_size=VOCAB_SIZE)

# Save the tokenizer
tokenizer.save(OUTPUT_DIR)
print(f"Saved to {OUTPUT_DIR}/")

# 4. Evaluate tokenizer
print("\n[4/4] Evaluating tokenizer...")

# Use a held-out test set (or the same data for demonstration)
test_set = preprocessed_dataset[:10]

metrics = evaluate_tokenizer(tokenizer, test_set)
print(f"\nTotal points: {metrics['total_points']:,}")
print(f"Total tokens: {metrics['total_tokens']:,}")
print(f"Compression ratio: {metrics['compression_ratio']:.2f}x")

# ==============================================================================
# Usage Example: Loading a Custom Tokenizer
# ==============================================================================

print("\n" + "=" * 70)
print("Loading and using a custom tokenizer:")
print("=" * 70)

# Load the custom tokenizer
custom_tokenizer = Tokinkizer.from_pretrained(OUTPUT_DIR, vocab_size=VOCAB_SIZE)

# Use it for encoding/decoding
test_ink = preprocessed_dataset[0]
tokens = custom_tokenizer.encode(test_ink)
reconstructed = custom_tokenizer.decode(tokens)

print(f"\nOriginal points: {len(test_ink)}")
print(f"Token count: {len(tokens)}")
print(f"Compression: {len(test_ink) / len(tokens):.2f}x")
