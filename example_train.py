"""
Example script demonstrating how to train a Tokinkizer.

This script shows how to use the Tokinkizer.train() classmethod
to train a new tokenizer from scratch using ink samples.
"""

from collections.abc import Iterator

from tokink.ink import Ink
from tokink.tokinkizer import Tokinkizer


def get_training_inks() -> Iterator[Ink[int]]:
    """
    Generator function that yields Ink samples for training.

    In a real scenario, this would load your training data from files,
    a database, or another data source.
    """
    # Example 1: Simple square
    yield Ink.from_coords([[(0, 0), (100, 0), (100, 100), (0, 100), (0, 0)]])

    # Example 2: Two separate strokes
    yield Ink.from_coords([[(0, 0), (50, 50)], [(100, 0), (50, 50)]])

    # Example 3: More complex shape
    yield Ink.from_coords(
        [[(0, 0), (20, 30), (40, 20), (60, 40), (80, 30), (100, 50)], [(50, 60), (50, 100)]]
    )

    # In practice, you would load many more samples, e.g.:
    # for file_path in training_data_paths:
    #     with open(file_path, 'r') as f:
    #         ink = Ink.model_validate_json(f.read())
    #         yield ink


def main():
    print("Training a new Tokinkizer...")
    print("=" * 60)

    # Train the tokenizer
    tokinkizer = Tokinkizer.train(
        inks=get_training_inks(),
    )

    print("\nTraining complete!")
    print(f"Vocabulary size: {len(tokinkizer._vocab)}")
    print(f"Number of merges: {len(tokinkizer._merges)}")

    # Test the trained tokenizer
    print("\n" + "=" * 60)
    print("Testing the trained tokenizer...")
    print("=" * 60)

    test_ink = Ink.from_coords([[(0, 0), (50, 50), (100, 0)]])
    print(f"\nTest ink: {test_ink}")

    tokens = tokinkizer.tokenize(test_ink)
    print(f"\nTokens: {tokens}")
    print(f"Number of tokens: {len(tokens)}")

    # Save the trained tokenizer
    save_path = "./trained_tokinkizer"
    tokinkizer.save(save_path)
    print(f"\nTokinkizer saved to: {save_path}")

    # Load it back to verify (use vocab_size=None to load the full trained vocab)
    loaded_tokinkizer = Tokinkizer.from_pretrained(save_path, vocab_size=None)
    loaded_tokens = loaded_tokinkizer.tokenize(test_ink)
    print(f"\nLoaded tokenizer produces same tokens: {tokens == loaded_tokens}")


if __name__ == "__main__":
    main()
