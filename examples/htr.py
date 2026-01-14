"""
Handwritten Text Recognition (HTR) Example

This example demonstrates a complete HTR pipeline using Tokink:
1. Load dataset (ink and labels)
2. Preprocess and augment ink
3. Tokenize ink into discrete token IDs
4. Train a mock HTR model
5. Perform inference on new ink

"""

from tokink import Ink, Tokinkizer
from tokink.processor import jitter, rotate, scale, to_int

# Configuration
SCALE_FACTOR = 1 / 16
VOCAB_SIZE = 32_000


def preprocess_ink(ink: Ink) -> Ink:
    """Scale down coordinates for better tokenization compression."""
    return scale(ink, SCALE_FACTOR)


def augment_ink(ink: Ink) -> Ink:
    """Apply rotation and jittering for data augmentation."""
    ink = rotate(ink, angle_degrees=5)
    ink = jitter(ink, sigma=0.5)
    return ink


# Mock HTR Model (replace with your actual model)
class HTRModel:
    def train(self, dataset: list[tuple[list[int], str]]) -> None:
        print(f"Training on {len(dataset)} samples...")
        # TODO: Implement training loop (e.g., with PyTorch/JAX)

    def predict(self, token_ids: list[int]) -> str:
        return "By Trevor Williams. A move"  # Mock prediction


# ==============================================================================
# Training Pipeline
# ==============================================================================

print("Handwritten Text Recognition Example")
print("=" * 70)

# 1. Load dataset (replace with your data)
dataset = [
    (Ink.example(), "By Trevor Williams. A move"),
    # Add more (ink, label) pairs...
]
print(f"\n[1/5] Loaded {len(dataset)} samples")

# 2. Preprocess: scale down and augment
processed_data = []
for ink, label in dataset:
    # Original (preprocessed)
    processed_data.append((to_int(preprocess_ink(ink)), label))
    # Augmented (preprocess then augment)
    processed_data.append((to_int(augment_ink(preprocess_ink(ink))), label))

print(f"[2/5] Preprocessed and augmented to {len(processed_data)} samples")

# 3. Tokenize
tokenizer = Tokinkizer.from_pretrained(vocab_size=VOCAB_SIZE)
tokenized_data = [(tokenizer.encode(ink), label) for ink, label in processed_data]

original_points = len(dataset[0][0])
token_count = len(tokenized_data[0][0])
print("[3/5] Tokenized dataset")

# 4. Train model
model = HTRModel()
model.train(tokenized_data)
print("[4/5] Training complete")

# 5. Inference
test_ink = preprocess_ink(Ink.example())
test_tokens = tokenizer.encode(to_int(test_ink))
prediction = model.predict(test_tokens)

print(f'[5/5] Prediction: "{prediction}"')
print("=" * 70)
