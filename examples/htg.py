"""
Handwritten Text Generation (HTG) Example

This example demonstrates a complete HTG pipeline using Tokink:
1. Load dataset (text and ink)
2. Preprocess and augment ink
3. Tokenize ink into discrete token IDs
4. Train a mock HTG model
5. Generate handwriting from text prompts

"""

from tokink import Ink, Tokinkizer
from tokink.processor import jitter, resample, rotate, scale, smooth, to_int

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


def postprocess_generated(ink: Ink) -> Ink:
    """
    Post-process generated ink for smooth, natural appearance.

    Steps:
    1. Scale back to original coordinate space
    2. Resample to increase point density
    3. Apply Savitzky-Golay smoothing to reduce tokenization artifacts
    """
    ink = scale(ink, 1 / SCALE_FACTOR)
    ink = resample(ink, sample_every=2)
    ink = smooth(ink)
    return ink


# Mock HTG Model (replace with your actual generative model)
class HTGModel:
    def __init__(self, tokenizer: Tokinkizer):
        self.tokenizer = tokenizer

    def train(self, dataset: list[tuple[str, list[int]]]) -> None:
        print(f"Training on {len(dataset)} samples...")
        # TODO: Implement training loop (e.g., with PyTorch/JAX)
        # Model learns: text → token sequence distribution

    def generate(self, prompt: str) -> list[int]:
        """Generate token IDs from text prompt (mock implementation)."""
        # TODO: Implement text-to-tokens generation with beam search/sampling
        # For now, just return tokens from example ink
        example_ink = to_int(preprocess_ink(Ink.example()))
        return self.tokenizer.encode(example_ink)


# ==============================================================================
# Training and Generation Pipeline
# ==============================================================================

print("Handwritten Text Generation Example")
print("=" * 70)

# 1. Load dataset (replace with your data)
training_data = [
    ("By Trevor Williams. A move", Ink.example()),
    # Add more (text, ink) pairs...
]
print(f"\n[1/5] Loaded {len(training_data)} training samples")

# 2. Preprocess: scale down and augment
processed_data = []
for text, ink in training_data:
    # Original (preprocessed)
    processed_data.append((text, to_int(preprocess_ink(ink))))
    # Augmented (preprocess then augment)
    processed_data.append((text, to_int(augment_ink(preprocess_ink(ink)))))

print(f"[2/5] Preprocessed and augmented to {len(processed_data)} samples")

# 3. Tokenize
tokenizer = Tokinkizer.from_pretrained(vocab_size=VOCAB_SIZE)
tokenized_data = [(text, tokenizer.encode(ink)) for text, ink in processed_data]
print("[3/5] Tokenized dataset")

# 4. Train model
model = HTGModel(tokenizer)
model.train(tokenized_data)
print("[4/5] Training complete")

# 5. Generate handwriting from text
prompt = "By Trevor Williams. A move"
print(f'[5/5] Generating from prompt: "{prompt}"')

# Generate, decode, and post-process
generated_tokens = model.generate(prompt)
raw_ink = tokenizer.decode(generated_tokens)
smooth_ink = postprocess_generated(raw_ink)

# Visualize the result
smooth_ink.plot()
