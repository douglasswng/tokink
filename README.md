# Tokink

**Tokink** is the accompanying library to [ScribeTokens: Fixed-Vocabulary Tokenization of Digital Ink](https://arxiv.org/abs/2603.02805), a Byte-Pair Encoding (BPE) tokenizer designed specifically for digital ink (online handwriting). It enables a compressed and discrete representation of digital ink, improving compatibility with the transformer architecture.

## Installation

```bash
pip install tokink
```

## Quick Start

```python
from tokink import Ink, Tokinkizer
from tokink.processor import scale, to_int

# Load or create digital ink
ink = Ink.example()  # or Ink.from_json("path/to/ink.json")

# Preprocess: scale down for better compression
ink = to_int(scale(ink, 1/16))

# Initialize tokenizer
tokenizer = Tokinkizer.from_pretrained(vocab_size=32_000)

# Encode ink to tokens
tokens = tokenizer.encode(ink)

# Decode tokens back to ink
reconstructed_ink = tokenizer.decode(tokens)

# Visualize
reconstructed_ink.plot()
```

> 💡 **Try it interactively**: Check out [`examples/quickstart.ipynb`](examples/quickstart.ipynb) for a hands-on notebook walkthrough.

## Background & Motivation

Digital ink is naturally represented as lists of strokes and points — verbose and continuous-valued, which is awkward for transformers that work best with discrete, compressed sequences. Naive discretization (one token per coordinate) leads to massive vocabularies and out-of-vocabulary problems.

**Tokink** takes a different approach inspired by [Bresenham's line algorithm](https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm), which rasterizes lines on pixelated displays. We decompose all pen movements into 8 directional arrows: `↑, ↓, ←, →, ↖, ↗, ↙, ↘`.

For example, rendering a line from (0, 0) to (10, 4):

![Bresenham's Line](assets/bresenham_decomposition.svg)

Combined with special `[UP]` and `[DOWN]` tokens for pen state, **any digital ink can be expressed using just 10 base tokens** — giving us a tiny vocabulary, high BPE compression, and zero out-of-vocabulary issues.

Example Tokenization: Pen strokes are decomposed into unit directional steps via Bresenham's algorithm, then compressed with BPE. Each color denotes a distinct BPE token; faint colors indicate pen-in-air movement between strokes. The zoom shows the sequence of arrows making up an example token.

![scribe](assets/scribetokens.svg)

## Usage Examples

### Handwritten Text Recognition (HTR)

Complete pipeline for recognizing handwritten text:

```python
from tokink import Ink, Tokinkizer
from tokink.processor import jitter, rotate, scale, to_int

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

# Load dataset
dataset = [
    (Ink.example(), "By Trevor Williams. A move"),
    # Add more (ink, label) pairs...
]

# Preprocess and augment
processed_data = []
for ink, label in dataset:
    # Original (preprocessed)
    processed_data.append((to_int(preprocess_ink(ink)), label))
    # Augmented (preprocess then augment)
    processed_data.append((to_int(augment_ink(preprocess_ink(ink))), label))

# Tokenize
tokenizer = Tokinkizer.from_pretrained(vocab_size=VOCAB_SIZE)
tokenized_data = [(tokenizer.encode(ink), label) for ink, label in processed_data]

# Train your model with tokenized data
# model.train(tokenized_data)
```

See [`examples/htr.py`](examples/htr.py) for the complete example.

### Handwritten Text Generation (HTG)

Generate handwriting from text prompts:

```python
from tokink import Ink, Tokinkizer
from tokink.processor import resample, scale, smooth, to_int

SCALE_FACTOR = 1 / 16
VOCAB_SIZE = 32_000

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

# Initialize tokenizer
tokenizer = Tokinkizer.from_pretrained(vocab_size=VOCAB_SIZE)

# Generate tokens from your model
# generated_tokens = model.generate("Hello world")

# Decode and post-process
raw_ink = tokenizer.decode(generated_tokens)
smooth_ink = postprocess_generated(raw_ink)
smooth_ink.plot()
```

See [`examples/htg.py`](examples/htg.py) for the complete example.

### Training Your Own Tokenizer

Train a custom tokenizer on your own digital ink dataset:

```python
from tokink import Ink, Tokinkizer
from tokink.processor import scale, to_int

# Load your dataset
dataset = [
    Ink.from_json("sample1.json"),
    Ink.from_json("sample2.json"),
    # ... more ink samples
]

# Preprocess: scale down for better compression
SCALE_FACTOR = 1 / 16
preprocessed = (to_int(scale(ink, SCALE_FACTOR)) for ink in dataset)

# Train tokenizer with custom vocabulary size
tokenizer = Tokinkizer.train(preprocessed, vocab_size=50_000)

# Save for later use
tokenizer.save("my_tokenizer/")

# Load your custom tokenizer
custom_tokenizer = Tokinkizer.from_pretrained("my_tokenizer/")
```

**When to train your own:**
- Your ink has unique characteristics (e.g., specific writing styles, languages, or symbols)
- You need a different vocabulary size for your model architecture
- You want to optimize compression for your specific use case

See [`examples/train_tokenizer.py`](examples/train_tokenizer.py) for a complete example with evaluation and best practices.

## API Reference

### Core Classes

- **`Ink`**: Represents digital ink as strokes
  - `Ink.example()`: Load example ink
  - `Ink.from_json(path)`: Load from JSON file
  - `plot()`: Visualize the ink

- **`Tokinkizer`**: BPE tokenizer for digital ink
  - `from_pretrained(vocab_size)`: Load pretrained tokenizer
  - `encode(ink)`: Convert ink to token IDs
  - `decode(tokens)`: Convert token IDs back to ink

### Preprocessing Functions

All available in `tokink.processor`:

- `scale(ink, factor)`: Scale coordinates
- `rotate(ink, angle_degrees)`: Rotate ink
- `jitter(ink, sigma)`: Add Gaussian noise for augmentation
- `resample(ink, sample_every)`: Resample points for density control
- `smooth(ink)`: Apply Savitzky-Golay smoothing
- `to_int(ink)`: Convert float coordinates to integers

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
