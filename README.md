# Tokink

**Tokink** is a Byte-Pair Encoding (BPE) tokenizer designed specifically for digital ink (online handwriting). It enables a compressed and discrete representation of digital ink, improving compatibility with the transformer architecture.

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

Digital ink is crucial for digital note-taking applications. Two important ML tasks involve digital ink: **Handwritten Text Recognition (HTR)** and **Handwritten Text Generation (HTG)**.

### Traditional Representation

The natural representation of digital ink as `list[list[tuple[float, float]]]` (strokes containing points) is awkward for modern ML models. Consider drawing an equal sign:

```python
[
    [(0.0, 0.0), (2.0, 0.0)],  # First horizontal line
    [(0.0, 1.0), (2.0, 1.0)]   # Second horizontal line
]
```

A common solution is the **Point-3** format: `list[(Δx, Δy, p)]`, where `p` is a binary pen state (pen down/up):

```python
[(2.0, 0.0, 1), (-2.0, 1.0, 0), (2.0, 0.0, 1)]
```

However, this representation has issues:
- **No compression**: Every coordinate change increases sequence length by one
- **Incompatible with transformers**: Transformers more naturally model distribution of discrete sequences

### Naive Token Approach

One might try rounding coordinates to integers and treating each *xy*-coordinate as a token:

```python
["[0, 0]", "[2, 0]", "[UP]", "[0, 1]", "[2, 1]"]
```

This has critical problems:
- **Massive vocabulary**: A 1000×1000 canvas requires 1 million tokens
- **Extreme sparsity**: Low compression with BPE
- **Out-of-vocabulary (OOV)**: When training on a 1000×1000 canvas, "[1001, 1000]" token is not in model's vocabulary

### The Tokink Solution

**Tokink** uses a novel approach inspired by [Bresenham's line algorithm](https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm), which rasterizes lines on pixelated displays. We break all pen movements into 8 directional arrows: `↑, ↓, ←, →, ↖, ↗, ↙, ↘`.

For example, rendering a line from (0, 0) to (10, 4):

![Bresenham's Line](assets/bresenhams_line.png)

Combined with special `[UP]` and `[DOWN]` tokens for pen state, we can express **any digital ink using just 10 base tokens**. The equal sign becomes:

```python
["[DOWN]", "→", "→", "[UP]", "←", "↙", "[DOWN]", "→", "→", "[UP]"]
```

This has several benefits:
- **Tiny vocabulary**: Only 10 base tokens before BPE
- **High compression**: Reduced sparsity enables effective BPE merging
- **No OOV**: Every digital ink can be tokenized by the 10 base tokens.

Example tokenization of "hello":

![Hello Tokens](assets/hello_tokens.png)

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
