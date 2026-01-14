# Tokink

**Tokink** is a tokenizer for digital ink data that uses Byte-Pair Encoding (BPE) on directional movements. It converts handwriting and drawings into efficient token sequences suitable for machine learning models.

## Installation

```bash
pip install tokink
```

Or install from source:

```bash
git clone https://github.com/douglasswng/tokink.git
cd tokink
pip install -e .
```

## Background & Motivation

Despite the prevalence of digital ink applications (handwriting recognition, sketch understanding, drawing generation), there is no unified representation. Prior approaches fall into two paradigms: **vector-based** and **token-based** representations.

### Vector Representations

Vector representations model digital ink as sequences of continuous vectors, where the next *xy*-coordinate is typically modeled as a mixture of 2-dimensional Gaussians. Common formats include:

- **Point-3**: `(Δx, Δy, p)` where `p` is a binary pen state (0=down, 1=up)
- **Point-5**: `(Δx, Δy, p₁, p₂, p₃)` with one-hot encoded pen states

**Drawbacks:**
1. **Lack of compression** often necessitates truncating long sequences during training
2. **Many design choices** required for input normalization, sequence initiation/termination
3. **Mixture density networks (MDNs)** introduce challenges:
   - Hyperparameter tuning for mixture count and loss weights
   - Numerical instability and mode collapse
   - Negative loss values that are difficult to interpret across datasets

### Token Representations

Token representations treat digital ink as sequences of discrete tokens, addressing vector representation challenges:

1. ✅ **Compression** via merging algorithms like BPE effectively reduces sequence lengths
2. ✅ **No normalization** required; special `[START]` and `[END]` tokens handle sequence boundaries
3. ✅ **Cross-entropy loss** provides stable training with interpretable loss values

Existing token representations include:

- **AbsTokens**: Each pixel coordinate is a separate token (suffers from out-of-vocabulary issues)
- **RelTokens**: Each relative coordinate `(Δx, Δy)` is a token
- **TextTokens**: Represents ink as literal text using digits and separators

**Tokink** improves upon these by using **directional arrow tokens** (↑, ↓, ←, →, ↖, ↗, ↙, ↘) combined with BPE compression to learn common movement patterns like "horizontal line" or "curve" automatically from data.

## Quick Start

### Basic Usage

```python
from tokink import Tokinkizer, Ink

# Load a pretrained tokenizer
tokenizer = Tokinkizer.from_pretrained()

# Create a simple drawing (a square)
ink = Ink.from_coords([
    [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]  # One stroke forming a square
])

# Tokenize the ink
tokens = tokenizer.tokenize(ink)
print(tokens)
# ['[BOS]', '[DOWN]', '→→→→→→→→→→', '[UP]', '[DOWN]', '↑↑↑↑↑↑↑↑↑↑', ...]

# Encode to IDs
ids = tokenizer.encode(ink)
print(ids)
# [1, 4, 2847, 3, 4, 2851, 3, ...]

# Decode back to ink
decoded_ink = tokenizer.decode(ids)
decoded_ink.plot()  # Visualize the drawing
```

### Working with Ink Objects

```python
from tokink import Ink, Stroke, Point

# Method 1: From coordinate lists
ink = Ink.from_coords([
    [(0, 0), (5, 0), (5, 5)],      # First stroke
    [(10, 10), (15, 10), (15, 15)] # Second stroke
])

# Method 2: From Point and Stroke objects
ink = Ink(strokes=[
    Stroke(points=[
        Point(x=0, y=0),
        Point(x=5, y=0),
        Point(x=5, y=5)
    ]),
    Stroke(points=[
        Point(x=10, y=10),
        Point(x=15, y=10),
        Point(x=15, y=15)
    ])
])

# Method 3: Load from JSON
ink = Ink.load("path/to/ink.json")

# Method 4: Use built-in example
ink = Ink.example()

# Visualize
ink.plot()

# Save
ink.save("my_drawing.json")
ink.save_plot("my_drawing.png")
```

### Training a Custom Tokenizer

```python
from tokink import Tokinkizer, Ink

# Prepare your training data
training_inks = [
    Ink.from_coords([[(0, 0), (10, 0), (10, 10)]]),
    Ink.from_coords([[(0, 0), (5, 5), (10, 10)]]),
    # ... more ink samples
]

# Train a tokenizer with BPE
tokenizer = Tokinkizer.train(
    iter(training_inks),
    vocab_size=50_000
)

# Save for later use
tokenizer.save("my_tokenizer/")

# Load it back
tokenizer = Tokinkizer.from_pretrained("my_tokenizer/")
```

### Token Visualization

```python
from tokink import Tokinkizer, Ink

tokenizer = Tokinkizer.from_pretrained()
ink = Ink.example()

# Visualize how the ink is tokenized
# Each token will be shown in a different color
tokenizer.plot_tokens(ink)

# Or plot specific tokens
tokens = ['[BOS]', '[DOWN]', '→→→', '↑↑', '[UP]', '[EOS]']
tokenizer.plot_tokens(tokens)
```

### Advanced: Token and ID Conversion

```python
from tokink import Tokinkizer

tokenizer = Tokinkizer.from_pretrained()

# Convert individual tokens to IDs
token_id = tokenizer.token_to_id("→")
print(token_id)  # 7

# Convert IDs back to tokens
token = tokenizer.id_to_token(7)
print(token)  # '→'

# Batch conversion
tokens = ["[BOS]", "→", "↑", "[DOWN]", "[UP]", "[EOS]"]
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)  # [1, 7, 5, 4, 3, 2]

recovered_tokens = tokenizer.convert_ids_to_tokens(ids)
print(recovered_tokens)  # ['[BOS]', '→', '↑', '[DOWN]', '[UP]', '[EOS]']
```

### Vocabulary Management

```python
from tokink import Tokinkizer

# Load with reduced vocabulary size
tokenizer = Tokinkizer.from_pretrained(vocab_size=10_000)

# Check vocabulary size
print(len(tokenizer._vocab))  # 10000

# Access vocabulary
print(tokenizer._vocab)  # {'[BOS]': 1, '[EOS]': 2, ...}
```

## How It Works

Tokink uses a three-step process:

1. **Discretization**: Converts continuous ink strokes into 8-directional arrow tokens using Bresenham's line algorithm
   - `(0, 1)` → `↑` (up)
   - `(1, 0)` → `→` (right)
   - `(1, 1)` → `↗` (diagonal up-right)
   - etc.

2. **BPE Compression**: Learns to merge frequently co-occurring arrow tokens
   - `→→→→→` might become a single "horizontal line" token
   - `↗↗↗` might become a single "diagonal line" token

3. **Special Tokens**: Adds control tokens for sequence structure
   - `[BOS]` - Beginning of sequence
   - `[EOS]` - End of sequence
   - `[DOWN]` - Pen down (start of stroke)
   - `[UP]` - Pen up (end of stroke)

## API Reference

### Tokinkizer

The main tokenizer class for digital ink.

#### Factory Methods

- `Tokinkizer.from_pretrained(path=None, vocab_size=None)` - Load a pretrained tokenizer
- `Tokinkizer.train(inks, vocab_size=100_000)` - Train a new tokenizer from ink samples

#### Tokenization Methods

- `tokenize(ink)` - Convert ink to token strings
- `detokenize(tokens)` - Convert token strings back to ink
- `encode(ink)` - Convert ink to vocabulary IDs
- `decode(ids)` - Convert vocabulary IDs back to ink

#### Conversion Methods

- `token_to_id(token)` - Look up ID for a token
- `id_to_token(id)` - Look up token for an ID
- `convert_tokens_to_ids(tokens)` - Batch token-to-ID conversion
- `convert_ids_to_tokens(ids)` - Batch ID-to-token conversion

#### Visualization Methods

- `plot_tokens(source)` - Visualize tokens (accepts `Ink` or `list[str]`)

#### Persistence Methods

- `save(save_path=None)` - Save vocabulary and merges to disk

### Ink

Represents a complete digital ink drawing.

#### Factory Methods

- `Ink.from_coords(coords)` - Create from nested coordinate lists
- `Ink.load(path)` - Load from JSON file
- `Ink.example()` - Load built-in example

#### Methods

- `plot()` - Display the drawing
- `save(path=None)` - Save as JSON
- `save_plot(path=None)` - Save as image
- `to_int()` - Convert all coordinates to integers

### Stroke

Represents a single continuous stroke.

#### Factory Methods

- `Stroke.from_coords(coords)` - Create from coordinate list

#### Methods

- `to_int()` - Convert all points to integers

### Point

Represents a 2D point.

#### Factory Methods

- `Point.from_coords(coords)` - Create from (x, y) tuple or list

#### Methods

- `to_int()` - Convert to integer coordinates

## Examples

### Example 1: Handwriting Recognition Pipeline

```python
from tokink import Tokinkizer, Ink

# Load tokenizer
tokenizer = Tokinkizer.from_pretrained()

# Load handwritten text
ink = Ink.load("handwriting_sample.json")

# Tokenize for model input
ids = tokenizer.encode(ink)

# Use with your ML model
# model_output = model(ids)
```

### Example 2: Sketch Generation

```python
from tokink import Tokinkizer

tokenizer = Tokinkizer.from_pretrained()

# Generate token IDs from your model
# generated_ids = model.generate(...)

# Convert to ink drawing
generated_ink = tokenizer.decode(generated_ids)

# Visualize or save
generated_ink.plot()
generated_ink.save("generated_sketch.json")
```

### Example 3: Data Preprocessing

```python
from tokink import Tokinkizer, Ink
import json

tokenizer = Tokinkizer.from_pretrained()

# Process a dataset
dataset = []
for ink_file in ink_files:
    ink = Ink.load(ink_file)
    ids = tokenizer.encode(ink)
    dataset.append({
        "input_ids": ids,
        "length": len(ids)
    })

# Save preprocessed dataset
with open("preprocessed_data.json", "w") as f:
    json.dump(dataset, f)
```

## Special Tokens

| Token | ID | Description |
|-------|-----|-------------|
| `[BOS]` | 1 | Beginning of sequence |
| `[EOS]` | 2 | End of sequence |
| `[UP]` | 3 | Pen up (end of stroke) |
| `[DOWN]` | 4 | Pen down (start of stroke) |

## Arrow Tokens

| Token | Direction | Coordinate Delta |
|-------|-----------|------------------|
| `↑` | Up | (0, 1) |
| `↓` | Down | (0, -1) |
| `←` | Left | (-1, 0) |
| `→` | Right | (1, 0) |
| `↖` | Up-Left | (-1, 1) |
| `↗` | Up-Right | (1, 1) |
| `↙` | Down-Left | (-1, -1) |
| `↘` | Down-Right | (1, -1) |

## Requirements

- Python >= 3.12
- matplotlib >= 3.10.8
- pydantic >= 2.12.5
- scipy >= 1.17.0
- tokenizers >= 0.22.2

## Development

```bash
# Clone the repository
git clone https://github.com/yourusername/tokink.git
cd tokink

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linter
ruff check .
```

## Citation

If you use Tokink in your research, please cite:

```bibtex
@software{tokink2026,
  title={Tokink: A Tokenizer for Digital Ink},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/tokink}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

This implementation is based on research into token representations for digital ink, building upon prior work in vector and token-based representations for handwriting and sketch data.
