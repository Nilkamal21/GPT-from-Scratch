# NumPyGPT — Decoder-Only Transformer Language Model From Scratch

A genuine GPT-style decoder-only Transformer language model built completely from scratch using **pure Python and NumPy**.

No PyTorch, TensorFlow, JAX, Keras, Hugging Face, or automatic differentiation frameworks.

---

## Project Structure

```text
numpy_gpt/
│
├── data/
│   └── input.txt          # Tiny Shakespeare corpus (1.11M characters)
│
├── tokenizer.py           # From-scratch Byte Pair Encoding (BPE) tokenizer
├── embeddings.py          # (Phase 2) Token embeddings & positional encodings
├── attention.py           # (Phase 3) Causal self-attention & multi-head attention
├── transformer.py         # (Phase 4) LayerNorm, GELU, FFN & Transformer blocks
├── model.py               # (Phase 5) Complete GPT assembly
├── loss.py                # (Phase 6) Softmax, Cross-Entropy & manual backpropagation
├── optimizer.py           # (Phase 7) Custom SGD & Adam optimizers
├── train.py               # (Phase 7/8) Training loop & checkpointing
├── generate.py            # (Phase 8) Autoregressive text generation
│
├── tests/
│   └── test_components.py # Verification tests for every component
│
├── explanation.md         # Step-by-step conceptual guide (What, Why, Output)
└── README.md
```

---

## Phase 1 Completed: Dataset & BPE Tokenizer

- **Corpus Ingestion**: Tiny Shakespeare dataset (1,115,394 characters, 65 unique base characters).
- **Custom BPE Tokenizer** (`tokenizer.py`):
  - Frequency-weighted symbol pair counting.
  - Greedy iterative merges to induce subword vocabulary.
  - Lossless encoding and decoding ($\text{decode}(\text{encode}(x)) == x$).
  - Special token support (`<|unk|>`, `<|endoftext|>`).
  - JSON serialization (`save` / `load`).

---

## Quickstart & Testing

### Prerequisites
- Python 3.10+
- NumPy
- Pytest

```bash
pip install numpy pytest
```

### Run Tests
```bash
python3 -m pytest tests/test_components.py -v
```

### Quick Verification in Python
```python
from tokenizer import BPETokenizer

# Load dataset
with open("data/input.txt") as f:
    text = f.read(50000)

# Train tokenizer
tokenizer = BPETokenizer(vocab_size=250)
tokenizer.train(text, verbose=True)

# Test encoding and decoding
sample = "First Citizen:\nBefore we proceed any further, hear me speak."
encoded = tokenizer.encode(sample)
decoded = tokenizer.decode(encoded)

print("Original:", sample)
print("Encoded IDs:", encoded)
print("Decoded:", decoded)
assert decoded == sample
```

For detailed function-by-function explanations, see [`explanation.md`](explanation.md).