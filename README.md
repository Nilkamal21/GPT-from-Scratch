# ForgeGPT
A Decoder-Only Transformer Language Model Built from Scratch using Python + NumPy.

---

## 1. Project Overview

ForgeGPT is a complete, educational implementation of a decoder-only Transformer language model (GPT-style architecture) built entirely from scratch using **pure Python and NumPy**. 

Every core component—including the Byte Pair Encoding (BPE) tokenizer, token and positional embeddings, multi-head causal self-attention, layer normalization, GELU activations, feed-forward networks, residual connections, analytical backward passes, cross-entropy loss, SGD/Adam optimizers, training loops, evaluation metrics (loss and perplexity), autoregressive generation, and an interactive CLI—was implemented manually without relying on deep learning frameworks such as PyTorch, TensorFlow, JAX, or Hugging Face.

---

## 2. Key Features

- **Byte Pair Encoding (BPE) Tokenizer**: Frequency-based pair counting, greedy merge induction, lossless encode/decode, and special token support.
- **Embeddings**: Learnable token and positional embedding lookup tables with analytical gradient accumulation.
- **Causal Self-Attention**: Scaled dot-product attention with strict lower-triangular causal masking to prevent future token leakage.
- **Multi-Head Attention**: Parallel projection into multiple attention heads, attention weight computation, and output projection.
- **Normalization & Activation**: Custom `LayerNorm` (with learnable $\gamma$ and $\beta$) and Gaussian Error Linear Unit (`GELU`) using the standard tanh approximation.
- **Feed-Forward Network (FFN)**: Two-layer MLP with GELU non-linearity expanding hidden dimensions by $4\times$.
- **Residual Connections**: Pre-LayerNorm residual architecture around both attention and feed-forward sublayers.
- **Decoder-Only Transformer**: Stackable `TransformerBlock` layers combined into an end-to-end `GPT` model.
- **Manual Backpropagation**: Exact analytical backward passes implemented across every sublayer, verified against finite-difference numerical gradients.
- **Cross-Entropy Loss**: Numerically stable Softmax combined with mean cross-entropy loss and direct analytical gradient calculation ($d\text{Logits} = P - Y$).
- **Optimizers**: Custom `SGD` (with optional momentum) and `Adam` (with first/second moment tracking, bias correction, and $\epsilon$ stabilization).
- **Training & Validation Evaluation**: Chronological 90/10 data splitting, parameter counting, and Monte Carlo loss estimation.
- **Perplexity Metric**: Exponentiated cross-entropy loss ($\text{PPL} = \exp(\mathcal{L})$) tracking effective vocabulary uncertainty.
- **Autoregressive Text Generation**: Inference-only generation supporting greedy decoding, temperature scaling, top-$k$ filtering, top-$p$ (nucleus) sampling, and EOS stopping.
- **Interactive Terminal Interface**: A modern CLI with persistent `>` prompt, ANSI box-drawing header, `/clear`, and `/quit` commands.

---

## 3. Architecture / Complete Flow

### Model & Generation Flow

```text
Raw Text
   ↓
BPE Tokenizer
   ↓
Token IDs: (B, T)
   ↓
Token + Positional Embeddings: (B, T, d_model)
   ↓
Transformer Blocks (x N)
├── LayerNorm
├── Causal Multi-Head Self-Attention
├── Residual Connection
├── LayerNorm
├── Feed-Forward Network + GELU
└── Residual Connection
   ↓
Final LayerNorm: (B, T, d_model)
   ↓
Linear LM Head: (B, T, vocab_size)
   ↓
Logits: (B, T, vocab_size)
   ↓
Softmax / Next Token Prediction
   ↓
Autoregressive Generation (Greedy / Temperature / Top-k / Top-p)
```

### Training Loop Flow

```text
Input Batch (X, Y)
   ↓
GPT.forward(X) ───────────────> Logits
   ↓
CrossEntropyLoss.forward(Logits, Y) ──> Scalar Loss
   ↓
CrossEntropyLoss.backward() ──> dLogits
   ↓
GPT.backward(dLogits) ────────> Layer-by-layer Gradients
   ↓
Optimizer.step(grads) ────────> In-place Parameter Updates
```

---

## 4. Project Structure

```text
/workspaces/GPT-from-Scratch/
├── data/
│   └── input.txt             # Tiny Shakespeare dataset (80,000 chars used)
├── tokenizer.py              # Byte Pair Encoding tokenizer
├── embeddings.py             # Token & Positional embeddings
├── attention.py              # Causal self-attention & Multi-Head Attention
├── transformer.py            # LayerNorm, GELU, FFN, & TransformerBlock
├── model.py                  # Complete GPT assembly & LM head
├── loss.py                   # Softmax & CrossEntropyLoss
├── optimizer.py              # SGD & Adam optimizers
├── train.py                  # Training loop & batch sampling
├── train_eval.py             # Real training & validation evaluation experiment
├── evaluate.py               # Evaluation framework (loss, PPL, param count)
├── generate.py               # Autoregressive generation (top-k/top-p/temperature)
├── chat.py                   # Terminal CLI interface
├── compare_optimizers.py     # Deterministic SGD vs. Adam benchmark
├── tests/
│   └── test_components.py    # 56 unit & integration tests
├── EXPLAINATION.md           # Step-by-step conceptual guide & derivations
├── REPORT.md                 # Results-focused experimental report
└── README.md                 # Project documentation
```

### File-by-File Implementation Details

| File | Purpose & Implementation Details |
| :--- | :--- |
| `tokenizer.py` | Implements `BPETokenizer` from scratch, including frequency-based pair counting, iterative merge induction, lossless encode/decode, special tokens (`<|endoftext|>`, `<|unk|>`), and JSON save/load. |
| `embeddings.py` | Implements `Embedding` and `PositionalEmbedding` lookup layers, supporting forward summation of token and position representations and analytical gradient accumulation during backpropagation. |
| `attention.py` | Implements `scaled_dot_product_attention` with strict causal upper-triangular masking, and `MultiHeadAttention` with linear query/key/value projections and analytical backward passes. |
| `transformer.py` | Implements `LayerNorm` (mean/variance normalization with learnable $\gamma, \beta$), `GELU` activation, two-layer `FeedForward` network, and `TransformerBlock` combining pre-LN residuals. |
| `model.py` | Assembles the full `GPT` language model by stacking $N$ `TransformerBlock` layers, a final `LayerNorm`, and a linear `LMHead` projecting to vocabulary logits; manages parameter and gradient dictionaries. |
| `loss.py` | Implements numerically stable `softmax` (subtracting max logit) and `CrossEntropyLoss`, calculating mean scalar loss across batches and sequences with analytical gradient $d\text{Logits} = (P - Y) / (B \cdot T)$. |
| `optimizer.py` | Implements `SGD` (with optional momentum buffer) and `Adam` (with first and second moment moving averages, timestep bias correction, and $\epsilon$ numerical stabilization). |
| `train.py` | Implements dataset batch sampling (`get_batch`) with 1-token shifted targets, single-step forward/backward optimization (`train_step`), and full training loops (`train`). |
| `train_eval.py` | Runs real training and validation experiments on Tiny Shakespeare, tracking loss and perplexity across iterations, and conducting a post-training generation check. |
| `evaluate.py` | Provides inference-only evaluation utilities: `train_val_split` (90/10 split), `count_parameters`, `compute_perplexity` ($\exp(\mathcal{L})$), and `estimate_loss`. |
| `generate.py` | Implements autoregressive text generation with greedy selection, temperature scaling, top-$k$ filtering, top-$p$ (nucleus) sampling, context window cropping, and EOS stopping. |
| `chat.py` | Implements a modern interactive CLI terminal interface with ANSI box-drawing header, persistent `>` prompt, `/clear` screen reset, and `/quit` exit commands. |
| `compare_optimizers.py` | Implements a benchmark comparing `SGD` and `Adam` under identical initial parameters, seeds, and training conditions. |
| `tests/test_components.py` | Comprehensive test suite containing 56 unit and integration tests covering every module, shape, numerical gradient check, and pipeline behavior. |

---

## 5. Technology Stack

- **Language**: Python 3.10+
- **Core Math / Array Operations**: NumPy
- **Testing & Verification**: pytest
- **Tokenizer**: Custom Byte Pair Encoding (BPE) implemented from scratch
- **Frameworks Used**: **None** (No PyTorch, No TensorFlow, No JAX, No Keras, No Hugging Face)
- **Pretrained Weights**: **None** (Trained completely from scratch from random initialization)

---

## 6. Model Configuration

The measured model architecture configured and trained in this repository:

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Vocabulary Size ($V$)** | 150 | Subwords induced by BPE on training corpus |
| **Transformer Blocks ($N$)** | 2 | Stacked decoder layers (`num_layers`) |
| **Attention Heads ($h$)** | 4 | Parallel attention heads per block |
| **Embedding Dimension ($d_{\text{model}}$)** | 32 | Hidden feature representation size |
| **Head Dimension ($d_k$)** | 8 | $d_{\text{model}} / h = 32 / 4$ |
| **Feed-Forward Dimension ($d_{\text{ff}}$)** | 128 | $4 \times d_{\text{model}}$ |
| **Context Length ($T_{\max}$)** | 32 | Maximum sequence length / attention window |
| **Trainable Parameters** | 36,246 | Total scalar weights and biases |

---

## 7. Training & Evaluation

- **Dataset**: `data/input.txt` (Tiny Shakespeare, first 80,000 characters).
- **Tokenized Size**: 56,310 total tokens.
- **Split Ratio**: Deterministic 90/10 split (`seed = 42`):
  - **Training split**: 50,679 tokens
  - **Validation split**: 5,631 tokens (strictly held out; never participates in backprop or parameter updates)
- **Training Budget**: 100 iterations, batch size $B = 4$, context length $T = 32$, learning rate $\alpha = 3 \times 10^{-3}$.

### Phase 9B Training Trajectory (Adam)

| Step | Train Loss | Val Loss | Train PPL | Val PPL |
| :---: | :---: | :---: | :---: | :---: |
| **0** | 5.0238 | 5.0227 | 151.99 | 151.82 |
| **20** | 4.1689 | 4.2343 | 64.64 | 69.01 |
| **40** | 3.8304 | 3.9234 | 46.08 | 50.57 |
| **60** | 3.6856 | 3.7751 | 39.87 | 43.60 |
| **80** | 3.5684 | 3.6455 | 35.46 | 38.30 |
| **100** | 3.4992 | 3.5495 | 33.09 | 34.80 |

---

## 8. Results

### SGD vs. Adam Benchmark

Both optimizers were evaluated under strictly identical conditions (same initial weights restored before each run, identical seeds, batch size 4, sequence length 32, 100 steps, learning rate $3 \times 10^{-3}$):

| Optimizer | Initial Loss | Final Train Loss | Final Val Loss | Final Train PPL | Final Val PPL |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **SGD** | 5.0238 | 4.5827 | 4.6178 | 97.78 | 101.27 |
| **Adam** | 5.0238 | 3.5459 | 3.6320 | 34.67 | 37.79 |

- **Adam**: **29.4% training loss reduction** (loss dropped from 5.0238 to 3.5459; validation perplexity dropped from 151.82 to 37.79).
- **SGD**: **8.8% training loss reduction** (loss dropped from 5.0238 to 4.5827; validation perplexity dropped from 151.82 to 101.27).
- **Verification**: **56/56 tests passing** across the complete test suite.

*Note: The optimizer comparison was performed under identical experimental conditions on this specific architecture and dataset.*

---

## 9. Text Generation

The text generation pipeline is **strictly inference-only** (no loss computation, no backward propagation, and no parameter updates):

```text
Prompt
   ↓
BPE Encode ─────────────> Token IDs: (1, T)
   ↓
GPT Forward Pass ───────> Logits: (1, T, V)
   ↓
Last-Position Logits ───> Logits[-1, :]
   ↓
Temperature Scaling ────> Logits / T
   ↓
Top-k / Top-p Filtering
   ↓
Softmax & Sample ───────> Next Token ID
   ↓
Append Token ───────────> (1, T + 1)
   ↓ (Repeat until max_new_tokens or EOS)
BPE Decode ─────────────> Generated Text
```

### Example Generation Output

- **Prompt**: `'ROMEO:'`
- **Settings**: `temperature = 0.8`, `top_k = 10`, `max_new_tokens = 25`
- **Output**:
  ```text
  'ROMEO:\nWar,y s pa f c\n\nEM\nAuts ca'
  ```

*Qualitative Assessment*: The model has learned whitespace formatting, character casing, and subword clustering from Shakespeare text within 100 steps on 36k parameters. It is an illustrative proof-of-concept, not a claim of grammatical English fluency.

---

## 10. Terminal Interface

ForgeGPT includes an interactive terminal chat interface in `chat.py` styled with ANSI box drawing:

```bash
python3 chat.py
```

### Example Session

```text
╭──────────────────────────────────────────────────────────────╮
│   ___                     ___ ___ _____                      │
│  | __|__ _ _ __ _ ___   / __| _ \_   _|                      │
│  | _/ _ \ '_/ _` / -_) | (_ |  _/ | |                        │
│  |_|\___/_| \__, \___|  \___|_|   |_|                        │
│                                                              │
│  Decoder-Only Transformer from Scratch                       │
│                                                              │
│  Commands: /clear to reset screen  •  /quit to exit          │
╰──────────────────────────────────────────────────────────────╯

> ROMEO:

ForgeGPT
War,y s pa f c

EM
Auts ca

────────────────────────────────────────────────────────────────

> /clear
[Screen clears and redraws header]

> /quit

Exiting ForgeGPT. Goodbye!
```

---

## 11. How to Run

### 1. Clone and Open Project
```bash
git clone https://github.com/Nilkamal21/GPT-from-Scratch.git
cd GPT-from-Scratch
```

### 2. Install Dependencies
```bash
pip install numpy pytest
```

### 3. Run the Test Suite
```bash
python3 -m pytest tests/test_components.py -v
```

### 4. Run Real Training & Validation Evaluation
```bash
python3 train_eval.py
```

### 5. Run Text Generation Demo
```bash
python3 generate.py
```

### 6. Run Optimizer Comparison (SGD vs. Adam)
```bash
python3 compare_optimizers.py
```

### 7. Launch Interactive Terminal Interface
```bash
python3 chat.py
```

---

## 12. Testing & Verification

The repository maintains an automated test suite in `tests/test_components.py`:

- **Total Tests**: **56**
- **Passed**: **56**
- **Failed**: **0**
- **Runtime**: ~2.2–2.5 seconds

### Test Coverage Breakdown:
- **Tokenizer (`tokenizer.py`)**: Subword growth, lossless roundtrips, edge cases, special tokens (`<|endoftext|>`, `<|unk|>`), unknown characters, and JSON serialization.
- **Embeddings (`embeddings.py`)**: Shape correctness, position sensitivity, determinism, context limits, and numerical gradient verification.
- **Attention (`attention.py`)**: Causal mask structure, scaled dot-product causality, multi-head projection shapes, zero future-token leakage, and numerical gradient verification.
- **Transformer Blocks (`transformer.py`)**: LayerNorm properties/gradients, GELU properties/gradients, Feed-Forward shapes/gradients, and TransformerBlock residual gradient flow.
- **Model Assembly (`model.py`)**: GPT output shapes, block stacking, final LayerNorm and LM Head, parameter/gradient consistency, and numerical gradient verification.
- **Loss & Backpropagation (`loss.py`)**: Softmax probabilities/shapes, numerical stability (large logits), cross-entropy values/shapes, numerical gradient verification, and GPT backward integration with loss.
- **Optimizers (`optimizer.py`)**: SGD update formulas, Adam first/second moment updates with bias correction, and parameter-gradient key matching.
- **Training Loop (`train.py`)**: `get_batch` shapes and 1-token shift, complete training step execution, and mini-training loop loss decrease.
- **Text Generation (`generate.py`)**: Top-$k$ filtering, top-$p$ filtering, token output shapes, greedy determinism, EOS early stopping, context window cropping, and argument validation.
- **Evaluation Framework (`evaluate.py`)**: Chronological dataset split boundaries, parameter counting, perplexity calculation, inference safety (parameters bitwise unchanged), and evaluation metric keys.
- **Real Training & Benchmark (`train_eval.py`, `compare_optimizers.py`)**: Experiment structure, training parameter updates with validation isolation, and SGD vs. Adam benchmark verification.

---

## 13. Limitations

- **Small Model Scale**: At 36,246 parameters ($d_{\text{model}} = 32, N = 2$), model capacity is limited to basic character/subword patterns.
- **Short Training Budget**: 100 steps (12,800 tokens processed) demonstrates optimization and validation convergence but is insufficient for coherent English generation.
- **Short Context Window**: 32-token context window restricts attention to local dependencies.
- **CPU-Oriented Implementation**: Pure Python/NumPy matrix operations run on CPU without GPU kernel acceleration or parallel tensor distribution.
- **Generation Quality**: Output reflects the small parameter and data budget; it does not produce fluent conversational text.

---

## 14. Future Improvements

- **Extended Training Runs**: Scale training to more iterations across the entire Shakespeare corpus or larger text datasets.
- **Larger Model Configurations**: Scale $d_{\text{model}}$, number of layers ($N$), and attention heads ($h$) to assess capacity scaling.
- **Longer Context Windows**: Expand sequence length ($T_{\max}$) to capture longer-range literary structures.
- **Key-Value (KV) Caching**: Implement KV caching during autoregressive generation to avoid redundant attention recomputation.
- **Model Checkpointing**: Add weight serialization and checkpoint saving/loading to resume long training runs.
- **Additional Evaluation Benchmarks**: Incorporate validation accuracy, cross-entropy calibration, and generation metrics.

---

## 15. Learning / Engineering Highlights

- **Manual Transformer Mathematics**: Implemented scaled dot-product attention, causal masks, layer normalization, and GELU activations directly from mathematical formulas in NumPy.
- **Analytical Backpropagation**: Derived and implemented backward passes across every sublayer, verified against finite-difference numerical gradients.
- **Optimizer Dynamics**: Implemented Adam and SGD from scratch, experimentally observing Adam's faster convergence (29.4% vs. 8.8% loss reduction) due to per-parameter adaptive scaling.
- **Inference Pipeline**: Built an autoregressive decoder supporting temperature scaling, top-$k$, and top-$p$ nucleus sampling.
- **Polished CLI Interface**: Designed an interactive terminal interface using standard ANSI escape codes.