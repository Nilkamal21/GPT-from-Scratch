# NumPyGPT — Interview Revision Guide

A concise, beginner-friendly revision guide for **NumPyGPT**. Every implemented phase follows the exact same 4-part structure: **Flowchart**, **Functions**, **Tiny Example**, and **Interview Key Point**.

---

## Current System Pipeline (Phases 1–3)

```text
Raw Text
   ↓
[Phase 1: tokenizer.py]   ──> Token IDs: (B, T)
   ↓
[Phase 2: embeddings.py]  ──> Token + Positional Embeddings: (B, T, D)
   ↓
[Phase 3: attention.py]   ──> Causal Multi-Head Attention: (B, T, D)
```

*(Subsequent phases: `transformer.py`, `model.py`, `loss.py`, `optimizer.py`, `train.py`, and `generate.py` will be added as each phase is built and tested.)*

---

## Phase 1: `tokenizer.py` (Byte Pair Encoding)

### 1. Flowchart

```text
Raw Text
   ↓
Regex Chunks (preserves words, spaces, punctuation)
   ↓
Character Base Symbols
   ↓
Count Adjacent Pairs
   ↓
Find Most Frequent Pair (argmax)
   ↓
Merge Pair into New Subword
   ↓
Repeat until Target Vocabulary Size
   ↓
token_to_id Lookup
   ↓
Integer Token IDs
```

### 2. Functions

#### `__init__(vocab_size=500, special_tokens=None)`
- **WHAT**: Initializes empty vocabulary and merge tables.
- **WHY**: Sets up lookup dictionaries before training or loading.
- **INPUT**: `vocab_size` (`int`, default: `500`), `special_tokens` (`list[str]`, default: `["<|unk|>", "<|endoftext|>"]`).
- **OUTPUT**: `None`.

#### `_init_base_vocab(text)`
- **WHAT**: Finds all unique characters in `text` and registers special tokens.
- **WHY**: BPE must start from atomic single characters before learning merges.
- **INPUT**: `text` (`str`).
- **OUTPUT**: `None` (populates `token_to_id` and `id_to_token`).

#### `train(text, verbose=False)`
- **WHAT**: Iteratively finds the most frequent adjacent symbol pair and merges it into a new subword token.
- **WHY**: Discovers common subwords without human supervision.
- **INPUT**: `text` (`str`), `verbose` (`bool`).
- **OUTPUT**: `None` (builds `merges` and vocabulary up to `vocab_size`).
- **MAIN FORMULA**: Implements $(a^*, b^*) = \arg\max \text{count}(a, b)$.

#### `_encode_chunk(chunk)`
- **WHAT**: Splits a word into characters and applies learned merges by priority rank.
- **WHY**: Ensures test-time words are merged using the exact rules learned during training.
- **INPUT**: `chunk` (`str`, e.g. `"Before"`).
- **OUTPUT**: `list[str]` of subword tokens (e.g. `["B", "e", "fore"]`).

#### `encode(text)`
- **WHAT**: Splits text into chunks, applies BPE, and maps subword tokens to integer IDs.
- **WHY**: Neural networks require integer indices to access embedding tables.
- **INPUT**: `text` (`str`).
- **OUTPUT**: `list[int]` of token IDs.

#### `decode(ids)`
- **WHAT**: Converts integer IDs back into subword strings and joins them.
- **WHY**: Reconstructs readable text from model-generated token IDs.
- **INPUT**: `ids` (`list[int]`).
- **OUTPUT**: `str` (exact reconstructed text).

#### `save(filepath)`
- **WHAT**: Saves vocabulary and merges to a JSON file.
- **WHY**: Allows the trained tokenizer to be reused without retraining.
- **INPUT**: `filepath` (`str`).
- **OUTPUT**: `None`.

#### `load(filepath)`
- **WHAT**: Loads vocabulary and merge rules from a JSON file.
- **WHY**: Restores a trained tokenizer for inference.
- **INPUT**: `filepath` (`str`).
- **OUTPUT**: `BPETokenizer` instance.

### 3. Tiny Example

```text
Text:
"low lower lowest"
   ↓ (regex split + initial characters)
['l', 'o', 'w', ' ', 'l', 'o', 'w', 'e', 'r', ' ', 'l', 'o', 'w', 'e', 's', 't']
   ↓ (BPE learned merges)
['low', ' ', 'low', 'er', ' ', 'low', 'est']
   ↓ (token_to_id mapping: 'low'→10, ' '→2, 'er'→25, 'est'→40)
[10, 2, 10, 25, 2, 10, 40]
```

### 4. Interview Key Point
> BPE starts with individual characters and iteratively merges the most frequent adjacent pair ($(a^*, b^*) = \arg\max \text{count}(a, b)$), producing compact sequences while eliminating out-of-vocabulary failures.

---

## Phase 2: `embeddings.py` (Token & Positional Embeddings)

### 1. Flowchart

```text
Token IDs: (B, T) ────────> W_e Lookup ────────> X_tok: (B, T, D)
                                                       ↓ (+)
Positions: [0..T-1] ──────> W_p Lookup ────────> X_pos: (T, D) (broadcasts over B)
                                                       ↓
                                                  X_out: (B, T, D)

Backward Pass:
dout: (B, T, D) ────┬───> np.sum(axis=0) ──────────────────────────> dW_p: (T_max, D)
                    └───> np.add.at(dW_e, token_ids, dout) ───────> dW_e: (V, D)
```

### 2. Functions

#### `__init__(vocab_size, max_seq_len, d_model, std=0.02)`
- **WHAT**: Initializes token weights $W_e \in \mathbb{R}^{V \times D}$ and positional weights $W_p \in \mathbb{R}^{T_{\max} \times D}$ with small Gaussian noise.
- **WHY**: Breaks symmetry so gradient descent can learn distinct representations.
- **INPUT**: `vocab_size` (`int`), `max_seq_len` (`int`), `d_model` (`int`), `std` (`float`).
- **OUTPUT**: `None`.
- **MAIN FORMULA**: Implements $W_e, W_p \sim \mathcal{N}(0, 0.02^2)$.

#### `forward(token_ids)`
- **WHAT**: Retrieves token vectors from $W_e$ and positional vectors from $W_p$, adding them element-wise.
- **WHY**: Injects what the token is and where it is in the sentence into a single dense vector.
- **INPUT**: `token_ids` (`np.ndarray` of shape `(B, T)`).
- **OUTPUT**: `np.ndarray` of shape `(B, T, D)`.
- **MAIN FORMULA**: Implements $X_{\text{out}} = W_e[\text{token\_ids}] + W_p[:T]$.

#### `backward(dout)`
- **WHAT**: Sums gradients across batch for $dW_p$, and accumulates gradients for $dW_e$ using `np.add.at`.
- **WHY**: Propagates loss gradients to embedding parameters; `np.add.at` safely handles duplicate token IDs.
- **INPUT**: `dout` (`np.ndarray` of shape `(B, T, D)`).
- **OUTPUT**: `None` (stores `dW_e` and `dW_p`).
- **MAIN FORMULA**: Implements $dW_p = \sum_B dout$ and $dW_e = \text{scatter-add}(dout)$.

#### `get_params()`
- **WHAT**: Returns dictionary with `W_e` and `W_p`.
- **WHY**: Exposes weights to the optimizer.
- **INPUT**: None.
- **OUTPUT**: `dict[str, np.ndarray]`.

#### `get_grads()`
- **WHAT**: Returns dictionary with `dW_e` and `dW_p`.
- **WHY**: Exposes gradients to the optimizer.
- **INPUT**: None.
- **OUTPUT**: `dict[str, np.ndarray]`.

### 3. Tiny Example

```text
Token IDs:
[10, 25]  (Sequence length T = 2)
   ↓
Token Lookup W_e[token_ids]:
[[0.40, -0.10],
 [0.85,  0.20]]
   ↓
Position Lookup W_p[0..1]:
[[0.05,  0.10],
 [0.12, -0.05]]
   ↓
Element-wise Addition (X_tok + X_pos):
[[0.45,  0.00],
 [0.97,  0.15]]
```

### 4. Interview Key Point
> We add learned positional embeddings to token embeddings ($X = W_e[X] + W_p[:T]$) because self-attention is permutation-equivariant and has no inherent sense of word order.

---

## Phase 3: `attention.py` (Causal Multi-Head Self-Attention)

### 1. Flowchart

```text
X: (B, T, D)
    ↓
Linear Projections: Q = XW_q + b_q, K = XW_k + b_k, V = XW_v + b_v
    ↓
Split into h Heads: (B, h, T, d_k)
    ↓
QKᵀ / √d_k (Raw Attention Scores): (B, h, T, T)
    ↓
Add Causal Mask (-1e9 on future tokens j > i)
    ↓
Softmax (Probabilities sum to 1.0 along rows)
    ↓
Attention Weights (A) × V
    ↓
Concatenate Heads: (B, T, D)
    ↓
Output Projection: Y = Concat @ W_o + b_o ──> (B, T, D)
```

### 2. Functions

#### `causal_mask(seq_len)`
- **WHAT**: Generates an upper-triangular matrix with $0.0$ on/below diagonal and $-10^9$ above diagonal.
- **WHY**: Blocks past tokens from attending to future tokens during next-token prediction.
- **INPUT**: `seq_len` (`int`).
- **OUTPUT**: `np.ndarray` of shape `(T, T)`.
- **MAIN FORMULA**: Implements $M_{i, j} = 0 \text{ if } j \le i \text{ else } -10^9$.

#### `softmax(x, axis=-1)`
- **WHAT**: Computes $\exp(x - \max(x)) / \sum \exp(x - \max(x))$ along the last axis.
- **WHY**: Converts raw scores into valid probabilities summing to $1.0$; subtracting $\max$ prevents $\text{NaN}$ overflow.
- **INPUT**: `x` (`np.ndarray`), `axis` (`int`, default: `-1`).
- **OUTPUT**: `np.ndarray` of probabilities in $[0, 1]$.
- **MAIN FORMULA**: Implements $\text{softmax}(z_i) = \frac{\exp(z_i - \max(z))}{\sum_j \exp(z_j - \max(z))}$.

#### `scaled_dot_product_attention(Q, K, V, mask=None)`
- **WHAT**: Computes dot-product scores, scales by $\sqrt{d_k}$, adds causal mask, applies softmax, and multiplies by $V$.
- **WHY**: Core attention mechanism that routes information between tokens.
- **INPUT**: `Q, K, V` (`(..., T, d_k)`), `mask` (`(T, T)` or `None`).
- **OUTPUT**: `tuple(output, A)` where `output` is `(..., T, d_k)` and `A` is `(..., T, T)`.
- **MAIN FORMULA**: Implements $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + \text{mask}\right)V$.

#### `MultiHeadAttention.__init__(d_model, num_heads, std=0.02)`
- **WHAT**: Validates $d_{model} \pmod{num\_heads} == 0$ and initializes $W_q, W_k, W_v, W_o \in \mathbb{R}^{D \times D}$ and biases.
- **WHY**: Sets up parallel subspace projection parameters.
- **INPUT**: `d_model` (`int`), `num_heads` (`int`), `std` (`float`).
- **OUTPUT**: `None`.
- **MAIN FORMULA**: Implements $d_k = d_{model} / num\_heads$.

#### `MultiHeadAttention.forward(X)`
- **WHAT**: Projects $X$ to $Q, K, V$, splits into $h$ heads, runs causal attention, concatenates heads, and applies $W_o$.
- **WHY**: Allows tokens to attend to previous tokens across multiple representation subspaces.
- **INPUT**: `X` (`np.ndarray` of shape `(B, T, D)`).
- **OUTPUT**: `np.ndarray` of shape `(B, T, D)`.
- **MAIN FORMULA**: Implements $Y = \text{Concat}(head_1, \dots, head_h) W_o + b_o$.

#### `MultiHeadAttention.backward(dout)`
- **WHAT**: Computes analytical gradients for all 8 projection parameters ($dW_q, db_q, dW_k, db_k, dW_v, db_v, dW_o, db_o$) and input gradient $dX$.
- **WHY**: Propagates loss gradients backward through the attention mechanism using the chain rule.
- **INPUT**: `dout` (`np.ndarray` of shape `(B, T, D)`).
- **OUTPUT**: `dX` (`np.ndarray` of shape `(B, T, D)`).
- **MAIN FORMULAS**:
  - $dV = A^T dout_{\text{heads}}, \quad dA = dout_{\text{heads}} V^T$
  - $dS = A \odot (dA - \sum (dA \odot A))$
  - $dQ = \frac{dS}{\sqrt{d_k}} K, \quad dK = \left(\frac{dS}{\sqrt{d_k}}\right)^T Q$
  - $dX = dQ W_q^T + dK W_k^T + dV W_v^T$

#### `MultiHeadAttention.get_params()` / `get_grads()`
- **WHAT**: Returns dictionaries of weight parameters and their computed gradients.
- **WHY**: Exposes weights and gradients to the optimizer.
- **INPUT**: None.
- **OUTPUT**: `dict[str, np.ndarray]`.

### 3. Tiny Example

```text
Input X:
[[0.10, 0.20],
 [0.30, 0.40]]
   ↓
Q = XW_q + b_q,  K = XW_k + b_k,  V = XW_v + b_v
Q = [[0.20, 0.50],   K = [[0.10, 0.40],   V = [[0.30, 0.20],
     [0.60, 0.80]]        [0.50, 0.70]]        [0.90, 0.60]]
   ↓
QKᵀ / √d_k (Raw Scores):
[[0.31, 0.42],
 [0.55, 0.91]]
   ↓
Causal Mask (Upper triangle set to -∞):
[[0.31, -∞],
 [0.55, 0.91]]
   ↓
Softmax (Rows sum to 1.00):
[[1.00, 0.00],
 [0.41, 0.59]]
   ↓
Attention Weights × V:
[[0.30, 0.20],
 [0.65, 0.44]]
   ↓
Output Projection @ W_o + b_o:
[[0.25, 0.18],
 [0.58, 0.39]]
```

### 4. Interview Key Point
> We divide $QK^T$ by $\sqrt{d_k}$ to prevent dot products from exploding in high dimensions (which causes vanishing gradients in softmax), and we add $-10^9$ to future positions so their attention probabilities become strictly $0.0$.
