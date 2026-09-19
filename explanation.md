# NumPyGPT — Quick Interview Revision Guide

A concise, beginner-friendly guide to understand the code flow, data transformations, and math behind **NumPyGPT**. Use this for fast interview revision and conceptual clarity.

---

## Complete Model Flowchart

```text
Raw Text
   │
   ▼
[tokenizer.py] ──> Token IDs: (B, T)
                      │
                      ▼
[embeddings.py] ──> Token + Positional Vectors: (B, T, D)
                      │
                      ▼
[attention.py] ───> Causal Multi-Head Attention: (B, T, D)
                      │
                      ▼
[transformer.py] ─> LayerNorm + FFN + Residuals: (B, T, D)
                      │
                      ▼
[model.py] ───────> Linear LM Head ──> Logits: (B, T, V)
                      │
                      ▼
[loss.py] ────────> Softmax + Cross-Entropy Loss: scalar L
                      │
                      ▼
[backward] ───────> Gradients (dL/dθ) via Chain Rule
                      │
                      ▼
[optimizer.py] ───> Adam / SGD Parameter Updates
                      │
                      ▼
[generate.py] ────> Autoregressive Text Generation
```

---

## 1. `tokenizer.py` (Byte Pair Encoding)

### ASCII Data Flow

```text
Raw Text: "hello world"
   │
   ▼  (regex split preserving spaces and punctuation)
Chunks: ["hello", " ", "world"]
   │
   ▼  (break into base characters)
Tokens: ['h', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd']
   │
   ▼  (iteratively merge most frequent adjacent pair: e.g. ('l', 'l') → 'll')
Subwords: ['he', 'll', 'o', ' ', 'world']
   │
   ▼  (map string to integer ID using token_to_id)
Token IDs: [42, 108, 15, 3, 204]
```

### Important Formula
- **Pair Selection Formula**:
  $$(a^*, b^*) = \arg\max_{(a, b)} \text{count}(a, b)$$
  *Implemented in*: `train()` — Finds the most frequent pair of adjacent symbols across all words in the corpus and fuses them into a single token.

---

### Functions in `tokenizer.py`

#### `__init__(vocab_size=500, special_tokens=None)`
- **WHAT**: Initializes vocabulary mappings, merge tables, and target vocabulary size.
- **WHY**: Sets up empty lookup tables before training or loading.
- **INPUT**: `vocab_size` (`int`, default: `500`), `special_tokens` (`list[str]`, default: `["<|unk|>", "<|endoftext|>"]`).
- **OUTPUT**: `None` (initializes object attributes).

#### `_init_base_vocab(text)`
- **WHAT**: Extracts every unique character in `text`, adds special tokens, and assigns base integer IDs.
- **WHY**: BPE must start with single characters before it can merge them into words.
- **INPUT**: `text` (`str`).
- **OUTPUT**: `None` (populates `token_to_id` and `id_to_token`).

#### `train(text, verbose=False)`
- **WHAT**: Counts adjacent symbol pairs, finds the most frequent pair, merges it into a new subword, and repeats until reaching `vocab_size`.
- **WHY**: Unsupervised vocabulary discovery — learns common subwords (e.g. `"th"`, `"ing"`, `"the"`) from raw text.
- **INPUT**: `text` (`str`), `verbose` (`bool`, default: `False`).
- **OUTPUT**: `None` (builds `merges`, `merge_ranks`, and vocabulary).
- **MAIN FORMULA**: $(a^*, b^*) = \arg\max \text{count}(a, b)$.

#### `_encode_chunk(chunk)`
- **WHAT**: Splits a word into characters and merges adjacent pairs in chronological order of their learned merge rank.
- **WHY**: Ensures new text is broken down using the exact rules discovered during training.
- **INPUT**: `chunk` (`str`, e.g. `"Before"`).
- **OUTPUT**: `list[str]` of subword tokens (e.g. `["B", "e", "fore"]`), NOT integer IDs.

#### `encode(text)`
- **WHAT**: Isolates special tokens, splits text into chunks, calls `_encode_chunk()`, and converts subword strings into integer IDs.
- **WHY**: Neural networks require integer indices to look up embedding vectors.
- **INPUT**: `text` (`str`).
- **OUTPUT**: `list[int]` (sequence of integer token IDs).

#### `decode(ids)`
- **WHAT**: Looks up each integer ID in `id_to_token` and joins the strings back together.
- **WHY**: Converts model-generated integer IDs back into readable text.
- **INPUT**: `ids` (`list[int]`).
- **OUTPUT**: `str` (exact reconstructed text).

#### `save(filepath)`
- **WHAT**: Saves `vocab_size`, `special_tokens`, `token_to_id`, and `merges` to a JSON file.
- **WHY**: Reuses a trained tokenizer without retraining on the corpus.
- **INPUT**: `filepath` (`str`).
- **OUTPUT**: `None` (writes JSON file to disk).

#### `load(filepath)`
- **WHAT**: Loads a JSON file and reconstructs the tokenizer and its priority merge ranks.
- **WHY**: Restores a saved tokenizer for inference or testing.
- **INPUT**: `filepath` (`str`).
- **OUTPUT**: `BPETokenizer` instance ready for encoding/decoding.

---

## 2. `embeddings.py` (Token & Positional Embeddings)

### ASCII Data Flow

```text
Forward Pass:
Token IDs: (B, T) ────────> W_e Lookup ────────> X_tok: (B, T, D)
                                                       │
                                                       ▼ (+)
Positions: [0..T-1] ──────> W_p Lookup ────────> X_pos: (T, D) (broadcasts over B)
                                                       │
                                                       ▼
                                                  X_out: (B, T, D)

Backward Pass:
dout: (B, T, D) ────┬───> np.sum(axis=0) ──────────────────────────> dW_p: (T_max, D)
                    └───> np.add.at(dW_e, token_ids, dout) ───────> dW_e: (V, D)
```

### Important Formulas
- **Forward Embedding Formula**:
  $$X_{\text{out}}[b, t] = W_e[X_{b, t}] + W_p[t]$$
  *Implemented in*: `forward()` — Combines token identity ($W_e$) with sequence order ($W_p$).
- **Backward Gradient Formulas**:
  $$dW_p[:T] = \sum_{b=0}^{B-1} dout[b], \quad dW_e = \sum \text{scatter-add}(dout)$$
  *Implemented in*: `backward()` — Sums gradients over batch for positions, and accumulates gradients across matching token IDs using `np.add.at`.

---

### Functions in `embeddings.py`

#### `__init__(vocab_size, max_seq_len, d_model, std=0.02)`
- **WHAT**: Initializes token weight matrix $W_e \in \mathbb{R}^{V \times D}$ and position weight matrix $W_p \in \mathbb{R}^{T_{\max} \times D}$ with Gaussian noise.
- **WHY**: Breaks symmetry so gradient descent can learn distinct continuous representations.
- **INPUT**: `vocab_size` (`int`), `max_seq_len` (`int`), `d_model` (`int`), `std` (`float`, default: `0.02`).
- **OUTPUT**: `None`.
- **MAIN FORMULA**: $W_e, W_p \sim \mathcal{N}(0, 0.02^2)$.

#### `forward(token_ids)`
- **WHAT**: Retrieves token vectors from $W_e$ and position vectors from $W_p$, adding them element-wise.
- **WHY**: Embeds what the token is and where it is in the sentence into a single vector.
- **INPUT**: `token_ids` (`np.ndarray` of shape `(B, T)`).
- **OUTPUT**: `np.ndarray` of shape `(B, T, D)`.
- **MAIN FORMULA**: $X_{\text{out}} = W_e[\text{token\_ids}] + W_p[:T]$.

#### `backward(dout)`
- **WHAT**: Computes gradients $dW_p$ by summing over batch $B$, and $dW_e$ by accumulating gradients at token ID indices using `np.add.at`.
- **WHY**: Propagates loss gradients to embedding parameters so they can be updated by the optimizer.
- **INPUT**: `dout` (`np.ndarray` of shape `(B, T, D)`).
- **OUTPUT**: `None` (stores gradients in `self.dW_e` and `self.dW_p`).
- **MAIN FORMULA**: $dW_p = \sum_B dout$, $dW_e = \text{scatter-add}(dout)$.

#### `get_params()`
- **WHAT**: Returns dictionary containing `W_e` and `W_p`.
- **WHY**: Lets the optimizer access trainable weights.
- **INPUT**: None.
- **OUTPUT**: `dict[str, np.ndarray]`.

#### `get_grads()`
- **WHAT**: Returns dictionary containing `dW_e` and `dW_p`.
- **WHY**: Lets the optimizer read parameter gradients for updates.
- **INPUT**: None.
- **OUTPUT**: `dict[str, np.ndarray]`.

---

## 3. `attention.py` (Causal Multi-Head Self-Attention)

### ASCII Data Flow

```text
X: (B, T, D)
    │
    ├──> @ W_q + b_q ──> Q: (B, h, T, d_k) ──┐
    ├──> @ W_k + b_k ──> K: (B, h, T, d_k) ──┴──> Q @ K^T / √d_k ──> Scores: (B, h, T, T)
    │                                                                       │
    │                                                                       ▼ (+)
    │                                                             Causal Mask: (T, T)
    │                                                             [-1e9 on future tokens]
    │                                                                       │
    │                                                                       ▼
    │                                                             Softmax ──> Attention Weights (A): (B, h, T, T)
    │                                                                       │
    └──> @ W_v + b_v ──> V: (B, h, T, d_k) ────────────────────────────────┴──> A @ V ──> Head Outputs: (B, h, T, d_k)
                                                                                               │
                                                                                               ▼
                                                                                    Concat Heads: (B, T, D)
                                                                                               │
                                                                                               ▼
                                                                                         @ W_o + b_o
                                                                                               │
                                                                                               ▼
                                                                                      Output (Y): (B, T, D)
```

### Important Formulas
- **Scaled Dot-Product Attention**:
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + \text{mask}\right) V$$
  *Implemented in*: `scaled_dot_product_attention()` — Measures compatibility between queries and keys, zeroes out future tokens, and computes weighted average of values.
- **Softmax Function (Numerically Stable)**:
  $$\text{softmax}(z_i) = \frac{\exp(z_i - \max(z))}{\sum_j \exp(z_j - \max(z))}$$
  *Implemented in*: `softmax()` — Converts raw scores into valid probabilities summing to $1.0$ without overflow.
- **Causal Mask Matrix**:
  $$M_{i, j} = \begin{cases} 0 & j \le i \\ -10^9 & j > i \end{cases}$$
  *Implemented in*: `causal_mask()` — Adds $-10^9$ to future positions so $\exp(-10^9) \approx 0.0$ in softmax.

---

### Functions in `attention.py`

#### `causal_mask(seq_len)`
- **WHAT**: Creates an upper-triangular matrix with $0.0$ on and below diagonal, and $-10^9$ strictly above diagonal.
- **WHY**: Blocks past tokens from attending to future tokens during training.
- **INPUT**: `seq_len` (`int`, context length $T$).
- **OUTPUT**: `np.ndarray` of shape `(T, T)`.
- **MAIN FORMULA**: $M_{i, j} = 0 \text{ if } j \le i \text{ else } -10^9$.

#### `softmax(x, axis=-1)`
- **WHAT**: Computes $\exp(x - \max(x)) / \sum \exp(x - \max(x))$ along the last axis.
- **WHY**: Turns attention scores into valid probabilities summing to $1.0$; subtracting $\max$ prevents $\text{NaN}$ overflow.
- **INPUT**: `x` (`np.ndarray`), `axis` (`int`, default: `-1`).
- **OUTPUT**: `np.ndarray` of same shape with values in $[0, 1]$.
- **MAIN FORMULA**: $\text{softmax}(z_i) = \frac{e^{z_i - \max(z)}}{\sum_j e^{z_j - \max(z)}}$.

#### `scaled_dot_product_attention(Q, K, V, mask=None)`
- **WHAT**: Calculates $Q K^T / \sqrt{d_k}$, adds causal mask, applies softmax, and multiplies by $V$.
- **WHY**: Core attention math: dynamically routes information between tokens.
- **INPUT**: `Q, K, V` (`(..., T, d_k)`), `mask` (`(T, T)` or `None`).
- **OUTPUT**: `tuple(output, A)` where `output` is `(..., T, d_k)` and `A` is `(..., T, T)`.
- **MAIN FORMULA**: $\text{softmax}(Q K^T / \sqrt{d_k} + \text{mask}) V$.

#### `MultiHeadAttention.__init__(d_model, num_heads, std=0.02)`
- **WHAT**: Initializes projection weights ($W_q, W_k, W_v, W_o \in \mathbb{R}^{D \times D}$) and biases ($b_q, b_k, b_v, b_o \in \mathbb{R}^D$).
- **WHY**: Projects input $X$ into $h$ independent attention subspaces.
- **INPUT**: `d_model` (`int`), `num_heads` (`int`), `std` (`float`, default: `0.02`).
- **OUTPUT**: `None`.
- **MAIN FORMULA**: $d_k = d_{model} / num\_heads$.

#### `MultiHeadAttention.forward(X)`
- **WHAT**: Projects $X$ to $Q, K, V$, splits into $h$ heads, runs causal attention, concatenates heads, and applies output projection $W_o$.
- **WHY**: Computes contextual representations where each token attends to relevant past tokens across multiple heads.
- **INPUT**: `X` (`np.ndarray` of shape `(B, T, D)`).
- **OUTPUT**: `np.ndarray` of shape `(B, T, D)`.
- **MAIN FORMULA**: $Y = \text{Concat}(head_1, \dots, head_h) W_o + b_o$.

#### `MultiHeadAttention.backward(dout)`
- **WHAT**: Computes gradients for all 8 parameters ($dW_q, db_q, dW_k, db_k, dW_v, db_v, dW_o, db_o$) and input gradient $dX$ using the chain rule.
- **WHY**: Allows backpropagation to flow backwards through attention to previous layers/embeddings.
- **INPUT**: `dout` (`np.ndarray` of shape `(B, T, D)`).
- **OUTPUT**: `dX` (`np.ndarray` of shape `(B, T, D)`).
- **MAIN FORMULAS**:
  - $dV = A^T dout_{\text{heads}}, \quad dA = dout_{\text{heads}} V^T$
  - $dS = A \odot (dA - \sum (dA \odot A))$
  - $dQ = \frac{dS}{\sqrt{d_k}} K, \quad dK = \left(\frac{dS}{\sqrt{d_k}}\right)^T Q$
  - $dX = dQ W_q^T + dK W_k^T + dV W_v^T$

#### `MultiHeadAttention.get_params()`
- **WHAT**: Returns dictionary with all 8 parameter arrays ($W_q, b_q, W_k, b_k, W_v, b_v, W_o, b_o$).
- **WHY**: Exposes parameters to the optimizer.
- **INPUT**: None.
- **OUTPUT**: `dict[str, np.ndarray]`.

#### `MultiHeadAttention.get_grads()`
- **WHAT**: Returns dictionary with all 8 gradient arrays.
- **WHY**: Exposes gradients to the optimizer for weight updates.
- **INPUT**: None.
- **OUTPUT**: `dict[str, np.ndarray]`.

---

## 4. Upcoming Roadmap & Code Flow Preview

Here is how the upcoming phases connect to what we have built:

```text
[embeddings.py] ──> X: (B, T, D)
                       │
                       ▼
┌───────────────────────────────────────────────────────────┐
│ [transformer.py] (Phase 4 & 5)                            │
│                                                           │
│   X ───────────────────────────────────────────┐          │
│   │                                            │ (skip)   │
│   ▼                                            ▼          │
│ LayerNorm ──> [attention.py] ──> (+) ──────────────────┐  │
│                                   │                    │  │
│                                   ▼                    ▼  │
│                               LayerNorm ──> FFN ──> (+)   │
│                                                      │    │
└──────────────────────────────────────────────────────┼────┘
                                                       ▼
                                             Final LayerNorm
                                                       │
                                                       ▼
                                         [model.py] LM Head (W_vocab)
                                                       │
                                                       ▼
                                             Logits: (B, T, V)
                                                       │
                                                       ▼
                                    [loss.py] Softmax + Cross-Entropy
                                                       │
                                                       ▼
                                            [optimizer.py] Adam/SGD
```

- **`transformer.py`** (Phase 4):
  - `LayerNorm`: $y = \gamma \left(\frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}\right) + \beta$ (normalizes activations).
  - `GELU`: Non-linear activation $\approx 0.5x(1 + \tanh(\dots))$.
  - `FeedForward`: Position-wise two-layer MLP ($D \to 4D \to D$).
  - `TransformerBlock`: Combines Pre-LayerNorm, MultiHeadAttention, FFN, and residual skip connections.
- **`model.py`** (Phase 5):
  - Stacks $L$ Transformer blocks, final LayerNorm, and linear LM head to output vocabulary logits $(B, T, V)$.
- **`loss.py`** (Phase 6):
  - Computes cross-entropy loss $L = -\log(p_{\text{target}})$ and initial backward gradient $P - Y$.
- **`optimizer.py`** (Phase 7):
  - Updates weights using SGD ($\theta \leftarrow \theta - \eta \nabla\theta$) and Adam (bias-corrected momentum).
- **`train.py`** (Phase 7/8):
  - Next-token training loop on Tiny Shakespeare.
- **`generate.py`** (Phase 8):
  - Autoregressive text generation with temperature and top-k sampling.
