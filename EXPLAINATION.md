# NumPyGPT — Interview Revision Guide

A concise, beginner-friendly revision guide for **NumPyGPT**. Every implemented phase follows the exact same 4-part structure: **Flowchart**, **Functions**, **Tiny Example**, and **Interview Key Point**.

---

## Complete System Pipeline (Phases 1–8)

```text
Raw Text
   ↓
[Phase 1: tokenizer.py]   ──> Token IDs: (B, T)
   ↓
[Phase 2: embeddings.py]  ──> Token + Positional Embeddings: (B, T, D)
   ↓
[Phase 3: attention.py]   ──> Causal Multi-Head Attention: (B, T, D)
   ↓
[Phase 4: transformer.py] ──> Pre-Norm LayerNorm + FFN + Residuals: (B, T, D)
   ↓
[Phase 5: model.py]       ──> Stacked Blocks + Final LayerNorm + LM Head: (B, T, V)
   ↓
[Phase 6: loss.py]        ──> Softmax + Cross-Entropy Loss + dLogits: Scalar Loss & (B, T, V)
   ↓
[Phase 7: optimizer.py & train.py] ──> SGD / Adam Updates & Full Training Loop
   ↓
[Phase 8: generate.py]    ──> Autoregressive Text Generation (Greedy / Sampling / Top-K / Top-P / EOS)
```

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
- **MAIN FORMULA**: Implements $X_{\text{out}} = W_e[X] + W_p[:T]$.

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
- **WHAT**: Validates $D \pmod h == 0$ and initializes $W_q, W_k, W_v, W_o \in \mathbb{R}^{D \times D}$ and biases.
- **WHY**: Sets up parallel subspace projection parameters.
- **INPUT**: `d_model` (`int`), `num_heads` (`int`), `std` (`float`).
- **OUTPUT**: `None`.
- **MAIN FORMULA**: Implements $d_k = D / h$.

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
  - $dV = A^T dO_{\text{head}}, \quad dA = dO_{\text{head}} V^T$
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

---

## Phase 4: `transformer.py` (LayerNorm, GELU, FFN & Residual Connections)

### 1. Flowchart

```text
                        X
                        │
                        ├────────────────────────────────────┐
                        ↓                                    │
             LayerNorm (Normalization)                       │
                        ↓                                    │
       Self-Attention (Token Interaction)                    │
             [Built in attention.py]                         │
                        ↓                                    │
       Attention Output (New Information)                    │
             [Built in attention.py]                         │
                        ↓                                    │
                        + ◄──────────────────────────────────┘
                        ↓
                 Result 1 (Residual Output)
                        │
                        ├────────────────────────────────────┐
                        ↓                                    │
             LayerNorm (Normalization)                       │
                        ↓                                    │
               Linear (Expansion)                            │
                        ↓                                    │
            GELU (Activation Function)                       │
                        ↓                                    │
               Linear (Projection)                           │
                        ↓                                    │
         FFN Output (Processed Information)                  │
                        ↓                                    │
                        + ◄──────────────────────────────────┘
                        ↓
                Transformer Output
```

#### Step-by-Step Flow:
1. **Phase 2**: Embedding layer produces input $X$.
2. **Phase 3**: `attention.py` takes normalized $X$, calculates $Q, K, V$, performs causal multi-head self-attention, and produces the Attention Output.
3. **Residual 1**: Add original $X$ to the Attention Output ($x_1 = x + \text{Attention}(\text{LN}(x))$).
4. **Second LayerNorm**: Normalize Result 1 ($\text{LN}(x_1)$).
5. **FFN**: Apply Linear $\to$ GELU $\to$ Linear to each token representation independently.
6. **Residual 2**: Add FFN output to Result 1 ($x_2 = x_1 + \text{FFN}(\text{LN}(x_1))$).
7. **Final**: This produces the Transformer block output.

#### Core Component Roles:
| Component | Role | What It Does |
| :--- | :--- | :--- |
| **Attention** | Token Interaction | Tokens communicate with each other across sequence context. |
| **FFN** | Feature Processing | Each token independently processes its own gathered information. |
| **LayerNorm** | Numerical Stability | Normalizes activations to mean 0 and variance 1 to prevent exploding/vanishing signals. |
| **Residual** | Gradient Highway | Preserves and adds previous information directly ($x + f(x)$). |
| **GELU** | Non-linear Activation | Smooth non-linearity allowing gradient flow even for small negative values. |

---

### 2. Functions

#### `LayerNorm.forward(x)`
- **WHAT**: Normalizes features across the last dimension $D$ to zero mean and unit variance, then scales by $\gamma$ and shifts by $\beta$.
- **WHY**: Keeps activations numerically stable across layers during training.
- **INPUT**: `x` (`np.ndarray` of shape `(B, T, D)`).
- **OUTPUT**: `np.ndarray` of shape `(B, T, D)`.
- **MAIN FORMULA**: Implements $\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}, \quad y = \gamma \hat{x} + \beta$.
- **CONNECTION**: Applied before Attention and before FFN in `TransformerBlock` (Pre-Norm architecture).

#### `LayerNorm.backward(dout)`
- **WHAT**: Computes analytical gradients $d\gamma$, $d\beta$, and input gradient $dx$.
- **WHY**: Propagates loss gradients backward through normalization to the previous layer.
- **INPUT**: `dout` (`np.ndarray` of shape `(B, T, D)`).
- **OUTPUT**: `dx` (`np.ndarray` of shape `(B, T, D)`).
- **MAIN FORMULA**: Implements $dx = \frac{1}{\sigma D} [D \cdot dx_{\text{hat}} - \sum dx_{\text{hat}} - \hat{x} \sum (dx_{\text{hat}} \cdot \hat{x})]$ where $\sigma = \sqrt{\sigma^2 + \epsilon}$.
- **CONNECTION**: Propagates gradients back to $X$ and residual branches.

#### `gelu(x)`
- **WHAT**: Smooth non-linear activation based on the Gaussian cumulative distribution function.
- **WHY**: Avoids the "dying ReLU" problem by allowing small negative gradients to pass through.
- **INPUT**: `x` (`np.ndarray`).
- **OUTPUT**: `np.ndarray` of same shape.
- **MAIN FORMULA**: Implements $\text{GELU}(x) \approx 0.5x (1 + \tanh(\sqrt{2/\pi}(x + 0.044715x^3)))$.
- **CONNECTION**: Activation function inside the hidden layer of `FeedForward`.

#### `gelu_backward(x, dout)`
- **WHAT**: Computes the exact analytical derivative of the GELU function multiplied by upstream gradient $dout$.
- **WHY**: Propagates gradients backward through the non-linear activation.
- **INPUT**: `x` (`np.ndarray`), `dout` (`np.ndarray`).
- **OUTPUT**: `np.ndarray` of same shape.
- **MAIN FORMULA**: Implements $dout \odot \frac{d}{dx}\text{GELU}(x)$.
- **CONNECTION**: Used inside `FeedForward.backward()`.

#### `FeedForward.forward(x)`
- **WHAT**: Projects $D \to D_{ff}$ via $W_1$, applies GELU, and projects $D_{ff} \to D$ via $W_2$.
- **WHY**: Performs position-wise non-linear feature transformation on each token.
- **INPUT**: `x` (`np.ndarray` of shape `(B, T, D)`).
- **OUTPUT**: `np.ndarray` of shape `(B, T, D)`.
- **MAIN FORMULA**: Implements $\text{FFN}(x) = W_2 \, \text{GELU}(W_1 x + b_1) + b_2$.
- **CONNECTION**: Second major sublayer inside `TransformerBlock`.

#### `FeedForward.backward(dout)`
- **WHAT**: Computes parameter gradients $dW_1, db_1, dW_2, db_2$ and input gradient $dx$.
- **WHY**: Trains the linear projection weights inside the FFN.
- **INPUT**: `dout` (`np.ndarray` of shape `(B, T, D)`).
- **OUTPUT**: `dx` (`np.ndarray` of shape `(B, T, D)`).
- **MAIN FORMULA**: Implements chain rule through linear layer 2 $\to$ GELU $\to$ linear layer 1.
- **CONNECTION**: Passes gradient to `LayerNorm` and residual add.

#### `TransformerBlock.forward(x)`
- **WHAT**: Executes Pre-Norm Attention + Residual, followed by Pre-Norm FFN + Residual.
- **WHY**: The complete core repeating decoder block of GPT.
- **INPUT**: `x` (`np.ndarray` of shape `(B, T, D)`).
- **OUTPUT**: `np.ndarray` of shape `(B, T, D)`.
- **MAIN FORMULA**: Implements $x_1 = x + \text{Attn}(\text{LN}_1(x)), \quad x_2 = x_1 + \text{FFN}(\text{LN}_2(x_1))$.
- **CONNECTION**: Takes embeddings from Phase 2/3, and will be stacked $L$ times in Phase 5 (`model.py`).

#### `TransformerBlock.backward(dout)`
- **WHAT**: Propagates gradient $dout$ through both residual branches (direct skip + sublayer path).
- **WHY**: Trains all internal attention, FFN, and normalization parameters end-to-end.
- **INPUT**: `dout` (`np.ndarray` of shape `(B, T, D)`).
- **OUTPUT**: `dx` (`np.ndarray` of shape `(B, T, D)`).
- **MAIN FORMULA**: Implements $dx_1 = dout + \text{LN}_{2,\text{back}}(\text{FFN}_{\text{back}}(dout))$, $dx = dx_1 + \text{LN}_{1,\text{back}}(\text{Attn}_{\text{back}}(dx_1))$.
- **CONNECTION**: Returns $dx$ to previous Transformer block or embeddings.

---

### 3. Tiny Example

```text
Input X:
[[ 1.00,  3.00],
 [ 2.00, -2.00]]
   ↓ (LayerNorm 1: normalize along last axis to mean 0, var 1)
LN1_out:
[[-1.00,  1.00],
 [ 1.00, -1.00]]
   ↓ (Self-Attention + Residual Add: x + Attn_out)
Result 1:
[[ 1.20,  3.10],
 [ 2.10, -1.80]]
   ↓ (LayerNorm 2: normalize Result 1)
LN2_out:
[[-1.00,  1.00],
 [ 1.00, -1.00]]
   ↓ (FFN: Linear(2→4) → GELU → Linear(4→2))
FFN_out:
[[ 0.15, -0.08],
 [-0.20,  0.12]]
   ↓ (Residual Add: Result 1 + FFN_out)
Transformer Output:
[[ 1.35,  3.02],
 [ 1.90, -1.68]]
```

---

### 4. Interview Key Point
> In a Transformer block, **Attention lets tokens communicate with each other**, while the **FFN lets each token process its new information independently**. **Residual connections ($x + f(x)$)** create a direct highway that allows gradients to flow backwards through deep layers without vanishing.

---

## Phase 5: `model.py` (Full Transformer & Complete GPT Architecture)

### 1. Flowchart

```text
Forward Pass:

Input Token IDs: (B, T)
         ↓
Embedding Layer (Token W_e + Positional W_p)
         ↓  X: (B, T, D)
┌──────────────────────────────────────────┐
│ TransformerBlock 1                       │
│  - Pre-Norm Causal Multi-Head Attention  │
│  - Residual Connection                   │
│  - Pre-Norm Feed-Forward Network (GELU)  │
│  - Residual Connection                   │
└──────────────────────────────────────────┘
         ↓  (B, T, D)
┌──────────────────────────────────────────┐
│ TransformerBlock 2                       │
└──────────────────────────────────────────┘
         ↓  (B, T, D)
        ...
         ↓  (B, T, D)
┌──────────────────────────────────────────┐
│ TransformerBlock N (num_layers)          │
└──────────────────────────────────────────┘
         ↓  X_final: (B, T, D)
Final LayerNorm (ln_f) ──> Stabilizes residual stream variance
         ↓  X_norm: (B, T, D)
LM Head (Linear Projection: W_vocab, b_vocab)
         ↓
Logits: (B, T, vocab_size)  ──> Unnormalized next-token prediction scores


Backward Pass:

Loss Gradient dlogits: (B, T, V)
         ↓
LM Head Backward:
  dW_vocab = X_norm^T @ dlogits,  db_vocab = sum(dlogits)
  dX_norm = dlogits @ W_vocab^T
         ↓  dX_norm: (B, T, D)
Final LayerNorm Backward:
  dgamma, dbeta, dX = ln_f.backward(dX_norm)
         ↓  dX: (B, T, D)
Transformer Blocks Backward (reversed: Block N -> ... -> Block 1):
  dX = block.backward(dX)
         ↓  dX: (B, T, D)
Embedding Backward:
  dW_e, dW_p accumulated via scatter-add and sum
```

### 2. Functions

#### `GPT.__init__(vocab_size, max_seq_len, d_model, num_heads, num_layers, d_ff=None, std=0.02)`
- **WHAT**: Instantiates the Embedding layer, a list of $N$ `TransformerBlock`s, a dedicated Final `LayerNorm`, and initializes the LM Head projection weights $W_{\text{vocab}}$ and bias $b_{\text{vocab}}$.
- **WHY**: Assembles all sublayers into an end-to-end deep language model architecture.
- **INPUT**: `vocab_size` (`int`), `max_seq_len` (`int`), `d_model` (`int`), `num_heads` (`int`), `num_layers` (`int`), `d_ff` (`int` or `None`), `std` (`float`).
- **OUTPUT**: `None`.
- **MAIN FORMULA**: $W_{\text{vocab}} \sim \mathcal{N}(0, 0.02^2), \quad b_{\text{vocab}} = \mathbf{0}$.
- **CONNECTION**: Top-level model constructor tying together all components from Phases 2–4.

#### `GPT.forward(token_ids)`
- **WHAT**: Runs token IDs through Embeddings, sequentially through all $N$ Transformer blocks, normalizes with Final LayerNorm, and projects to vocabulary logits via the LM Head.
- **WHY**: Computes next-token prediction scores across the vocabulary for every position in the sequence.
- **INPUT**: `token_ids` (`np.ndarray` of shape `(B, T)`).
- **OUTPUT**: `logits` (`np.ndarray` of shape `(B, T, V)`).
- **MAIN FORMULA**: Implements $\text{logits} = \text{LN}_f(f_N(\dots f_1(X_{\text{emb}}))) W_{\text{vocab}} + b_{\text{vocab}}$.
- **CONNECTION**: Consumes token IDs from `BPETokenizer.encode()` and produces raw logits for `loss.py` (Phase 6).

#### `GPT.backward(dlogits)`
- **WHAT**: Computes gradients for the LM Head ($dW_{\text{vocab}}, db_{\text{vocab}}$), then backpropagates through the Final LayerNorm, all Transformer blocks in reverse order ($N \to 1$), and the Embedding layer.
- **WHY**: Computes exact analytical gradients for all learnable parameters in the entire network via the chain rule.
- **INPUT**: `dlogits` (`np.ndarray` of shape `(B, T, V)`).
- **OUTPUT**: `None` (stores gradients in all sublayers).
- **MAIN FORMULAS**:
  - $dW_{\text{vocab}} = X_{\text{norm}}^T @ dlogits, \quad db_{\text{vocab}} = \sum_{B, T} dlogits$
  - $dX_{\text{norm}} = dlogits @ W_{\text{vocab}}^T$
  - `dX = ln_f.backward(dX_norm)`
  - `dX = block[i].backward(dX)` for $i = N \dots 1$
  - `token_embeddings.backward(dX)`
- **CONNECTION**: Receives loss gradient $dlogits$ from `loss.py` (Phase 6) and populates parameter gradients for `optimizer.py` (Phase 7).

#### `GPT.get_params()`
- **WHAT**: Gathers all trainable parameter arrays across the entire model into a single named dictionary.
- **WHY**: Gives the optimizer direct access to all model weights.
- **INPUT**: None.
- **OUTPUT**: `dict[str, np.ndarray]`.
- **CONNECTION**: Passed to the optimizer in Phase 7 for weight updates.

#### `GPT.get_grads()`
- **WHAT**: Gathers all computed gradient arrays across the entire model into a single named dictionary matching `get_params()`.
- **WHY**: Gives the optimizer the gradients needed to perform gradient descent updates.
- **INPUT**: None.
- **OUTPUT**: `dict[str, np.ndarray]`.
- **CONNECTION**: Passed to the optimizer in Phase 7 for weight updates.

---

### 3. Tiny Example

```text
Input Token IDs:
[[1, 0]]  (B = 1, T = 2)
   ↓ (Embedding: W_e + W_p)
X:
[[ 0.45,  0.00],
 [ 0.97,  0.15]]
   ↓ (TransformerBlock 1: Attention + FFN + Residuals)
Block 1 Output:
[[ 1.35,  3.02],
 [ 1.90, -1.68]]
   ↓ (Final LayerNorm: normalize to mean 0, variance 1)
X_norm:
[[-1.00,  1.00],
 [ 1.00, -1.00]]
   ↓ (LM Head: X_norm @ W_vocab + b_vocab, where W_vocab is (2, 3) and b_vocab is (3,))
W_vocab:
[[ 0.20, -0.10,  0.50],
 [-0.30,  0.40,  0.10]]
b_vocab:
[ 0.00,  0.00,  0.00]
   ↓
Logits (shape (1, 2, 3)):
[[-0.50,  0.50, -0.40],
 [ 0.50, -0.50,  0.40]]
```

---

### 4. Interview Key Point
> **1. Final LayerNorm**: In GPT's Pre-LayerNorm architecture, each Transformer block adds its unnormalized output directly to the residual stream ($x_l = x_{l-1} + f(\text{LN}(x_{l-1}))$). Over $N$ layers, the variance of the residual activations accumulates and grows. The **Final LayerNorm** stabilizes this variance back to $\mu = 0, \sigma^2 = 1$ right before the LM Head, ensuring stable inputs to the final classification layer.
>
> **2. LM Head**: The **LM Head (Language Modeling Head)** is a linear projection layer ($X_{\text{norm}} W_{\text{vocab}} + b_{\text{vocab}}$) mapping each token's hidden representation of dimension $D$ to an unnormalized score (logit) for every word in the vocabulary of size $V$. The highest logit indicates the model's most likely prediction for the next token.

---

## Phase 6: `loss.py` (Softmax, Cross-Entropy Loss & Backprop Launch)

### 1. Flowchart

```text
Forward Pass:

Token IDs: (B, T)
      ↓
GPT Model (model.py)
      ↓
Logits: (B, T, V)
      ↓
Softmax (along vocab axis: exp(z - max(z)) / sum(exp(z - max(z))))
      ↓
Probabilities: (B, T, V)
      │
      ├──────────────────────────────┐
      ↓                              ↓
True Target IDs: (B, T)      Probabilities: (B, T, V)
      ↓                              ↓
Target Probabilities: p_target = probs[b, t, targets[b, t]]
      ↓
Cross-Entropy per Token: -log(p_target)
      ↓
Mean Loss: (1 / (B * T)) * sum(-log(p_target)) ──> Scalar Training Loss


Backward Pass:

dLogits = (Probabilities - Targets_one_hot) / (B * T)  ──> Shape: (B, T, V)
      ↓
GPT.backward(dlogits)
      ↓
Backpropagate through: LM Head ──> Final LayerNorm ──> Transformer Blocks ──> Embeddings
```

### 2. Functions

#### `softmax(x, axis=-1)`
- **WHAT**: Normalizes logits into probabilities summing to $1.0$ after subtracting the maximum value along the axis.
- **WHY**: Subtracting $\max(x)$ prevents exponential overflow ($\exp(1000) \to \infty$) while preserving exact probability ratios.
- **INPUT**: `x` (`np.ndarray` of shape `(B, T, V)` or any shape), `axis` (`int`, default: `-1`).
- **OUTPUT**: `np.ndarray` of probabilities in $[0, 1]$ summing to $1.0$.
- **MAIN FORMULA**: Implements $\text{softmax}(z_i) = \frac{\exp(z_i - \max(z))}{\sum_j \exp(z_j - \max(z))}$.
- **CONNECTION**: Converts raw GPT logits to vocabulary probability distributions.

#### `cross_entropy_loss(logits, targets, eps=1e-15)`
- **WHAT**: Computes numerically stable softmax, evaluates the mean negative log-likelihood of target tokens, and computes analytical gradient $dLogits$.
- **WHY**: Standard training objective for language modeling, measuring how surprised the model is by the actual next token.
- **INPUT**: `logits` (`np.ndarray` of shape `(B, T, V)`), `targets` (`np.ndarray` of shape `(B, T)`), `eps` (`float`, default: `1e-15`).
- **OUTPUT**: `tuple(loss, dlogits)` where `loss` is a scalar `float` and `dlogits` is `np.ndarray` of shape `(B, T, V)`.
- **MAIN FORMULA**:
  - $\text{Loss} = -\frac{1}{B \times T} \sum_{b=1}^B \sum_{t=1}^T \log(\text{probs}[b, t, \text{targets}[b, t]])$
  - $dLogits = \frac{\text{probs} - \text{targets}_{\text{one-hot}}}{B \times T}$
- **CONNECTION**: Directly provides the scalar loss for monitoring and $dLogits$ to feed into `GPT.backward()`.

#### `CrossEntropyLoss.forward(logits, targets)`
- **WHAT**: Computes and caches cross-entropy loss, predicted probabilities, and target tokens.
- **WHY**: Stateful class interface matching standard deep learning loss layers.
- **INPUT**: `logits` (`(B, T, V)`), `targets` (`(B, T)`).
- **OUTPUT**: `loss` (`float`).
- **CONNECTION**: Called during each training step forward pass.

#### `CrossEntropyLoss.backward()`
- **WHAT**: Returns cached analytical gradient $dLogits$.
- **WHY**: Supplies the starting gradient vector to launch the model-wide backpropagation.
- **INPUT**: None.
- **OUTPUT**: `dlogits` (`np.ndarray` of shape `(B, T, V)`).
- **MAIN FORMULA**: Implements $\frac{\partial \mathcal{L}}{\partial z} = \frac{P - Y}{N}$.
- **CONNECTION**: Directly passed to `GPT.backward(dlogits)`.

---

### 3. Tiny Example

Let batch $B = 1$, sequence length $T = 2$, vocabulary size $V = 3$.

```text
1. Token IDs:
[[1, 0]]
   ↓ (GPT Forward Pass)
2. Logits (B=1, T=2, V=3):
[[ 1.00,  2.00,  0.00],
 [ 0.00,  3.00,  1.00]]
   ↓ (Softmax along axis=-1)
3. Probabilities:
[[0.21, 0.58, 0.08],
 [0.04, 0.84, 0.12]]
   ↓
4. Target Token IDs:
[[1, 2]]  (At t=0, target is 1; at t=1, target is 2)
   ↓
5. Target Probabilities:
At t=0: probs[0, 0, 1] = 0.58
At t=1: probs[0, 1, 2] = 0.12
   ↓
6. Cross-Entropy per Token (-log(p_target)):
At t=0: -log(0.58) = 0.54
At t=1: -log(0.12) = 2.12
   ↓
7. Mean Loss:
Loss = (0.54 + 2.12) / 2 = 1.33
   ↓
8. dLogits = (Probabilities - Targets_one_hot) / (B * T):
Targets_one_hot:
[[0, 1, 0],
 [0, 0, 1]]

(Probabilities - Targets_one_hot):
[[ 0.21, -0.42,  0.08],
 [ 0.04,  0.84, -0.88]]

Divided by N = B * T = 2:
dLogits:
[[ 0.105, -0.210,  0.040],
 [ 0.020,  0.420, -0.440]]
   ↓
9. GPT.backward(dlogits)
Gradients flow backwards into LM Head, Final LayerNorm, Transformer blocks, and Embeddings!
```

---

### 4. Interview Key Point
> **1. Why subtract $\max(z)$ in Softmax?**
> In floating-point arithmetic, $\exp(z)$ overflows to `+inf` for $z > 709$. By computing $\exp(z - \max(z))$, the largest exponent becomes $\exp(0) = 1$, mathematically guaranteeing that every term is in $(0, 1]$ and eliminating overflow while preserving the exact probability distribution.
>
> **2. Why does $dLogits = \frac{\text{probs} - \text{targets}}{N}$ have such a simple derivative?**
> When combining Cross-Entropy Loss ($\mathcal{L} = -\sum y_k \log p_k$) with Softmax ($p_k = \frac{e^{z_k}}{\sum e^{z_j}}$), the messy exponential terms in the quotient rule cancel out perfectly, yielding the remarkably elegant derivative $\frac{\partial \mathcal{L}}{\partial z_i} = \frac{p_i - y_i}{N}$. The gradient is simply the model's prediction error!

---

## End-to-End Backpropagation Flow (Step-by-Step Chain)

This section traces exactly how the loss gradient travels backward through every single file and function in the codebase, from the scalar loss down to the embedding tables.

### 1. Simple Backpropagation Flowchart

```text
                               Loss L (Cross-Entropy)
                                         ↓
    Step 1: loss.py                      ↓  Input: Probs (B, T, V), Targets (B, T)
    cross_entropy_loss()                 ↓  Calculates: dLogits = (Probs - Targets) / N
                                         ↓
                                 dLogits: (B, T, V)
                                         ↓
    Step 2: model.py                     ↓  Calculates: dX_norm = dLogits @ W_vocabᵀ
    GPT.backward() (LM Head)             ├──> Computes & Saves: dW_vocab, db_vocab
                                         ↓
                                 dX_norm: (B, T, D)
                                         ↓
    Step 3: transformer.py               ↓  Calculates: dx through LayerNorm
    LayerNorm.backward() (ln_f)          ├──> Computes & Saves: dgamma, dbeta (Final LN)
                                         ↓
                                    dX: (B, T, D)
                                         ↓
    Step 4: transformer.py               ↓  Backprop through Block N down to 1:
    TransformerBlock.backward()          │  - FFN backward ──────> Saves: dW1, db1, dW2, db2
    (Repeats N → 1 in reverse)           │  - LayerNorm 2 ───────> Saves: dgamma, dbeta
                                         │  - Attention backward ─> Saves: dW_q,k,v,o, db_q,k,v,o
                                         │  - LayerNorm 1 ───────> Saves: dgamma, dbeta
                                         ↓
                                    dX: (B, T, D)
                                         ↓
    Step 5: embeddings.py                ↓  Calculates:
    Embedding.backward()                 ├──> Computes & Saves: dW_e (Token Embeddings)
                                         └──> Computes & Saves: dW_p (Positional Embeddings)
                                         ↓
                            All Gradients Saved in Memory!
                                         ↓
    Step 6: optimizer.py                 ↓  Optimizer.step(GPT.get_grads())
    SGD / Adam                           └──> Updates All Model Weights in-place: W = W - ΔW
```

### 2. Gradient Flow Summary Table

| Step | File | Function | Input Gradient | What Gets Saved | Output Gradient Sent Next |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | `loss.py` | `cross_entropy_loss()` | *(from loss)* | — | `dlogits` $(B, T, V)$ |
| **2** | `model.py` | `GPT.backward()` (LM Head) | `dlogits` $(B, T, V)$ | $dW_{\text{vocab}}, db_{\text{vocab}}$ | `dX_norm` $(B, T, D)$ |
| **3** | `transformer.py` | `LayerNorm.backward()` (`ln_f`) | `dX_norm` $(B, T, D)$ | $d\gamma, d\beta$ (Final LN) | `dx` $(B, T, D)$ |
| **4** | `transformer.py`<br>`attention.py` | `TransformerBlock.backward()`<br>*(reversed: $N \to 1$)* | `dx` $(B, T, D)$ | Block $N \dots 1$ weights:<br>• FFN ($W_1, b_1, W_2, b_2$)<br>• Attention ($W_{q,k,v,o}, b_{q,k,v,o}$)<br>• LN 1 & 2 ($\gamma, \beta$) | `dx` $(B, T, D)$ |
| **5** | `embeddings.py` | `Embedding.backward()` | `dx` $(B, T, D)$ | $dW_e, dW_p$ | — *(All gradients ready!)* |
| **6** | `optimizer.py` | `Optimizer.step()` | `GPT.get_grads()` | Updates all weights in-place | — *(Parameters updated)* |

### 3. Step-by-Step Walkthrough

1. **Step 1 (`loss.py` $\to$ `cross_entropy_loss`)**:
   - Takes predicted probabilities and true target IDs.
   - Calculates prediction error: $dLogits = \frac{\text{probs} - \text{targets}}{B \times T}$.
   - Sends `dlogits` $(B, T, V)$ to `model.py`.

2. **Step 2 (`model.py` $\to$ `GPT.backward`)**:
   - Takes `dlogits` and cached activations $X_{\text{norm}}$.
   - Calculates and saves LM Head gradients: $dW_{\text{vocab}} = X_{\text{norm}}^T @ dlogits$ and $db_{\text{vocab}} = \sum dlogits$.
   - Calculates $dX_{\text{norm}} = dlogits @ W_{\text{vocab}}^T$ and sends it to `ln_f` in `transformer.py`.

3. **Step 3 (`transformer.py` $\to$ `LayerNorm.backward`)**:
   - Takes $dX_{\text{norm}}$ and computes $d\gamma, d\beta$ for the Final LayerNorm (`ln_f`).
   - Computes normalized gradient $dx$ and passes it to the last Transformer Block ($N$).

4. **Step 4 (`transformer.py` & `attention.py` $\to$ `TransformerBlock.backward`)**:
   - Loops backward through blocks ($N \to N-1 \dots \to 1$).
   - Inside each block, gradient flows through both residual branches:
     - **FFN Branch**: `FeedForward.backward()` calculates $dW_2, db_2, dW_1, db_1$, then `LayerNorm.backward()` calculates $d\gamma_2, d\beta_2$.
     - **Attention Branch**: `MultiHeadAttention.backward()` calculates $dW_o, db_o, dW_q, db_q, dW_k, db_k, dW_v, db_v$, then `LayerNorm.backward()` calculates $d\gamma_1, d\beta_1$.
   - Outputs gradient $dx$ $(B, T, D)$ to the previous block (or to embeddings).

5. **Step 5 (`embeddings.py` $\to$ `Embedding.backward`)**:
   - Takes $dx$ from Block 1.
   - Sums across batch for positional weights ($dW_p$) and uses `np.add.at` for token weights ($dW_e$).
   - Gradients for all model parameters are now fully computed!

6. **Step 6 (`optimizer.py` $\to$ `Optimizer.step`)**:
   - Takes the complete dictionary of gradients from `GPT.get_grads()`.
   - Updates each parameter in-place: $W = W - \text{update}$ (via SGD or Adam).

---

## Phase 7: `optimizer.py` & `train.py` (Optimizers & Complete Training Loop)

### 1. Flowchart

```text
                            Complete Training Step Loop:
                            
                1D Tokenized Dataset: data [0, 1, 2, 3, ...]
                                     ↓
                    get_batch(data, batch_size, seq_len)
                                     ↓
             ┌───────────────────────┴───────────────────────┐
             ↓                                               ↓
       Inputs X: (B, T)                               Targets Y: (B, T)
             ↓                                               │
       GPT.forward(X)                                        │
             ↓                                               │
       Logits: (B, T, V)                                     │
             ↓                                               │
      CrossEntropyLoss.forward(Logits, Y) ◄──────────────────┘
             ↓
       Scalar Loss (reported for progress monitoring)
             ↓
      CrossEntropyLoss.backward()
             ↓
       dLogits: (B, T, V)
             ↓
       GPT.backward(dLogits)
             ↓ (Backpropagation through LM Head, LN_f, Blocks N..1, Embeddings)
       GPT.get_grads()
             ↓ (Dictionary of parameter gradients: dW, db, dgamma, dbeta, etc.)
       Optimizer.step(grads) [SGD or Adam]
             ↓
       In-place Parameter Updates: W = W - update
             ↓
       Next Training Step (Loss Decreases!)
```

### 2. Functions

#### `Optimizer.__init__(params, lr=1e-3)`
- **WHAT**: Base class storing references to trainable parameter arrays and the learning rate.
- **WHY**: Provides a unified interface for all optimization algorithms.
- **INPUT**: `params` (`dict[str, np.ndarray]`), `lr` (`float`).
- **OUTPUT**: `None`.
- **CONNECTION**: Receives dictionary returned by `GPT.get_params()`.

#### `SGD.step(grads)`
- **WHAT**: Updates each parameter along the negative gradient direction: $W = W - \text{lr} \cdot g$ (with optional velocity momentum).
- **WHY**: Moves parameters down the loss surface to minimize prediction error.
- **INPUT**: `grads` (`dict[str, np.ndarray]`).
- **OUTPUT**: `None` (modifies parameters in-place).
- **MAIN FORMULA**: Implements $W_{t} = W_{t-1} - \alpha \nabla \mathcal{L}(W_{t-1})$.
- **CONNECTION**: Uses gradients from `GPT.get_grads()` to update model weights.

#### `Adam.__init__(params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0)`
- **WHAT**: Initializes first moment buffers $m$, second moment buffers $v$, and timestep $t=0$ for every parameter tensor.
- **WHY**: Adam tracks running averages of both gradient direction ($m$) and gradient magnitude/variance ($v$) for each weight individually.
- **INPUT**: `params` (`dict`), `lr` (`float`), `beta1` (`float`), `beta2` (`float`), `eps` (`float`), `weight_decay` (`float`).
- **OUTPUT**: `None`.
- **CONNECTION**: Stores per-parameter state matching `GPT.get_params()`.

#### `Adam.step(grads)`
- **WHAT**: Updates first moment $m$, second moment $v$, computes bias corrections $\hat{m}$ and $\hat{v}$, and updates parameters in-place.
- **WHY**: Automatically scales learning rates: gives large updates to infrequently updated features and small, cautious updates to noisy features.
- **INPUT**: `grads` (`dict[str, np.ndarray]`).
- **OUTPUT**: `None` (modifies parameters in-place).
- **MAIN FORMULAS**:
  - $m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad$ *(First moment: exponential moving average of gradient)*
  - $v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \quad$ *(Second moment: exponential moving average of squared gradient)*
  - $\hat{m}_t = \frac{m_t}{1 - \beta_1^t} \quad$ *(Bias correction for first moment)*
  - $\hat{v}_t = \frac{v_t}{1 - \beta_2^t} \quad$ *(Bias correction for second moment)*
  - $W_t = W_{t-1} - \frac{\alpha \hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} \quad$ *(Adaptive parameter update)*
- **CONNECTION**: Applied after `GPT.backward()` on every training iteration.

#### `get_batch(data, batch_size, seq_len)`
- **WHAT**: Randomly slices chunks of length `seq_len` from 1D token data as inputs $X$, and chunks shifted by 1 position as next-token targets $Y$.
- **WHY**: Provides mini-batches where the goal at every position $t$ is to predict the next token $t+1$.
- **INPUT**: `data` (`np.ndarray` of shape `(N,)`), `batch_size` (`int`), `seq_len` (`int`).
- **OUTPUT**: `tuple(x, y)` both of shape `(batch_size, seq_len)`.
- **MAIN FORMULA**: $Y_{b, t} = X_{b, t+1}$.
- **CONNECTION**: Prepares data batches for `GPT.forward()` and `CrossEntropyLoss.forward()`.

#### `train_step(model, criterion, optimizer, x, y)`
- **WHAT**: Executes one complete iteration: forward pass $\to$ loss calculation $\to$ backpropagation $\to$ optimizer parameter update.
- **WHY**: The atomic training step of gradient descent.
- **INPUT**: `model` (`GPT`), `criterion` (`CrossEntropyLoss`), `optimizer` (`Optimizer`), `x` (`(B, T)`), `y` (`(B, T)`).
- **OUTPUT**: `loss` (`float`).
- **CONNECTION**: Glues Phases 5, 6, and 7 together.

#### `train(model, data, optimizer, criterion=None, batch_size=4, seq_len=16, num_steps=100, log_interval=10, verbose=True)`
- **WHAT**: Loops `train_step` over multiple iterations, reports loss progress periodically, and returns loss history.
- **WHY**: Trains the GPT model end-to-end to minimize cross-entropy loss on the dataset.
- **INPUT**: Model, data, optimizer, hyperparameters.
- **OUTPUT**: `list[float]` of loss values per step.
- **CONNECTION**: High-level training entry point for the entire project.

---

### 3. Tiny Example

Let's trace a single parameter update with **SGD** and **Adam** on a tiny weight scalar $W = 1.0$, gradient $g = 0.2$, learning rate $\alpha = 0.1$:

```text
1. SGD Update:
Initial W = 1.0000,  Gradient g = 0.2000,  lr = 0.1000

Update:
W_new = W - lr * g
      = 1.0000 - (0.1000 * 0.2000)
      = 1.0000 - 0.0200
      = 0.9800

---

2. Adam Update (Step t = 1):
Initial W = 1.0000,  Gradient g = 0.2000,  lr = 0.1000,  beta1 = 0.9,  beta2 = 0.999,  eps = 1e-8

Step 1: First Moment (m_1):
m_1 = beta1 * 0 + (1 - beta1) * g
    = 0.9 * 0 + 0.1 * 0.2000 = 0.0200

Step 2: Second Moment (v_1):
v_1 = beta2 * 0 + (1 - beta2) * g^2
    = 0.999 * 0 + 0.001 * (0.0400) = 0.0000400

Step 3: Bias Corrections (t = 1):
m_hat_1 = m_1 / (1 - beta1^1) = 0.0200 / 0.1000 = 0.2000
v_hat_1 = v_1 / (1 - beta2^1) = 0.0000400 / 0.0010 = 0.04000

Step 4: Parameter Update:
sqrt(v_hat_1) + eps = sqrt(0.04000) + 1e-8 = 0.2000
effective_step = m_hat_1 / (sqrt(v_hat_1) + eps) = 0.2000 / 0.2000 = 1.0000

W_new = W - lr * effective_step
      = 1.0000 - (0.1000 * 1.0000)
      = 0.9000
```

---

### 4. Interview Key Point
> **1. Why do we need an Optimizer instead of just subtracting gradients?**
> Gradients only indicate the direction of steepest ascent at a single point. Simple SGD can oscillate wildly across steep ravines while making almost no progress along flat directions. **Adam** solves this by maintaining per-parameter adaptive learning rates: dividing by $\sqrt{v_t}$ normalizes the update magnitude so steep directions are damped and flat directions are accelerated.
>
> **2. Why is Bias Correction ($1 - \beta^t$) needed in Adam?**
> Because $m$ and $v$ are initialized to zero, the uncorrected moving averages are heavily biased towards zero during early iterations (e.g. at $t=1$, $m_1 = 0.1 g$, which is 10× too small!). Dividing by $(1 - \beta^t)$ cancels this initialization bias, ensuring accurate gradient estimation from the very first step.

---

## Phase 8: `generate.py` (Autoregressive Text Generation)

### 1. Flowchart

```text
Prompt: "ROMEO:"
   ↓
[tokenizer.py] tokenizer.encode("ROMEO:")
   ↓
Token IDs: idx = [[15, 42]] (shape: 1, T)
   ↓
┌─────────────────────────────────────────────────────────────┐
│ Autoregressive Generation Loop (repeats max_new_tokens):   │
│                                                             │
│ 1. Crop Context: idx_cond = idx[:, -max_seq_len:]           │
│ 2. Forward Pass: logits = GPT.forward(idx_cond)             │
│ 3. Last Logits:  next_logits = logits[:, -1, :] (1, V)      │
│ 4. Temperature:  scaled_logits = next_logits / T            │
│ 5. Filter:       apply top-k and top-p (nucleus)            │
│ 6. Probabilities: probs = softmax(scaled_logits)            │
│ 7. Select Token:                                            │
│      - Greedy:   argmax(next_logits)                        │
│      - Sample:   np.random.choice(V, p=probs)               │
│ 8. Append:       idx = [idx, next_token]                    │
│ 9. Check EOS:    if next_token == <|endoftext|>, break!     │
└─────────────────────────────────────────────────────────────┘
   ↓
All Token IDs: [15, 42, 8, 22, 31, ...]
   ↓
[tokenizer.py] tokenizer.decode(idx[0])
   ↓
Completed Text: "ROMEO: What lady is that?"
```

### 2. Core Concepts & Functions

#### Training vs. Generation
| Aspect | Training (Phase 7) | Generation (Phase 8) |
| :--- | :--- | :--- |
| **Data Flow** | Batch of fixed sequences $(B, T)$ in parallel | Token-by-token sequentially (autoregressive) |
| **Targets** | Known ground-truth next tokens | No targets exist; model generates its own next token |
| **Loss & Backward** | Computes Cross-Entropy & backpropagates gradients | **NO** loss, **NO** backward pass, **NO** gradients |
| **Weights** | Updated by Optimizer ($W = W - \Delta W$) | **Frozen** (inference-only forward passes) |

#### Why Use Only the Last-Position Logits?
- GPT is a causal (autoregressive) language model.
- Because of the **causal mask**, token position $t$ only attends to positions $\le t$.
- Therefore, the output vector at the **final position** $T-1$ (`logits[:, -1, :]`) contains the aggregated representation of the entire prompt and represents the model's prediction for token $T$.

#### Greedy vs. Sampling
- **Greedy Decoding (`argmax`)**: Always picks the single token with the highest logit. Deterministic and fast, but prone to repetitive loops and dull text.
- **Sampling (`np.random.choice`)**: Treats the softmax output as a probability distribution and draws a random sample. Introduces variety, creativity, and natural phrasing.

#### Temperature Scaling
- Modifies logits before softmax: $z' = z / T$.
- **$T < 1.0$ (Low)**: Sharpens probabilities towards the highest logit (more deterministic and confident).
- **$T = 1.0$ (Neutral)**: Standard model probabilities.
- **$T > 1.0$ (High)**: Flattens probabilities towards uniform (more diverse, creative, but potentially nonsensical).
- **$T = 0.0$**: Mathematically equivalent to greedy `argmax`.

#### Top-K & Top-P (Nucleus) Filtering
- **Top-K**: Keeps only the $k$ tokens with the highest logits; sets all other logits to $-\infty$ so their probability becomes strictly 0. Prevents sampling bizarre tail tokens.
- **Top-P (Nucleus)**: Dynamically keeps the smallest set of top tokens whose cumulative probability $\ge p$ (e.g. $p = 0.9$). Adjusts dynamically: keeps fewer tokens when the model is confident, and more when uncertain.

#### End-of-Sequence (EOS) Stopping
- When the model predicts `<|endoftext|>`, it signals that the thought or document is complete. Generation breaks out of the loop immediately rather than continuing up to `max_new_tokens`.

#### Why Backpropagation is NOT Used in Generation
- Generation is **inference**, not learning.
- The weights are fixed. The goal is solely to sample text from the learned probability distribution, requiring only forward passes.

---

### 3. Tiny Example

Let prompt token be `[1]`, vocabulary size $V = 3$, temperature $T = 0.5$:

```text
1. Prompt Token:
idx = [[1]]
   ↓ (GPT Forward Pass)
2. Last Logits:
next_logits = [[2.0, 1.0, 0.0]]
   ↓ (Temperature Scaling: z / 0.5)
scaled_logits = [[4.0, 2.0, 0.0]]
   ↓ (Softmax)
probs = [[0.84, 0.14, 0.02]]
   ↓
3. Selection:
   - Greedy: argmax([2.0, 1.0, 0.0]) = Token 0
   - Sampling (T=0.5): 84% chance Token 0, 14% Token 1, 2% Token 2
   ↓
4. Append Selected Token:
idx = [[1, 0]]
   ↓
5. Repeat for next step!
```

---

### 4. Interview Key Point
> In text generation, **GPT predicts one token at a time autoregressively** by feeding its own previous outputs back as inputs. We use **temperature** to control randomness, **top-k / top-p** to eliminate the low-probability tail, and **no backpropagation** because inference only queries the learned distribution without updating model weights.
