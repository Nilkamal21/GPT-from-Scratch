# NumPyGPT — Experimental Results

## 1. Evaluation Setup
- **Dataset**: `data/input.txt` (Tiny Shakespeare, 80,000 characters)
- **Train/Validation Split**: 90% training (50,679 tokens) / 10% validation (5,631 tokens), chronologically split
- **Vocabulary Size**: 150 subwords (learned via BPE)
- **Model Size / Parameter Count**: 2-layer GPT, 4 heads, $d_{\text{model}} = 32$, $d_{\text{ff}} = 128$ (36,246 trainable parameters)
- **Context Length**: 32 tokens
- **Random Seed**: 42 (deterministic split and batch sampling)

## 2. Phase 9B — Training Results

| Step | Train Loss | Val Loss | Train PPL | Val PPL |
|:----:|:----------:|:--------:|:---------:|:-------:|
| **0**   | 5.0238 | 5.0227 | 151.99 | 151.82 |
| **20**  | 4.1689 | 4.2343 | 64.64  | 69.01  |
| **40**  | 3.8304 | 3.9234 | 46.08  | 50.57  |
| **60**  | 3.6856 | 3.7751 | 39.87  | 43.60  |
| **80**  | 3.5684 | 3.6455 | 35.46  | 38.30  |
| **100** | 3.4992 | 3.5495 | 33.09  | 34.80  |

- **Training Loss**: Decreased from 5.0238 to 3.4992 (30.3% reduction; initial loss matches random baseline $\ln(150) \approx 5.01$).
- **Validation Loss**: Decreased from 5.0227 to 3.5495 (perplexity dropped from 151.82 to 34.80).
- **Overfitting Assessment**: No obvious overfitting observed; validation loss closely tracked training loss throughout (generalization gap $\Delta \mathcal{L} \approx 0.05$).

## 3. SGD vs Adam

| Optimizer | Initial Loss | Final Train Loss | Final Val Loss | Final Train PPL | Final Val PPL |
|:----------|:------------:|:----------------:|:--------------:|:---------------:|:-------------:|
| **SGD**   | 5.0238       | 4.5827           | 4.6178         | 97.78           | 101.27        |
| **Adam**  | 5.0238       | 3.5459           | 3.6320         | 34.67           | 37.79         |

- **Adam**: 29.4% training-loss reduction (perplexity: 151.99 $\to$ 34.67).
- **SGD**: 8.8% training-loss reduction (perplexity: 151.99 $\to$ 97.78).
- Both experiments used identical initial parameters and training conditions ($B=4, T=32, \text{steps}=100, \text{lr}=3\times 10^{-3}$, seed 42).
- This result is specific to this experiment and should not be generalized beyond the tested setup.

## 4. Generation Example
- **Prompt**: `'ROMEO:'`
- **Temperature**: 0.8
- **Top-k**: 10
- **Generated Output**:
  ```text
  'ROMEO:\nWar,y s pa f c\n\nEM\nAuts ca'
  ```
- *Note*: This is a qualitative sample demonstrating basic token grouping and formatting, not a formal measure of language quality.

## 5. Verification
- **Total Tests**: 56
- **Passed**: 56
- **Failed**: 0
- **Test Runtime**: 2.45s

## 6. Limitations
- **Small Model**: 36,246 parameters with limited capacity to represent full syntax or semantics.
- **Short Training Run**: 100 steps (12,800 tokens processed) is sufficient to observe convergence dynamics, but not enough for fluency.
- **Limited Context / Model Capacity**: 32-token context length restricts attention to short-range local patterns.

## 7. Resume-Relevant Metrics
- **36,246** trainable parameters
- **56,310** tokens
- **29.4%** Adam training-loss reduction
- **8.8%** SGD training-loss reduction
- **37.79** Adam validation perplexity
- **56/56** tests passed

