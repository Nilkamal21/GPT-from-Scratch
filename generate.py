"""
Text Generation Module for NumPyGPT (Phase 8).

This module implements autoregressive text generation using:
1. Greedy decoding (argmax).
2. Stochastic sampling with temperature scaling.
3. Top-k filtering (truncating logits to top k choices).
4. Top-p (nucleus) filtering (truncating logits to cumulative probability threshold).
5. End-of-sequence (EOS) early stopping.
6. Context window cropping to respect model.max_seq_len.

Generation is strictly inference-only (no loss, no backward pass, no optimizer updates).
"""

from typing import Optional
import numpy as np
from model import GPT
from tokenizer import BPETokenizer
from loss import softmax


def top_k_filtering(logits: np.ndarray, top_k: int) -> np.ndarray:
    """
    Filters logits to keep only the top k values, setting all others to -infinity.
    
    Args:
        logits (np.ndarray): Logits tensor of shape (..., V).
        top_k (int): Number of top logits to preserve (must be >= 1).
        
    Returns:
        np.ndarray: Filtered logits tensor of the same shape.
    """
    if top_k <= 0:
        raise ValueError(f"top_k must be a positive integer, got {top_k}")

    V = logits.shape[-1]
    if top_k >= V:
        return logits.copy()

    filtered = logits.copy()
    # Find the threshold value (top-k-th largest element)
    kth_val = np.partition(filtered, -top_k, axis=-1)[..., -top_k, np.newaxis]
    filtered[filtered < kth_val] = -np.inf
    return filtered


def top_p_filtering(logits: np.ndarray, top_p: float) -> np.ndarray:
    """
    Filters logits using nucleus (top-p) sampling.
    Keeps the smallest set of top tokens whose cumulative probability exceeds top_p.
    
    Args:
        logits (np.ndarray): Logits tensor of shape (..., V).
        top_p (float): Cumulative probability threshold in (0.0, 1.0].
        
    Returns:
        np.ndarray: Filtered logits tensor of the same shape.
    """
    if top_p <= 0.0 or top_p > 1.0:
        raise ValueError(f"top_p must be in range (0.0, 1.0], got {top_p}")

    if top_p >= 1.0:
        return logits.copy()

    filtered = logits.copy()

    # Sort logits in descending order
    sorted_indices = np.argsort(-filtered, axis=-1)
    sorted_logits = np.take_along_axis(filtered, sorted_indices, axis=-1)

    # Compute sorted probabilities and cumulative probabilities
    sorted_probs = softmax(sorted_logits, axis=-1)
    cumulative_probs = np.cumsum(sorted_probs, axis=-1)

    # Mask tokens beyond the cumulative threshold
    # Shift right by 1 to always preserve at least the top-1 token
    sorted_mask = cumulative_probs > top_p
    sorted_mask[..., 1:] = sorted_mask[..., :-1].copy()
    sorted_mask[..., 0] = False

    # Scatter mask back to original logit positions
    mask = np.zeros_like(sorted_mask)
    np.put_along_axis(mask, sorted_indices, sorted_mask, axis=-1)
    filtered[mask] = -np.inf

    return filtered


def generate_tokens(
    model: GPT,
    idx: np.ndarray,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    greedy: bool = False,
    eos_id: Optional[int] = None,
) -> np.ndarray:
    """
    Autoregressively generates new token IDs given an initial conditioning sequence.
    
    Args:
        model (GPT): Trained GPT model instance.
        idx (np.ndarray): 2D integer array of shape (B, T) with initial token IDs.
        max_new_tokens (int): Maximum number of new tokens to generate (> 0).
        temperature (float): Sampling temperature (>= 0.0). 0.0 forces greedy decoding.
        top_k (Optional[int]): Keep only top k logits (>= 1).
        top_p (Optional[float]): Nucleus sampling probability threshold in (0.0, 1.0].
        greedy (bool): If True, deterministically selects argmax token at each step.
        eos_id (Optional[int]): End-of-sequence token ID. Stops generation when emitted.
        
    Returns:
        np.ndarray: Extended token array of shape (B, T + num_generated).
    """
    if idx.ndim != 2:
        raise ValueError(f"Expected 2D array of shape (B, T), got {idx.shape}")
    if max_new_tokens <= 0:
        raise ValueError(f"max_new_tokens must be > 0, got {max_new_tokens}")
    if temperature < 0.0:
        raise ValueError(f"temperature must be >= 0.0, got {temperature}")
    if top_k is not None and top_k <= 0:
        raise ValueError(f"top_k must be > 0, got {top_k}")
    if top_p is not None and (top_p <= 0.0 or top_p > 1.0):
        raise ValueError(f"top_p must be in range (0.0, 1.0], got {top_p}")

    # Temperature 0.0 implies deterministic greedy selection
    if temperature == 0.0:
        greedy = True

    current_idx = idx.copy()

    for _ in range(max_new_tokens):
        # 1. Respect max_seq_len by cropping context window to the last max_seq_len tokens
        idx_cond = (
            current_idx[:, -model.max_seq_len:]
            if current_idx.shape[1] > model.max_seq_len
            else current_idx
        )

        # 2. Forward pass through model (inference-only, no backprop or optimizer)
        logits = model.forward(idx_cond)

        # 3. Take logits from the LAST sequence position: (B, V)
        next_token_logits = logits[:, -1, :]

        # 4. Token selection
        if greedy:
            # Deterministic argmax selection
            next_token = np.argmax(next_token_logits, axis=-1, keepdims=True)
        else:
            # Temperature scaling
            scaled_logits = next_token_logits / temperature

            # Optional top-k filtering
            if top_k is not None:
                scaled_logits = top_k_filtering(scaled_logits, top_k)

            # Optional top-p filtering
            if top_p is not None:
                scaled_logits = top_p_filtering(scaled_logits, top_p)

            # Convert to probabilities
            probs = softmax(scaled_logits, axis=-1)

            # Sample next token for each batch element
            batch_tokens = []
            for b in range(current_idx.shape[0]):
                p = probs[b]
                # Numerical safety check
                p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
                p_sum = np.sum(p)
                if p_sum > 0:
                    p = p / p_sum
                else:
                    p = np.ones_like(p) / len(p)
                sampled_token = np.random.choice(len(p), p=p)
                batch_tokens.append(sampled_token)

            next_token = np.array(batch_tokens, dtype=np.int32)[:, np.newaxis]

        # 5. Append sampled token to the running sequence: (B, T + 1)
        current_idx = np.concatenate([current_idx, next_token], axis=1)

        # 6. Check for early stopping on EOS token
        if eos_id is not None:
            # For single sequence (B=1), stop if EOS is produced
            if current_idx.shape[0] == 1 and next_token[0, 0] == eos_id:
                break
            # For batch > 1, stop if all sequences produced EOS
            elif np.all(next_token[:, 0] == eos_id):
                break

    return current_idx


def generate(
    model: GPT,
    tokenizer: BPETokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    greedy: bool = False,
    eos_token: Optional[str] = "<|endoftext|>",
) -> str:
    """
    Generates text continuation from a string prompt.
    
    Pipeline:
        Prompt
            ↓
        tokenizer.encode(prompt) ──> Token IDs: (1, T)
            ↓
        generate_tokens(...)     ──> Extended Token IDs: (1, T + N)
            ↓
        tokenizer.decode(...)    ──> Generated Text String
        
    Args:
        model (GPT): Trained GPT model instance.
        tokenizer (BPETokenizer): Tokenizer with encode and decode methods.
        prompt (str): Text prompt to continue.
        max_new_tokens (int): Maximum new tokens to generate.
        temperature (float): Sampling temperature (0.0 for greedy).
        top_k (Optional[int]): Top-k filtering threshold.
        top_p (Optional[float]): Nucleus sampling threshold.
        greedy (bool): Whether to use greedy decoding.
        eos_token (Optional[str]): Special token string marking end of text.
        
    Returns:
        str: Completed text continuation.
    """
    # 1. Encode prompt to token IDs
    prompt_ids = tokenizer.encode(prompt)

    # If prompt is empty, start from EOS token or token 0
    if len(prompt_ids) == 0:
        start_id = tokenizer.token_to_id.get(eos_token, 0) if eos_token else 0
        prompt_ids = [start_id]

    idx = np.array([prompt_ids], dtype=np.int32)

    # 2. Resolve EOS token ID
    eos_id = tokenizer.token_to_id.get(eos_token, None) if eos_token else None

    # 3. Autoregressive token generation
    output_idx = generate_tokens(
        model=model,
        idx=idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        greedy=greedy,
        eos_id=eos_id,
    )

    # 4. Decode generated tokens back to readable text
    generated_ids = output_idx[0].tolist()
    return tokenizer.decode(generated_ids)


if __name__ == "__main__":
    print("NumPyGPT Phase 8: Text Generation Demo")
    print("=" * 45)

    # Initialize small tokenizer and GPT model
    vocab_size = 60
    tokenizer = BPETokenizer(vocab_size=vocab_size)
    sample_corpus = (
        "To be, or not to be, that is the question:\n"
        "Whether 'tis nobler in the mind to suffer\n"
        "The slings and arrows of outrageous fortune,\n"
        "Or to take arms against a sea of troubles."
    )
    tokenizer.train(sample_corpus)

    model = GPT(
        vocab_size=vocab_size,
        max_seq_len=32,
        d_model=32,
        num_heads=4,
        num_layers=2,
    )

    prompt = "To be"
    print(f"Prompt: {repr(prompt)}")

    # Greedy generation
    greedy_output = generate(model, tokenizer, prompt, max_new_tokens=15, greedy=True)
    print(f"\n[Greedy Generation]:\n{repr(greedy_output)}")

    # Temperature sampling with top-k
    sample_output = generate(
        model,
        tokenizer,
        prompt,
        max_new_tokens=15,
        temperature=0.8,
        top_k=5,
    )
    print(f"\n[Sampled (T=0.8, top_k=5)]:\n{repr(sample_output)}")
