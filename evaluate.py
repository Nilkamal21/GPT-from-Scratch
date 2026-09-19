"""
Evaluation Module for NumPyGPT (Phase 9A).

This module implements the evaluation framework:
1. Train/Validation dataset splitting (chronological or seeded).
2. Trainable parameter counting.
3. Numerically stable cross-entropy loss estimation across evaluation batches.
4. Perplexity calculation (PPL = exp(loss)).
5. Strict inference-only evaluation safety (no backward passes, no parameter updates).
"""

from typing import Dict, Optional, Tuple
import numpy as np
from model import GPT
from loss import CrossEntropyLoss
from train import get_batch


def train_val_split(
    data: np.ndarray,
    split_ratio: float = 0.9,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Splits 1D tokenized data into train and validation sets chronologically.
    
    Chronological splitting preserves the sequential autoregressive structure
    of text, ensuring no data leakage from future text into training.
    
    Args:
        data (np.ndarray): 1D integer array of token IDs.
        split_ratio (float): Fraction of data used for training (default: 0.9).
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (train_data, val_data).
    """
    if data.ndim != 1:
        raise ValueError(f"Expected 1D data array, got shape {data.shape}")
    if not (0.0 < split_ratio < 1.0):
        raise ValueError(f"split_ratio must be in (0.0, 1.0), got {split_ratio}")

    n = int(len(data) * split_ratio)
    train_data = data[:n]
    val_data = data[n:]

    return train_data, val_data


def count_parameters(model: GPT) -> int:
    """
    Counts the total number of trainable parameters in the GPT model.
    
    Args:
        model (GPT): GPT model instance.
        
    Returns:
        int: Total number of scalar parameters across all layers.
    """
    params = model.get_params()
    return sum(p.size for p in params.values())


def compute_perplexity(loss: float) -> float:
    """
    Computes perplexity from cross-entropy loss.
    
    Formula:
        Perplexity = exp(loss)
        
    Args:
        loss (float): Cross-entropy loss in natural log (nats).
        
    Returns:
        float: Perplexity metric (effective vocabulary branching factor).
    """
    if loss < 0.0:
        raise ValueError(f"Loss cannot be negative, got {loss}")
    # Cap loss to prevent overflow in exp()
    clipped_loss = min(loss, 100.0)
    return float(np.exp(clipped_loss))


def estimate_loss(
    model: GPT,
    data: np.ndarray,
    criterion: Optional[CrossEntropyLoss] = None,
    batch_size: int = 4,
    seq_len: int = 16,
    eval_iters: int = 20,
    seed: Optional[int] = None,
) -> float:
    """
    Estimates mean cross-entropy loss over multiple randomly sampled batches.
    
    Strictly inference-only:
    - Only calls model.forward()
    - Does NOT call backward()
    - Does NOT calculate gradients
    - Does NOT invoke optimizer
    - Model weights remain completely untouched
    
    Args:
        model (GPT): GPT model instance.
        data (np.ndarray): 1D tokenized data to evaluate on.
        criterion (Optional[CrossEntropyLoss]): Loss function.
        batch_size (int): Number of sequences per batch.
        seq_len (int): Context window length.
        eval_iters (int): Number of batches to average over.
        seed (Optional[int]): Random seed for deterministic batch selection.
        
    Returns:
        float: Mean cross-entropy loss across all evaluation batches.
    """
    if criterion is None:
        criterion = CrossEntropyLoss()

    if seed is not None:
        rng_state = np.random.get_state()
        np.random.seed(seed)

    losses = []
    for _ in range(eval_iters):
        x, y = get_batch(data, batch_size=batch_size, seq_len=seq_len)
        logits = model.forward(x)
        loss = criterion.forward(logits, y)
        losses.append(loss)

    if seed is not None:
        np.random.set_state(rng_state)

    return float(np.mean(losses))


def evaluate_model(
    model: GPT,
    train_data: np.ndarray,
    val_data: np.ndarray,
    criterion: Optional[CrossEntropyLoss] = None,
    batch_size: int = 4,
    seq_len: int = 16,
    eval_iters: int = 20,
    seed: Optional[int] = 42,
) -> Dict[str, float]:
    """
    Runs full evaluation on both train and validation sets.
    
    Returns:
        Dict[str, float]: Dictionary containing:
            - train_loss
            - val_loss
            - train_perplexity
            - val_perplexity
            - num_parameters
    """
    if criterion is None:
        criterion = CrossEntropyLoss()

    train_loss = estimate_loss(
        model, train_data, criterion, batch_size=batch_size, seq_len=seq_len, eval_iters=eval_iters, seed=seed
    )
    val_loss = estimate_loss(
        model, val_data, criterion, batch_size=batch_size, seq_len=seq_len, eval_iters=eval_iters, seed=seed
    )

    return {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_perplexity": compute_perplexity(train_loss),
        "val_perplexity": compute_perplexity(val_loss),
        "num_parameters": float(count_parameters(model)),
    }


if __name__ == "__main__":
    print("NumPyGPT Phase 9A: Evaluation Demo")
    print("=" * 45)

    # Initialize tokenizer and dataset
    with open("data/input.txt", "r", encoding="utf-8") as f:
        text = f.read(50_000)

    vocab_size = 100
    from tokenizer import BPETokenizer
    tokenizer = BPETokenizer(vocab_size=vocab_size)
    tokenizer.train(text)

    token_ids = np.array(tokenizer.encode(text), dtype=np.int32)
    train_tokens, val_tokens = train_val_split(token_ids, split_ratio=0.9)
    print(f"Total tokens:      {len(token_ids):,}")
    print(f"Train split (90%): {len(train_tokens):,} tokens")
    print(f"Val split (10%):   {len(val_tokens):,} tokens")

    # Initialize GPT model
    model = GPT(
        vocab_size=vocab_size,
        max_seq_len=32,
        d_model=32,
        num_heads=4,
        num_layers=2,
    )

    metrics = evaluate_model(
        model=model,
        train_data=train_tokens,
        val_data=val_tokens,
        batch_size=4,
        seq_len=16,
        eval_iters=25,
        seed=42,
    )

    print("\nEvaluation Metrics (Untrained Model):")
    print(f"Total Parameters:    {int(metrics['num_parameters']):,}")
    print(f"Train Loss:          {metrics['train_loss']:.4f}")
    print(f"Validation Loss:     {metrics['val_loss']:.4f}")
    print(f"Train Perplexity:    {metrics['train_perplexity']:.2f}")
    print(f"Val Perplexity:      {metrics['val_perplexity']:.2f}")
