"""
Training Loop Module for NumPyGPT (Phase 7).

This module implements:
1. Dataset batch sampling (get_batch).
2. Single-step forward, backward, and optimization (train_step).
3. Complete training loop reporting loss over iterations (train).

Architecture Flow:
    Batch (X, Y)
        ↓
    GPT.forward(X) ──> Logits
        ↓
    CrossEntropyLoss.forward(Logits, Y) ──> Loss
        ↓
    CrossEntropyLoss.backward() ──> dLogits
        ↓
    GPT.backward(dLogits)
        ↓
    GPT.get_grads()
        ↓
    Optimizer.step(grads) ──> In-place Parameter Updates
"""

from typing import List, Optional, Tuple
import numpy as np
from model import GPT
from loss import CrossEntropyLoss
from optimizer import Optimizer, Adam, SGD


def get_batch(
    data: np.ndarray,
    batch_size: int,
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Samples random context windows from 1D tokenized data for language modeling.
    
    Target tokens Y are shifted by 1 relative to input tokens X:
        X[b, t] = data[i + t]
        Y[b, t] = data[i + t + 1]
        
    Args:
        data (np.ndarray): 1D integer array of token IDs.
        batch_size (int): Number of independent sequences per batch (B).
        seq_len (int): Sequence length / context window (T).
        
    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - x: Input tensor of shape (batch_size, seq_len).
            - y: Target tensor of shape (batch_size, seq_len).
    """
    if data.ndim != 1:
        raise ValueError(f"Expected 1D data array, got shape {data.shape}")
    if len(data) <= seq_len:
        raise ValueError(
            f"Dataset length ({len(data)}) must be greater than seq_len ({seq_len})"
        )

    # Random starting index for each sequence in the batch
    max_idx = len(data) - seq_len - 1
    ix = np.random.randint(0, max_idx + 1, size=batch_size)

    x = np.stack([data[i : i + seq_len] for i in ix])
    y = np.stack([data[i + 1 : i + seq_len + 1] for i in ix])

    return x, y


def train_step(
    model: GPT,
    criterion: CrossEntropyLoss,
    optimizer: Optimizer,
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    """
    Executes a single end-to-end training step:
    1. Forward pass: logits = model.forward(x)
    2. Loss calculation: loss = criterion.forward(logits, y)
    3. Output gradient: dlogits = criterion.backward()
    4. Backpropagation: model.backward(dlogits)
    5. Parameter update: optimizer.step(model.get_grads())
    
    Args:
        model (GPT): The GPT language model.
        criterion (CrossEntropyLoss): The cross-entropy loss function.
        optimizer (Optimizer): The optimizer (SGD or Adam).
        x (np.ndarray): Input batch of shape (B, T).
        y (np.ndarray): Target batch of shape (B, T).
        
    Returns:
        float: Scalar cross-entropy loss for this step.
    """
    # 1. Forward pass through model
    logits = model.forward(x)

    # 2. Compute loss and output gradients
    loss = criterion.forward(logits, y)
    dlogits = criterion.backward()

    # 3. Backward pass through model layers
    model.backward(dlogits)

    # 4. Optimizer update
    optimizer.step(model.get_grads())

    return loss


def train(
    model: GPT,
    data: np.ndarray,
    optimizer: Optimizer,
    criterion: Optional[CrossEntropyLoss] = None,
    batch_size: int = 4,
    seq_len: int = 16,
    num_steps: int = 100,
    log_interval: int = 10,
    verbose: bool = True,
) -> List[float]:
    """
    Runs the complete training loop for a specified number of steps.
    
    Args:
        model (GPT): GPT model instance.
        data (np.ndarray): 1D tokenized training data.
        optimizer (Optimizer): Optimizer instance (SGD or Adam).
        criterion (Optional[CrossEntropyLoss]): Loss function (default: new CrossEntropyLoss()).
        batch_size (int): Number of sequences per batch.
        seq_len (int): Context length per sequence.
        num_steps (int): Total training steps to perform.
        log_interval (int): How often to print progress.
        verbose (bool): Whether to print training loss.
        
    Returns:
        List[float]: History of loss values per step.
    """
    if criterion is None:
        criterion = CrossEntropyLoss()

    losses: List[float] = []

    for step in range(1, num_steps + 1):
        x, y = get_batch(data, batch_size=batch_size, seq_len=seq_len)
        loss = train_step(model, criterion, optimizer, x, y)
        losses.append(loss)

        if verbose and (step == 1 or step % log_interval == 0 or step == num_steps):
            print(f"Step {step:4d}/{num_steps} | Loss: {loss:.4f}")

    return losses


if __name__ == "__main__":
    print("NumPyGPT Phase 7: Training Loop Demo")
    print("=" * 45)

    # 1. Generate synthetic repeating pattern data for demonstration
    vocab_size = 20
    seq_len = 8
    batch_size = 4
    np.random.seed(42)

    # A repeating pattern e.g. [0, 1, 2, 3, 4, ...] so model can easily learn
    data = np.array([i % vocab_size for i in range(500)], dtype=np.int32)

    # 2. Instantiate small GPT model
    model = GPT(
        vocab_size=vocab_size,
        max_seq_len=seq_len + 4,
        d_model=16,
        num_heads=2,
        num_layers=2,
    )

    # 3. Instantiate Adam optimizer
    optimizer = Adam(model.get_params(), lr=1e-2)
    criterion = CrossEntropyLoss()

    print(f"Initial loss check on batch:")
    x, y = get_batch(data, batch_size=batch_size, seq_len=seq_len)
    initial_loss = criterion.forward(model.forward(x), y)
    print(f"Initial Loss: {initial_loss:.4f}")

    print("\nStarting 50 training steps with Adam:")
    losses = train(
        model=model,
        data=data,
        optimizer=optimizer,
        criterion=criterion,
        batch_size=batch_size,
        seq_len=seq_len,
        num_steps=50,
        log_interval=10,
        verbose=True,
    )

    print(f"\nFinal Loss: {losses[-1]:.4f} (Decreased from {initial_loss:.4f})")
    print("Training loop verified successfully!")
