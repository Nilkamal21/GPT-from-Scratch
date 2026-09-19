"""
Loss Module for NumPyGPT (Phase 6).

This module implements:
1. Numerically stable Softmax across vocabulary dimension.
2. Cross-Entropy Loss with mean reduction over all tokens in the batch.
3. Analytical gradient dLogits = (Probabilities - Targets) / (B * T).
4. Connection to GPT.backward(dlogits) for end-to-end backpropagation.
"""

from typing import Optional, Tuple
import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Numerically stable Softmax function.
    
    Subtracts the maximum value along the specified axis before exponentiating
    to prevent numerical overflow with large logits.
    
    Formula:
        softmax(z_i) = exp(z_i - max(z)) / sum_j exp(z_j - max(z))
        
    Args:
        x (np.ndarray): Input tensor of any shape (typically (B, T, V)).
        axis (int): Axis along which to compute softmax (default: -1).
        
    Returns:
        np.ndarray: Probability distribution of same shape as x, summing to 1.0 along axis.
    """
    # Subtract max for numerical stability: max(x) becomes 0, exp(0) = 1, prevents overflow to +inf
    max_x = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - max_x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def cross_entropy_loss(
    logits: np.ndarray,
    targets: np.ndarray,
    eps: float = 1e-15,
) -> Tuple[float, np.ndarray]:
    """
    Computes mean Cross-Entropy Loss and analytical gradient dLogits.
    
    Args:
        logits (np.ndarray): Unnormalized model predictions of shape (B, T, V).
        targets (np.ndarray): Ground-truth target token IDs of shape (B, T),
            with integer values in range [0, V - 1].
        eps (float): Small constant to avoid log(0) numerical underflow.
        
    Returns:
        Tuple[float, np.ndarray]:
            - loss: Scalar mean cross-entropy loss over all B * T tokens.
            - dlogits: Analytical gradient dLoss/dLogits of shape (B, T, V).
    """
    if logits.ndim != 3:
        raise ValueError(f"Expected 3D logits of shape (B, T, V), got shape {logits.shape}")
    if targets.ndim != 2:
        raise ValueError(f"Expected 2D targets of shape (B, T), got shape {targets.shape}")

    B, T, V = logits.shape
    if targets.shape != (B, T):
        raise ValueError(
            f"Targets shape {targets.shape} does not match logits batch/seq shape ({B}, {T})"
        )
    if np.any(targets < 0) or np.any(targets >= V):
        raise ValueError(f"Target token IDs must be in range [0, {V - 1}]")

    # 1. Compute probabilities using numerically stable softmax: (B, T, V)
    probs = softmax(logits, axis=-1)

    # 2. Extract probability of true target tokens using advanced indexing
    b_idx = np.arange(B)[:, None]
    t_idx = np.arange(T)[None, :]
    target_probs = probs[b_idx, t_idx, targets]  # Shape: (B, T)

    # 3. Mean negative log-likelihood over all B * T tokens
    loss = float(-np.mean(np.log(np.clip(target_probs, eps, 1.0))))

    # 4. Analytical gradient dLogits = (probs - targets_one_hot) / (B * T)
    dlogits = probs.copy()
    dlogits[b_idx, t_idx, targets] -= 1.0
    dlogits /= (B * T)

    return loss, dlogits


class CrossEntropyLoss:
    """
    Cross-Entropy Loss Layer with state caching for forward/backward passes.
    
    Attributes:
        eps (float): Small epsilon for numerical stability in log.
        probs (Optional[np.ndarray]): Cached softmax probabilities of shape (B, T, V).
        targets (Optional[np.ndarray]): Cached ground-truth target IDs of shape (B, T).
        dlogits (Optional[np.ndarray]): Cached gradients of shape (B, T, V).
    """

    def __init__(self, eps: float = 1e-15) -> None:
        self.eps = eps
        self.probs: Optional[np.ndarray] = None
        self.targets: Optional[np.ndarray] = None
        self.dlogits: Optional[np.ndarray] = None

    def forward(self, logits: np.ndarray, targets: np.ndarray) -> float:
        """
        Computes forward cross-entropy loss and caches intermediate states.
        
        Args:
            logits (np.ndarray): Logits tensor of shape (B, T, V).
            targets (np.ndarray): Integer target token IDs of shape (B, T).
            
        Returns:
            float: Scalar mean loss.
        """
        loss, dlogits = cross_entropy_loss(logits, targets, eps=self.eps)
        self.probs = softmax(logits, axis=-1)
        self.targets = targets
        self.dlogits = dlogits
        return loss

    def backward(self) -> np.ndarray:
        """
        Returns cached analytical gradient dLogits.
        
        Returns:
            np.ndarray: Gradient tensor of shape (B, T, V).
        """
        if self.dlogits is None:
            raise RuntimeError("Cannot call backward() before forward().")
        return self.dlogits
