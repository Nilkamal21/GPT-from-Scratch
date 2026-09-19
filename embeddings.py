"""
Embeddings Module for NumPyGPT.

This module implements:
1. Learnable token embeddings (lookup table from token ID to vector).
2. Learnable positional embeddings (lookup table from sequence position to vector).
3. Forward pass combining token and positional representations.
4. Exact analytical backward pass computing parameter gradients via chain rule.
"""

from typing import Dict, Optional
import numpy as np


class Embedding:
    """
    Combined Token and Positional Embedding layer.
    
    Attributes:
        vocab_size (int): Total number of unique tokens in the vocabulary (V).
        max_seq_len (int): Maximum sequence context length (T_max).
        d_model (int): Dimensionality of the embedding vectors (D).
        W_e (np.ndarray): Learnable token embedding matrix of shape (V, D).
        W_p (np.ndarray): Learnable positional embedding matrix of shape (T_max, D).
        dW_e (np.ndarray): Gradient of loss with respect to W_e of shape (V, D).
        dW_p (np.ndarray): Gradient of loss with respect to W_p of shape (T_max, D).
    """

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int,
        std: float = 0.02,
    ) -> None:
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model

        # Initialize weights with Gaussian noise: N(0, std^2)
        # Using std = 0.02 is the standard convention established by GPT-2
        self.W_e: np.ndarray = np.random.randn(vocab_size, d_model) * std
        self.W_p: np.ndarray = np.random.randn(max_seq_len, d_model) * std

        # Gradients accumulated during backward pass
        self.dW_e: Optional[np.ndarray] = None
        self.dW_p: Optional[np.ndarray] = None

        # Cached inputs for backpropagation
        self.token_ids: Optional[np.ndarray] = None

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """
        Forward pass: retrieves and sums token and positional embeddings.
        
        Args:
            token_ids (np.ndarray): 2D integer array of shape (B, T) where
                B is batch size and T is sequence length (T <= max_seq_len).
                
        Returns:
            np.ndarray: Continuous representation tensor of shape (B, T, D).
        """
        if token_ids.ndim != 2:
            raise ValueError(f"Expected 2D array of shape (B, T), got shape {token_ids.shape}")

        B, T = token_ids.shape
        if T > self.max_seq_len:
            raise ValueError(
                f"Sequence length T={T} exceeds max_seq_len={self.max_seq_len}"
            )

        # 1. Token embedding lookup: (B, T) -> (B, T, D)
        X_tok = self.W_e[token_ids]

        # 2. Positional embedding lookup: positions 0..T-1 -> (T, D)
        X_pos = self.W_p[:T]

        # 3. Element-wise addition (X_pos broadcasts across batch dimension B)
        X_out = X_tok + X_pos

        # Cache input for backward pass
        self.token_ids = token_ids

        return X_out

    def backward(self, dout: np.ndarray) -> None:
        """
        Backward pass: computes exact analytical gradients dW_e and dW_p.
        
        Args:
            dout (np.ndarray): Upstream gradient of shape (B, T, D).
        """
        if self.token_ids is None:
            raise RuntimeError("Cannot run backward before calling forward.")
        if dout.shape[:2] != self.token_ids.shape or dout.shape[2] != self.d_model:
            raise ValueError(
                f"dout shape {dout.shape} does not match expected ({self.token_ids.shape[0]}, {self.token_ids.shape[1]}, {self.d_model})"
            )

        B, T = self.token_ids.shape

        # 1. Gradient with respect to W_p: sum upstream gradients across batch dimension
        # dL/dW_p[t, :] = sum_b dL/dX_out[b, t, :]
        self.dW_p = np.zeros_like(self.W_p)
        self.dW_p[:T] = np.sum(dout, axis=0)

        # 2. Gradient with respect to W_e: accumulate gradients for each token ID
        # Must use np.add.at to safely handle duplicate token IDs within a batch
        self.dW_e = np.zeros_like(self.W_e)
        np.add.at(self.dW_e, self.token_ids, dout)

    def get_params(self) -> Dict[str, np.ndarray]:
        """Return dictionary of trainable weight parameters."""
        return {"W_e": self.W_e, "W_p": self.W_p}

    def get_grads(self) -> Dict[str, np.ndarray]:
        """Return dictionary of gradients for trainable parameters."""
        if self.dW_e is None or self.dW_p is None:
            raise RuntimeError("Gradients have not been computed yet. Run backward() first.")
        return {"W_e": self.dW_e, "W_p": self.dW_p}
