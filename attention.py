"""
Attention Module for NumPyGPT.

This module implements:
1. Causal autoregressive masking.
2. Scaled dot-product self-attention.
3. Multi-head self-attention with learned linear projections (Q, K, V, O).
4. Exact analytical backward pass computing gradients for all parameters and inputs.
"""

from typing import Dict, Optional, Tuple
import numpy as np


def causal_mask(seq_len: int) -> np.ndarray:
    """
    Generate an upper-triangular causal attention mask.
    
    Positions where j > i are set to -1e9 so that exp(score) -> 0 in softmax.
    Positions where j <= i are set to 0.0 (no change to dot-product score).
    
    Args:
        seq_len (int): Sequence length (T).
        
    Returns:
        np.ndarray: Causal mask of shape (T, T).
    """
    # np.triu with k=1 selects elements strictly above the main diagonal
    return np.triu(np.full((seq_len, seq_len), -1e9), k=1)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Compute numerically stable softmax along the specified axis.
    
    Formula:
        softmax(z_i) = exp(z_i - max(z)) / sum_j exp(z_j - max(z))
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Scaled Dot-Product Attention.
    
    Formula:
        Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k) + mask) @ V
        
    Args:
        Q (np.ndarray): Query tensor of shape (..., T, d_k).
        K (np.ndarray): Key tensor of shape (..., T, d_k).
        V (np.ndarray): Value tensor of shape (..., T, d_k).
        mask (Optional[np.ndarray]): Optional additive mask of shape (T, T).
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (output_tensor, attention_weights)
            output_tensor has shape (..., T, d_k)
            attention_weights has shape (..., T, T)
    """
    d_k = Q.shape[-1]
    # Q @ K^T: (..., T, d_k) @ (..., d_k, T) -> (..., T, T)
    scores = np.matmul(Q, K.swapaxes(-1, -2)) / np.sqrt(d_k)

    if mask is not None:
        scores = scores + mask

    A = softmax(scores, axis=-1)
    # A @ V: (..., T, T) @ (..., T, d_k) -> (..., T, d_k)
    out = np.matmul(A, V)

    return out, A


class MultiHeadAttention:
    """
    Causal Multi-Head Self-Attention.
    
    Splits the hidden dimension d_model into num_heads parallel attention heads,
    computes causal scaled dot-product attention in each subspace, concatenates
    the head outputs, and projects back through a linear layer.
    
    Attributes:
        d_model (int): Hidden representation dimensionality (D).
        num_heads (int): Number of parallel attention heads (h).
        d_k (int): Dimensionality of each individual head: d_k = d_model / num_heads.
        W_q, W_k, W_v, W_o (np.ndarray): Projection weights of shape (D, D).
        b_q, b_k, b_v, b_o (np.ndarray): Projection biases of shape (D,).
        dW_q, dW_k, dW_v, dW_o (np.ndarray): Parameter gradients.
        db_q, db_k, db_v, db_o (np.ndarray): Bias gradients.
    """

    def __init__(self, d_model: int, num_heads: int, std: float = 0.02) -> None:
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections for Query, Key, Value, and Output
        self.W_q: np.ndarray = np.random.randn(d_model, d_model) * std
        self.b_q: np.ndarray = np.zeros(d_model)

        self.W_k: np.ndarray = np.random.randn(d_model, d_model) * std
        self.b_k: np.ndarray = np.zeros(d_model)

        self.W_v: np.ndarray = np.random.randn(d_model, d_model) * std
        self.b_v: np.ndarray = np.zeros(d_model)

        self.W_o: np.ndarray = np.random.randn(d_model, d_model) * std
        self.b_o: np.ndarray = np.zeros(d_model)

        # Gradients
        self.dW_q: Optional[np.ndarray] = None
        self.db_q: Optional[np.ndarray] = None
        self.dW_k: Optional[np.ndarray] = None
        self.db_k: Optional[np.ndarray] = None
        self.dW_v: Optional[np.ndarray] = None
        self.db_v: Optional[np.ndarray] = None
        self.dW_o: Optional[np.ndarray] = None
        self.db_o: Optional[np.ndarray] = None

        # Cached forward activations for backpropagation
        self.X: Optional[np.ndarray] = None
        self.Q: Optional[np.ndarray] = None
        self.K: Optional[np.ndarray] = None
        self.V: Optional[np.ndarray] = None
        self.A: Optional[np.ndarray] = None
        self.out_concat: Optional[np.ndarray] = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass for Causal Multi-Head Self-Attention.
        
        Args:
            X (np.ndarray): Input representations of shape (B, T, D).
            
        Returns:
            np.ndarray: Attended representations of shape (B, T, D).
        """
        B, T, D = X.shape
        self.X = X

        # 1. Project to Q, K, V: (B, T, D) @ (D, D) -> (B, T, D)
        Q = X @ self.W_q + self.b_q
        K = X @ self.W_k + self.b_k
        V = X @ self.W_v + self.b_v

        # 2. Reshape and transpose to split heads: (B, T, D) -> (B, h, T, d_k)
        self.Q = Q.reshape(B, T, self.num_heads, self.d_k).swapaxes(1, 2)
        self.K = K.reshape(B, T, self.num_heads, self.d_k).swapaxes(1, 2)
        self.V = V.reshape(B, T, self.num_heads, self.d_k).swapaxes(1, 2)

        # 3. Scaled dot-product attention with causal mask
        mask = causal_mask(T)
        out_heads, self.A = scaled_dot_product_attention(
            self.Q, self.K, self.V, mask=mask
        )  # out_heads: (B, h, T, d_k), A: (B, h, T, T)

        # 4. Concatenate heads: (B, h, T, d_k) -> (B, T, h, d_k) -> (B, T, D)
        self.out_concat = out_heads.swapaxes(1, 2).reshape(B, T, D)

        # 5. Output linear projection: (B, T, D) @ (D, D) -> (B, T, D)
        out = self.out_concat @ self.W_o + self.b_o

        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Backward pass: computes parameter gradients and propagates error to input X.
        
        Args:
            dout (np.ndarray): Upstream gradient of shape (B, T, D).
            
        Returns:
            np.ndarray: Gradient with respect to input X of shape (B, T, D).
        """
        if self.X is None or self.Q is None or self.K is None or self.V is None or self.A is None or self.out_concat is None:
            raise RuntimeError("Cannot call backward before forward.")

        B, T, D = self.X.shape

        # 1. Output projection backward
        # out = out_concat @ W_o + b_o
        self.dW_o = self.out_concat.reshape(-1, D).T @ dout.reshape(-1, D)
        self.db_o = np.sum(dout, axis=(0, 1))
        dout_concat = dout @ self.W_o.T  # (B, T, D)

        # 2. Reshape back to heads: (B, T, D) -> (B, h, T, d_k)
        dout_heads = dout_concat.reshape(B, T, self.num_heads, self.d_k).swapaxes(1, 2)

        # 3. Scaled dot-product attention backward:
        # out_heads = A @ V
        # dV = A^T @ dout_heads
        # dA = dout_heads @ V^T
        dV = np.matmul(self.A.swapaxes(-1, -2), dout_heads)  # (B, h, T, d_k)
        dA = np.matmul(dout_heads, self.V.swapaxes(-1, -2))  # (B, h, T, T)

        # 4. Softmax backward:
        # dS = A * (dA - sum(dA * A, axis=-1, keepdims=True))
        # Note: for masked positions, A == 0, so dS automatically becomes 0!
        dS = self.A * (dA - np.sum(dA * self.A, axis=-1, keepdims=True))

        # 5. Scaled dot-product score backward: S = Q @ K^T / sqrt(d_k)
        dS_scaled = dS / np.sqrt(self.d_k)
        dQ = np.matmul(dS_scaled, self.K)  # (B, h, T, d_k)
        dK = np.matmul(dS_scaled.swapaxes(-1, -2), self.Q)  # (B, h, T, d_k)

        # 6. Reshape Q, K, V gradients back: (B, h, T, d_k) -> (B, T, D)
        dQ = dQ.swapaxes(1, 2).reshape(B, T, D)
        dK = dK.swapaxes(1, 2).reshape(B, T, D)
        dV = dV.swapaxes(1, 2).reshape(B, T, D)

        # 7. Linear projections backward:
        # Q = X @ W_q + b_q, K = X @ W_k + b_k, V = X @ W_v + b_v
        X_flat = self.X.reshape(-1, D)
        self.dW_q = X_flat.T @ dQ.reshape(-1, D)
        self.db_q = np.sum(dQ, axis=(0, 1))

        self.dW_k = X_flat.T @ dK.reshape(-1, D)
        self.db_k = np.sum(dK, axis=(0, 1))

        self.dW_v = X_flat.T @ dV.reshape(-1, D)
        self.db_v = np.sum(dV, axis=(0, 1))

        # 8. Gradient with respect to input representation X:
        dX = dQ @ self.W_q.T + dK @ self.W_k.T + dV @ self.W_v.T

        return dX

    def get_params(self) -> Dict[str, np.ndarray]:
        """Return dictionary of trainable weight and bias parameters."""
        return {
            "W_q": self.W_q,
            "b_q": self.b_q,
            "W_k": self.W_k,
            "b_k": self.b_k,
            "W_v": self.W_v,
            "b_v": self.b_v,
            "W_o": self.W_o,
            "b_o": self.b_o,
        }

    def get_grads(self) -> Dict[str, np.ndarray]:
        """Return dictionary of parameter gradients."""
        if self.dW_q is None:
            raise RuntimeError("Gradients have not been computed yet. Run backward() first.")
        return {
            "W_q": self.dW_q,
            "b_q": self.db_q,
            "W_k": self.dW_k,
            "b_k": self.db_k,
            "W_v": self.dW_v,
            "b_v": self.db_v,
            "W_o": self.dW_o,
            "b_o": self.db_o,
        }
