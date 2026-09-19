"""
Transformer Block Components for NumPyGPT.

This module implements:
1. Layer Normalization (LayerNorm) with learnable scale (gamma) and shift (beta).
2. Gaussian Error Linear Unit (GELU) activation function and its analytical derivative.
3. Position-wise Feed-Forward Network (FFN) with two linear projections.
4. Pre-Norm TransformerBlock combining Multi-Head Attention, FFN, and Residual Connections.
"""

from typing import Dict, Optional, Tuple
import numpy as np
from attention import MultiHeadAttention


def gelu(x: np.ndarray) -> np.ndarray:
    """
    Gaussian Error Linear Unit (GELU) activation function.
    
    Approximation formula (Hendrycks & Gimpel, 2016):
        GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        
    Args:
        x (np.ndarray): Input tensor of any shape.
        
    Returns:
        np.ndarray: Activated tensor of the same shape.
    """
    c = np.sqrt(2.0 / np.pi)
    u = c * (x + 0.044715 * np.power(x, 3))
    return 0.5 * x * (1.0 + np.tanh(u))


def gelu_backward(x: np.ndarray, dout: np.ndarray) -> np.ndarray:
    """
    Exact analytical gradient of the GELU activation function.
    
    Formula:
        d/dx GELU(x) = 0.5 * (1 + tanh(u)) + 0.5 * x * (1 - tanh^2(u)) * du/dx
        where u = sqrt(2/pi) * (x + 0.044715 * x^3)
              du/dx = sqrt(2/pi) * (1 + 3 * 0.044715 * x^2)
              
    Args:
        x (np.ndarray): Original forward input tensor.
        dout (np.ndarray): Upstream gradient of same shape.
        
    Returns:
        np.ndarray: Gradient with respect to x.
    """
    c = np.sqrt(2.0 / np.pi)
    u = c * (x + 0.044715 * np.power(x, 3))
    tanh_u = np.tanh(u)
    du_dx = c * (1.0 + 3.0 * 0.044715 * np.power(x, 2))
    dgelu_dx = 0.5 * (1.0 + tanh_u) + 0.5 * x * (1.0 - np.power(tanh_u, 2)) * du_dx
    return dout * dgelu_dx


class LayerNorm:
    """
    Layer Normalization across the hidden feature dimension.
    
    Normalizes each token vector to zero mean and unit variance, then applies
    a learnable element-wise scale (gamma) and shift (beta).
    
    Attributes:
        d_model (int): Hidden representation dimension (D).
        eps (float): Numerical stability constant added to variance.
        gamma (np.ndarray): Learnable scale parameter of shape (D,), initialized to 1.0.
        beta (np.ndarray): Learnable shift parameter of shape (D,), initialized to 0.0.
        dgamma (np.ndarray): Gradient with respect to gamma.
        dbeta (np.ndarray): Gradient with respect to beta.
    """

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        self.d_model = d_model
        self.eps = eps

        # Learnable scale (gamma) and shift (beta)
        self.gamma: np.ndarray = np.ones(d_model)
        self.beta: np.ndarray = np.zeros(d_model)

        # Gradients
        self.dgamma: Optional[np.ndarray] = None
        self.dbeta: Optional[np.ndarray] = None

        # Cached intermediate values for backward pass
        self.x: Optional[np.ndarray] = None
        self.x_hat: Optional[np.ndarray] = None
        self.std_inv: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass for Layer Normalization.
        
        Formula:
            mean = (1/D) * sum(x)
            var  = (1/D) * sum((x - mean)^2)
            x_hat = (x - mean) / sqrt(var + eps)
            y = gamma * x_hat + beta
            
        Args:
            x (np.ndarray): Input tensor of shape (B, T, D).
            
        Returns:
            np.ndarray: Normalized tensor of shape (B, T, D).
        """
        self.x = x
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        self.std_inv = 1.0 / np.sqrt(var + self.eps)
        self.x_hat = (x - mean) * self.std_inv
        return self.gamma * self.x_hat + self.beta

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Exact analytical backward pass for Layer Normalization.
        
        Formula:
            dgamma = sum_{B, T} (dout * x_hat)
            dbeta  = sum_{B, T} (dout)
            dx = (std_inv / D) * [ D * dx_hat - sum(dx_hat) - x_hat * sum(dx_hat * x_hat) ]
            where dx_hat = dout * gamma
            
        Args:
            dout (np.ndarray): Upstream gradient of shape (B, T, D).
            
        Returns:
            np.ndarray: Gradient with respect to input x of shape (B, T, D).
        """
        if self.x_hat is None or self.std_inv is None:
            raise RuntimeError("Cannot call backward before forward.")

        # Gradients for learnable parameters
        self.dgamma = np.sum(dout * self.x_hat, axis=(0, 1))
        self.dbeta = np.sum(dout, axis=(0, 1))

        # Gradient for normalized input
        dx_hat = dout * self.gamma
        D = self.d_model

        # Gradient with respect to x
        dx = (self.std_inv / D) * (
            D * dx_hat
            - np.sum(dx_hat, axis=-1, keepdims=True)
            - self.x_hat * np.sum(dx_hat * self.x_hat, axis=-1, keepdims=True)
        )
        return dx

    def get_params(self) -> Dict[str, np.ndarray]:
        """Return dictionary of learnable parameters."""
        return {"gamma": self.gamma, "beta": self.beta}

    def get_grads(self) -> Dict[str, np.ndarray]:
        """Return dictionary of parameter gradients."""
        if self.dgamma is None or self.dbeta is None:
            raise RuntimeError("Gradients not computed yet. Run backward() first.")
        return {"gamma": self.dgamma, "beta": self.dbeta}


class FeedForward:
    """
    Position-wise Feed-Forward Network (FFN).
    
    Two-layer MLP applied to each token vector independently:
        FFN(x) = W2 @ GELU(W1 @ x + b1) + b2
        
    Expands dimension from D to D_ff (typically 4 * D) and projects back to D.
    
    Attributes:
        d_model (int): Input and output dimension (D).
        d_ff (int): Inner hidden layer dimension (D_ff, default: 4 * D).
        W1 (np.ndarray): First linear layer weight of shape (D, D_ff).
        b1 (np.ndarray): First linear layer bias of shape (D_ff,).
        W2 (np.ndarray): Second linear layer weight of shape (D_ff, D).
        b2 (np.ndarray): Second linear layer bias of shape (D,).
    """

    def __init__(self, d_model: int, d_ff: Optional[int] = None, std: float = 0.02) -> None:
        self.d_model = d_model
        self.d_ff = d_ff or (4 * d_model)

        # Weight matrices initialized with Gaussian noise N(0, std^2)
        self.W1: np.ndarray = np.random.randn(self.d_model, self.d_ff) * std
        self.b1: np.ndarray = np.zeros(self.d_ff)

        self.W2: np.ndarray = np.random.randn(self.d_ff, self.d_model) * std
        self.b2: np.ndarray = np.zeros(self.d_model)

        # Parameter gradients
        self.dW1: Optional[np.ndarray] = None
        self.db1: Optional[np.ndarray] = None
        self.dW2: Optional[np.ndarray] = None
        self.db2: Optional[np.ndarray] = None

        # Cached activations for backpropagation
        self.x: Optional[np.ndarray] = None
        self.h1: Optional[np.ndarray] = None
        self.a1: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass for the Feed-Forward Network.
        
        Args:
            x (np.ndarray): Input tensor of shape (B, T, D).
            
        Returns:
            np.ndarray: Output tensor of shape (B, T, D).
        """
        self.x = x
        # 1. First linear layer: (B, T, D) @ (D, D_ff) -> (B, T, D_ff)
        self.h1 = x @ self.W1 + self.b1
        # 2. GELU non-linear activation: (B, T, D_ff)
        self.a1 = gelu(self.h1)
        # 3. Second linear layer: (B, T, D_ff) @ (D_ff, D) -> (B, T, D)
        out = self.a1 @ self.W2 + self.b2
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Backward pass for the Feed-Forward Network.
        
        Args:
            dout (np.ndarray): Upstream gradient of shape (B, T, D).
            
        Returns:
            np.ndarray: Gradient with respect to input x of shape (B, T, D).
        """
        if self.x is None or self.h1 is None or self.a1 is None:
            raise RuntimeError("Cannot call backward before forward.")

        D = self.d_model
        D_ff = self.d_ff

        # 1. Gradients for W2 and b2: out = a1 @ W2 + b2
        self.dW2 = self.a1.reshape(-1, D_ff).T @ dout.reshape(-1, D)
        self.db2 = np.sum(dout, axis=(0, 1))

        # 2. Backprop into GELU activation
        da1 = dout @ self.W2.T  # (B, T, D_ff)
        dh1 = gelu_backward(self.h1, da1)  # (B, T, D_ff)

        # 3. Gradients for W1 and b1: h1 = x @ W1 + b1
        self.dW1 = self.x.reshape(-1, D).T @ dh1.reshape(-1, D_ff)
        self.db1 = np.sum(dh1, axis=(0, 1))

        # 4. Gradient with respect to input x
        dx = dh1 @ self.W1.T  # (B, T, D)
        return dx

    def get_params(self) -> Dict[str, np.ndarray]:
        """Return dictionary of learnable parameters."""
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    def get_grads(self) -> Dict[str, np.ndarray]:
        """Return dictionary of parameter gradients."""
        if self.dW1 is None or self.dW2 is None:
            raise RuntimeError("Gradients not computed yet. Run backward() first.")
        return {"W1": self.dW1, "b1": self.db1, "W2": self.dW2, "b2": self.db2}


class TransformerBlock:
    """
    Pre-Norm Decoder-Only Transformer Block.
    
    Structure:
        x1 = x + Attention(LayerNorm1(x))
        x2 = x1 + FFN(LayerNorm2(x1))
        
    Attributes:
        ln1 (LayerNorm): First Layer Normalization layer (pre-attention).
        attn (MultiHeadAttention): Causal Multi-Head Self-Attention layer.
        ln2 (LayerNorm): Second Layer Normalization layer (pre-FFN).
        ffn (FeedForward): Position-wise Feed-Forward Network.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: Optional[int] = None, std: float = 0.02) -> None:
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff or (4 * d_model)

        self.ln1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, std=std)
        self.ln2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, self.d_ff, std=std)

        # Cached intermediate activations for backpropagation
        self.x: Optional[np.ndarray] = None
        self.x1: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass for the Transformer Block.
        
        Args:
            x (np.ndarray): Input tensor of shape (B, T, D).
            
        Returns:
            np.ndarray: Output tensor of shape (B, T, D).
        """
        self.x = x

        # 1. Pre-Norm Attention with Residual Connection: x1 = x + Attn(LN1(x))
        ln1_out = self.ln1.forward(x)
        attn_out = self.attn.forward(ln1_out)
        self.x1 = x + attn_out

        # 2. Pre-Norm FFN with Residual Connection: x2 = x1 + FFN(LN2(x1))
        ln2_out = self.ln2.forward(self.x1)
        ffn_out = self.ffn.forward(ln2_out)
        x2 = self.x1 + ffn_out

        return x2

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Backward pass for the Transformer Block.
        
        Propagates gradients through the two residual branches:
            dx1 = dout + LN2_back(FFN_back(dout))
            dx  = dx1  + LN1_back(Attn_back(dx1))
            
        Args:
            dout (np.ndarray): Upstream gradient of shape (B, T, D).
            
        Returns:
            np.ndarray: Gradient with respect to input x of shape (B, T, D).
        """
        if self.x is None or self.x1 is None:
            raise RuntimeError("Cannot call backward before forward.")

        # Residual 2 branch: x2 = x1 + FFN(LN2(x1))
        # dout flows directly to x1 and to FFN
        dx1 = dout.copy()
        dffn = self.ffn.backward(dout)
        dln2 = self.ln2.backward(dffn)
        dx1 += dln2

        # Residual 1 branch: x1 = x + Attn(LN1(x))
        # dx1 flows directly to x and to Attention
        dx = dx1.copy()
        dattn = self.attn.backward(dx1)
        dln1 = self.ln1.backward(dattn)
        dx += dln1

        return dx

    def get_params(self) -> Dict[str, np.ndarray]:
        """Return dictionary of all learnable parameters in the block."""
        params = {}
        for name, p in self.ln1.get_params().items():
            params[f"ln1.{name}"] = p
        for name, p in self.attn.get_params().items():
            params[f"attn.{name}"] = p
        for name, p in self.ln2.get_params().items():
            params[f"ln2.{name}"] = p
        for name, p in self.ffn.get_params().items():
            params[f"ffn.{name}"] = p
        return params

    def get_grads(self) -> Dict[str, np.ndarray]:
        """Return dictionary of all parameter gradients in the block."""
        grads = {}
        for name, g in self.ln1.get_grads().items():
            grads[f"ln1.{name}"] = g
        for name, g in self.attn.get_grads().items():
            grads[f"attn.{name}"] = g
        for name, g in self.ln2.get_grads().items():
            grads[f"ln2.{name}"] = g
        for name, g in self.ffn.get_grads().items():
            grads[f"ffn.{name}"] = g
        return grads
