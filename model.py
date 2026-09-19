"""
Complete GPT Model Architecture for NumPyGPT (Phase 5).

This module connects:
1. Embedding: Token + Positional embeddings (B, T) -> (B, T, D)
2. N x TransformerBlock: Pre-Norm Self-Attention + FFN with Residual Connections
3. Final LayerNorm: Normalizes residual stream activations before LM Head
4. LM Head: Linear projection (B, T, D) -> (B, T, V) computing unnormalized next-token logits

Pure NumPy implementation supporting:
- forward(token_ids) -> logits
- backward(dlogits) -> propagates analytical gradients through all components
- get_params() / get_grads() for optimization
"""

from typing import Dict, List, Optional
import numpy as np
from embeddings import Embedding
from transformer import LayerNorm, TransformerBlock


class GPT:
    """
    Decoder-only Generative Pre-trained Transformer (GPT) language model.
    
    Architecture:
        token_ids: (B, T)
            ↓
        Embedding (Token W_e + Positional W_p)
            ↓
        TransformerBlock 1
            ↓
        TransformerBlock 2
            ↓
        ...
            ↓
        TransformerBlock N (num_layers)
            ↓
        Final LayerNorm (ln_f)
            ↓
        LM Head (Linear: W_vocab, b_vocab)
            ↓
        Logits: (B, T, vocab_size)
        
    Attributes:
        vocab_size (int): Size of the token vocabulary (V).
        max_seq_len (int): Maximum sequence length supported by positional embeddings (T_max).
        d_model (int): Hidden dimension / embedding size (D).
        num_heads (int): Number of attention heads (h).
        num_layers (int): Number of stacked Transformer blocks (N).
        d_ff (int): Inner dimension of the feed-forward network (D_ff).
        token_embeddings (Embedding): Combined token and positional embedding layer.
        blocks (List[TransformerBlock]): List of stacked Transformer blocks.
        ln_f (LayerNorm): Final Layer Normalization before LM projection.
        W_vocab (np.ndarray): LM Head projection weights of shape (D, V).
        b_vocab (np.ndarray): LM Head projection bias of shape (V,).
        dW_vocab (np.ndarray): Gradient with respect to W_vocab.
        db_vocab (np.ndarray): Gradient with respect to b_vocab.
    """

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: Optional[int] = None,
        std: float = 0.02,
    ) -> None:
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff or (4 * d_model)
        self.std = std

        # 1. Token & Positional Embeddings
        self.token_embeddings = Embedding(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            d_model=d_model,
            std=std,
        )

        # 2. Stack of N Transformer Blocks
        self.blocks: List[TransformerBlock] = [
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=self.d_ff,
                std=std,
            )
            for _ in range(num_layers)
        ]

        # 3. Final Layer Normalization (with separate gamma and beta)
        self.ln_f = LayerNorm(d_model=d_model)

        # 4. Language Modeling Head (Linear projection to vocabulary)
        # Initialized with Gaussian noise N(0, std^2)
        self.W_vocab: np.ndarray = np.random.randn(d_model, vocab_size) * std
        self.b_vocab: np.ndarray = np.zeros(vocab_size)

        # Parameter gradients for LM head
        self.dW_vocab: Optional[np.ndarray] = None
        self.db_vocab: Optional[np.ndarray] = None

        # Cached activations for backpropagation
        self.token_ids: Optional[np.ndarray] = None
        self.x_norm: Optional[np.ndarray] = None

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """
        Forward pass for the complete GPT model.
        
        Maps input token IDs to unnormalized next-token prediction logits.
        
        Args:
            token_ids (np.ndarray): 2D integer array of shape (B, T) where
                B is batch size and T is sequence length (T <= max_seq_len).
                
        Returns:
            np.ndarray: Logits tensor of shape (B, T, vocab_size).
        """
        if token_ids.ndim != 2:
            raise ValueError(f"Expected 2D token_ids of shape (B, T), got {token_ids.shape}")

        B, T = token_ids.shape
        if T > self.max_seq_len:
            raise ValueError(
                f"Sequence length T={T} exceeds maximum allowed context length {self.max_seq_len}"
            )

        self.token_ids = token_ids

        # 1. Embeddings: (B, T) -> (B, T, D)
        x = self.token_embeddings.forward(token_ids)

        # 2. Sequential Transformer Blocks: (B, T, D) -> (B, T, D)
        for block in self.blocks:
            x = block.forward(x)

        # 3. Final Layer Normalization: (B, T, D) -> (B, T, D)
        self.x_norm = self.ln_f.forward(x)

        # 4. LM Head projection: (B, T, D) @ (D, V) + (V,) -> (B, T, V)
        logits = self.x_norm @ self.W_vocab + self.b_vocab

        return logits

    def backward(self, dlogits: np.ndarray) -> None:
        """
        Analytical backward pass for the complete GPT model.
        
        Propagates gradients backward from logits through:
        LM Head -> Final LayerNorm -> Transformer Blocks (in reverse) -> Embeddings.
        
        Args:
            dlogits (np.ndarray): Upstream gradient of loss with respect to logits,
                of shape (B, T, vocab_size).
        """
        if self.token_ids is None or self.x_norm is None:
            raise RuntimeError("Cannot run backward before calling forward.")

        B, T, V = dlogits.shape
        D = self.d_model

        if V != self.vocab_size:
            raise ValueError(
                f"dlogits vocab dimension {V} does not match model vocab_size {self.vocab_size}"
            )

        # 1. Gradients for LM Head: logits = x_norm @ W_vocab + b_vocab
        # dW_vocab = x_norm^T @ dlogits: (D, B*T) @ (B*T, V) -> (D, V)
        self.dW_vocab = self.x_norm.reshape(-1, D).T @ dlogits.reshape(-1, V)
        # db_vocab = sum across batch and sequence dimensions: (V,)
        self.db_vocab = np.sum(dlogits, axis=(0, 1))

        # Backpropagate gradient into normalized activations x_norm: (B, T, D)
        dx_norm = dlogits @ self.W_vocab.T

        # 2. Backpropagate through Final Layer Normalization: (B, T, D)
        dx = self.ln_f.backward(dx_norm)

        # 3. Backpropagate through Transformer Blocks in reverse order
        for block in reversed(self.blocks):
            dx = block.backward(dx)

        # 4. Backpropagate into Token and Positional Embeddings
        self.token_embeddings.backward(dx)

    def get_params(self) -> Dict[str, np.ndarray]:
        """
        Return a dictionary of all learnable parameters in the model.
        
        Returns:
            Dict[str, np.ndarray]: Mapping of parameter names to weight arrays.
        """
        params: Dict[str, np.ndarray] = {}

        # Embeddings
        for name, p in self.token_embeddings.get_params().items():
            params[f"token_embeddings.{name}"] = p

        # Transformer Blocks
        for i, block in enumerate(self.blocks):
            for name, p in block.get_params().items():
                params[f"blocks.{i}.{name}"] = p

        # Final LayerNorm
        for name, p in self.ln_f.get_params().items():
            params[f"ln_f.{name}"] = p

        # LM Head
        params["lm_head.W_vocab"] = self.W_vocab
        params["lm_head.b_vocab"] = self.b_vocab

        return params

    def get_grads(self) -> Dict[str, np.ndarray]:
        """
        Return a dictionary of all parameter gradients matching get_params().
        
        Returns:
            Dict[str, np.ndarray]: Mapping of parameter names to gradient arrays.
        """
        if self.dW_vocab is None or self.db_vocab is None:
            raise RuntimeError("Gradients have not been computed yet. Run backward() first.")

        grads: Dict[str, np.ndarray] = {}

        # Embeddings
        for name, g in self.token_embeddings.get_grads().items():
            grads[f"token_embeddings.{name}"] = g

        # Transformer Blocks
        for i, block in enumerate(self.blocks):
            for name, g in block.get_grads().items():
                grads[f"blocks.{i}.{name}"] = g

        # Final LayerNorm
        for name, g in self.ln_f.get_grads().items():
            grads[f"ln_f.{name}"] = g

        # LM Head
        grads["lm_head.W_vocab"] = self.dW_vocab
        grads["lm_head.b_vocab"] = self.db_vocab

        return grads
