"""
Optimizer Module for NumPyGPT (Phase 7).

This module implements:
1. Base Optimizer interface.
2. Stochastic Gradient Descent (SGD) with optional momentum.
3. Adaptive Moment Estimation (Adam) optimizer with bias correction.

Supports all parameters and gradients returned by GPT.get_params() and GPT.get_grads().
"""

from typing import Dict, Optional
import numpy as np


class Optimizer:
    """
    Base Optimizer class for updating model parameters.
    
    Attributes:
        params (Dict[str, np.ndarray]): Dictionary of model parameter tensors.
        lr (float): Learning rate.
    """

    def __init__(self, params: Dict[str, np.ndarray], lr: float = 1e-3) -> None:
        self.params = params
        self.lr = lr

    def step(self, grads: Dict[str, np.ndarray]) -> None:
        """
        Updates parameters in-place using the provided gradients.
        
        Args:
            grads (Dict[str, np.ndarray]): Dictionary of parameter gradients.
        """
        raise NotImplementedError("Subclasses must implement step().")


class SGD(Optimizer):
    """
    Stochastic Gradient Descent (SGD) optimizer.
    
    Formula:
        Without momentum:
            W = W - lr * grad
        With momentum:
            v_t = momentum * v_{t-1} + grad
            W = W - lr * v_t
            
    Attributes:
        params (Dict[str, np.ndarray]): Model parameters.
        lr (float): Learning rate.
        momentum (float): Momentum factor (default: 0.0 for pure SGD).
        velocities (Dict[str, np.ndarray]): Velocity buffers for momentum.
    """

    def __init__(
        self,
        params: Dict[str, np.ndarray],
        lr: float = 1e-3,
        momentum: float = 0.0,
    ) -> None:
        super().__init__(params, lr)
        self.momentum = momentum
        self.velocities: Dict[str, np.ndarray] = {
            k: np.zeros_like(v) for k, v in params.items()
        }

    def step(self, grads: Dict[str, np.ndarray]) -> None:
        """
        Updates parameters in-place using SGD.
        
        Args:
            grads (Dict[str, np.ndarray]): Gradients matching self.params keys.
        """
        for k in self.params:
            if k not in grads or grads[k] is None:
                continue
            g = grads[k]

            if self.momentum > 0.0:
                self.velocities[k] = self.momentum * self.velocities[k] + g
                self.params[k] -= self.lr * self.velocities[k]
            else:
                self.params[k] -= self.lr * g


class Adam(Optimizer):
    """
    Adaptive Moment Estimation (Adam) optimizer.
    
    Maintains exponential moving averages of both the gradients (first moment m)
    and the uncentered variance / squared gradients (second moment v), with bias
    correction for initial steps.
    
    Formulas:
        m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        v_t = beta2 * v_{t-1} + (1 - beta2) * (g_t^2)
        m_hat_t = m_t / (1 - beta1^t)
        v_hat_t = v_t / (1 - beta2^t)
        W_t = W_{t-1} - lr * m_hat_t / (sqrt(v_hat_t) + eps)
        
    Attributes:
        params (Dict[str, np.ndarray]): Model parameters.
        lr (float): Learning rate (alpha).
        beta1 (float): Exponential decay rate for first moment (default: 0.9).
        beta2 (float): Exponential decay rate for second moment (default: 0.999).
        eps (float): Small constant to prevent division by zero (default: 1e-8).
        t (int): Current timestep counter (used for bias correction).
        m (Dict[str, np.ndarray]): First moment estimates for each parameter.
        v (Dict[str, np.ndarray]): Second moment estimates for each parameter.
    """

    def __init__(
        self,
        params: Dict[str, np.ndarray],
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0  # Timestep

        # First moment: m_t = E[g_t]
        self.m: Dict[str, np.ndarray] = {
            k: np.zeros_like(v) for k, v in params.items()
        }
        # Second moment: v_t = E[g_t^2]
        self.v: Dict[str, np.ndarray] = {
            k: np.zeros_like(v) for k, v in params.items()
        }

    def step(self, grads: Dict[str, np.ndarray]) -> None:
        """
        Updates parameters in-place using Adam.
        
        Args:
            grads (Dict[str, np.ndarray]): Gradients matching self.params keys.
        """
        self.t += 1

        for k in self.params:
            if k not in grads or grads[k] is None:
                continue

            g = grads[k]
            if self.weight_decay > 0.0:
                g = g + self.weight_decay * self.params[k]

            # 1. Update biased first moment estimate
            self.m[k] = self.beta1 * self.m[k] + (1.0 - self.beta1) * g

            # 2. Update biased second raw moment estimate
            self.v[k] = self.beta2 * self.v[k] + (1.0 - self.beta2) * np.square(g)

            # 3. Compute bias-corrected first moment estimate
            m_hat = self.m[k] / (1.0 - (self.beta1 ** self.t))

            # 4. Compute bias-corrected second raw moment estimate
            v_hat = self.v[k] / (1.0 - (self.beta2 ** self.t))

            # 5. Update parameter tensor in-place
            self.params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
