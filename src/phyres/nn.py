"""A minimal multilayer perceptron with manual backpropagation and Adam.

Implemented in pure NumPy so that the entire research pipeline is
dependency-free, deterministic under a seed, and auditable line by line:
there is no framework autograd to hide the mathematics. The network is a
tanh MLP trained with mean-squared error, which is all a residual dynamics
model needs.
"""

from __future__ import annotations

import numpy as np


class MLP:
    """Tanh MLP with MSE loss, trained by full-batch or minibatch Adam."""

    def __init__(
        self,
        in_dim: int,
        hidden: tuple[int, ...],
        out_dim: int,
        rng: np.random.Generator,
    ) -> None:
        self.rng = rng
        dims = [in_dim, *hidden, out_dim]
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            # Xavier/Glorot initialisation keeps tanh pre-activations in range.
            scale = np.sqrt(2.0 / (d_in + d_out))
            self.weights.append(rng.normal(0.0, scale, size=(d_in, d_out)))
            self.biases.append(np.zeros(d_out))
        self._adam_state: list[dict[str, np.ndarray]] | None = None
        self._adam_t = 0

    # ------------------------------------------------------------------ #
    # Forward / backward
    # ------------------------------------------------------------------ #
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Predict outputs for a batch x of shape (B, in_dim)."""
        h = x
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            h = np.tanh(h @ w + b)
        return h @ self.weights[-1] + self.biases[-1]

    def _forward_cached(self, x: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        activations = [x]
        h = x
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            h = np.tanh(h @ w + b)
            activations.append(h)
        out = h @ self.weights[-1] + self.biases[-1]
        return out, activations

    def _backward(
        self, x: np.ndarray, y: np.ndarray
    ) -> tuple[float, list[np.ndarray], list[np.ndarray]]:
        """MSE loss and gradients w.r.t. all weights and biases."""
        batch = x.shape[0]
        out, acts = self._forward_cached(x)
        err = out - y
        loss = float(np.mean(err**2))
        # dL/d(out) for mean over batch and output dims
        delta = (2.0 / (batch * y.shape[1])) * err
        grads_w: list[np.ndarray] = [np.zeros_like(w) for w in self.weights]
        grads_b: list[np.ndarray] = [np.zeros_like(b) for b in self.biases]
        grads_w[-1] = acts[-1].T @ delta
        grads_b[-1] = delta.sum(axis=0)
        for layer in range(len(self.weights) - 2, -1, -1):
            delta = (delta @ self.weights[layer + 1].T) * (1.0 - acts[layer + 1] ** 2)
            grads_w[layer] = acts[layer].T @ delta
            grads_b[layer] = delta.sum(axis=0)
        return loss, grads_w, grads_b

    # ------------------------------------------------------------------ #
    # Optimisation
    # ------------------------------------------------------------------ #
    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int = 400,
        batch_size: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
    ) -> list[float]:
        """Train with minibatch Adam; returns the per-epoch loss history."""
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        if self._adam_state is None:
            self._adam_state = [
                {
                    "mw": np.zeros_like(w),
                    "vw": np.zeros_like(w),
                    "mb": np.zeros_like(b),
                    "vb": np.zeros_like(b),
                }
                for w, b in zip(self.weights, self.biases)
            ]
        history: list[float] = []
        n = x.shape[0]
        for _ in range(epochs):
            perm = self.rng.permutation(n)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                loss, grads_w, grads_b = self._backward(x[idx], y[idx])
                epoch_loss += loss
                n_batches += 1
                self._adam_t += 1
                t = self._adam_t
                for layer, st in enumerate(self._adam_state):
                    gw = grads_w[layer] + weight_decay * self.weights[layer]
                    gb = grads_b[layer]
                    st["mw"] = beta1 * st["mw"] + (1 - beta1) * gw
                    st["vw"] = beta2 * st["vw"] + (1 - beta2) * gw**2
                    st["mb"] = beta1 * st["mb"] + (1 - beta1) * gb
                    st["vb"] = beta2 * st["vb"] + (1 - beta2) * gb**2
                    mw_hat = st["mw"] / (1 - beta1**t)
                    vw_hat = st["vw"] / (1 - beta2**t)
                    mb_hat = st["mb"] / (1 - beta1**t)
                    vb_hat = st["vb"] / (1 - beta2**t)
                    self.weights[layer] -= lr * mw_hat / (np.sqrt(vw_hat) + eps)
                    self.biases[layer] -= lr * mb_hat / (np.sqrt(vb_hat) + eps)
            history.append(epoch_loss / max(n_batches, 1))
        return history
