"""Uncertainty-penalised Cross-Entropy Method (CEM) model-predictive control.

At every timestep the planner samples candidate action sequences, rolls
them forward through the learned world model, and refits a Gaussian
sampling distribution to the elite (lowest-cost) candidates. The planning
cost is

    J(a_{0:H}) = sum_t c(s_t, a_t)  +  lambda_u * sum_t d(s_t, a_t),

where ``d`` is the ensemble disagreement from
:class:`phyres.ensemble.ResidualEnsembleWorldModel`. The penalty term
steers the optimiser away from state-action regions where the model family
disagrees with itself (regions outside the training distribution),
which is the mechanism that makes model-based control *risk-aware* rather
than merely optimistic (cf. pessimistic MPC / conservative model-based RL).

Only the first action of the optimised sequence is executed (receding
horizon); the sampling mean is warm-started by shifting the previous
solution one step forward. Generic over action dimensionality and
per-dimension bounds, so the same planner drives every benchmark task.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class CEMPlanner:
    """Receding-horizon CEM planner over a (possibly hybrid) world model."""

    def __init__(
        self,
        world_model,
        cost_fn,
        action_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        horizon: int = 25,
        n_samples: int = 300,
        n_elites: int = 30,
        n_iters: int = 4,
        uncertainty_weight: float = 0.0,
        discount: float = 1.0,
        init_mean: Optional[np.ndarray] = None,
        seed: int = 0,
    ) -> None:
        self.model = world_model
        self.cost_fn = cost_fn
        self.action_dim = action_dim
        self.action_low = np.broadcast_to(np.asarray(action_low, dtype=float), (action_dim,)).copy()
        self.action_high = np.broadcast_to(np.asarray(action_high, dtype=float), (action_dim,)).copy()
        self.horizon = horizon
        self.n_samples = n_samples
        self.n_elites = n_elites
        self.n_iters = n_iters
        self.uncertainty_weight = uncertainty_weight
        self.discount = discount
        self.init_mean = (
            np.zeros(action_dim) if init_mean is None else np.asarray(init_mean, dtype=float)
        )
        self.rng = np.random.default_rng(seed)
        self._warm_mean: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Forget the warm start (call at the beginning of an episode)."""
        self._warm_mean = None

    def _evaluate(self, state: np.ndarray, action_seqs: np.ndarray) -> np.ndarray:
        """Total planning cost of (N, H, action_dim) sequences from one state."""
        n = action_seqs.shape[0]
        s = np.tile(state[None, :], (n, 1))
        total = np.zeros(n)
        gamma = 1.0
        for t in range(self.horizon):
            a = action_seqs[:, t, :]
            s_next, disagreement = self.model.predict(s, a)
            total += gamma * self.cost_fn(s, a)
            total += self.uncertainty_weight * disagreement
            s = s_next
            gamma *= self.discount
        return total

    def plan(self, state: np.ndarray) -> np.ndarray:
        """Return the first action of the optimised sequence, shape (action_dim,)."""
        h = self.horizon
        if self._warm_mean is None:
            mean = np.tile(self.init_mean, (h, 1))
        else:  # shift previous solution one step, pad with the nominal action
            mean = np.vstack([self._warm_mean[1:], self.init_mean[None, :]])
        # Quarter of the full action range == 0.5 * u_max for the original
        # symmetric [-u_max, u_max] pendulum bounds. Preserved here so
        # generalising to per-dimension bounds doesn't silently double the
        # sampling variance (and with it, the planner's behaviour).
        std = np.tile(0.25 * (self.action_high - self.action_low), (h, 1))
        best_seq = mean
        for _ in range(self.n_iters):
            noise = self.rng.normal(size=(self.n_samples, h, self.action_dim))
            candidates = np.clip(
                mean[None] + std[None] * noise, self.action_low, self.action_high
            )
            costs = self._evaluate(state, candidates)
            elite_idx = np.argsort(costs)[: self.n_elites]
            elites = candidates[elite_idx]
            mean = elites.mean(axis=0)
            std = elites.std(axis=0) + 1e-3
            best_seq = elites[0]
        self._warm_mean = best_seq
        return best_seq[0]
