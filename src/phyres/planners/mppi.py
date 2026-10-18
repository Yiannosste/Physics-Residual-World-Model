"""Uncertainty-penalised Model Predictive Path Integral (MPPI) control.

MPPI (Williams, Aldrich & Theodorou, 2017, "Information Theoretic MPC for
Model-Based Reinforcement Learning") replaces CEM's hard elite-selection
with a *soft*, information-theoretic reweighting of every sampled
trajectory: each of the ``N`` sampled action sequences contributes to the
next nominal sequence in proportion to ``exp(-cost / temperature)``. Because
every sample (not just the top few) pulls the mean a little, the
resulting control sequence is markedly smoother (lower jerk) than CEM's,
which recomputes mean and standard deviation purely from a truncated elite
set each iteration.

The planning cost is identical to :class:`phyres.planners.cem.CEMPlanner`'s:

    J(a_{0:H}) = sum_t c(s_t, a_t)  +  lambda_u * sum_t d(s_t, a_t),

so the same ensemble disagreement signal that makes CEM risk-aware also
makes MPPI risk-aware: trajectories that pass through poorly-modelled
regions are still charged extra and therefore down-weighted in the softmax.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class MPPIPlanner:
    """Receding-horizon MPPI planner over a (possibly hybrid) world model."""

    def __init__(
        self,
        world_model,
        cost_fn,
        action_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        horizon: int = 25,
        n_samples: int = 300,
        temperature: float = 1.0,
        noise_std: Optional[np.ndarray] = None,
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
        self.temperature = temperature
        self.noise_std = (
            0.5 * (self.action_high - self.action_low)
            if noise_std is None
            else np.broadcast_to(np.asarray(noise_std, dtype=float), (action_dim,)).copy()
        )
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

    def _rollout_costs(self, state: np.ndarray, action_seqs: np.ndarray) -> np.ndarray:
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
        """Return the first action of the updated nominal sequence, shape (action_dim,)."""
        h = self.horizon
        if self._warm_mean is None:
            mean = np.tile(self.init_mean, (h, 1))
        else:  # shift previous solution one step, pad with the nominal action
            mean = np.vstack([self._warm_mean[1:], self.init_mean[None, :]])

        noise = self.rng.normal(size=(self.n_samples, h, self.action_dim)) * self.noise_std
        candidates = np.clip(mean[None] + noise, self.action_low, self.action_high)
        # Effective perturbation after clipping, so the weighted update stays
        # consistent with the trajectories actually simulated and costed.
        eps = candidates - mean[None]

        costs = self._rollout_costs(state, candidates)
        costs -= costs.min()  # numerically stable softmax
        weights = np.exp(-costs / self.temperature)
        weights /= weights.sum() + 1e-12

        update = np.einsum("n,nha->ha", weights, eps)
        new_mean = np.clip(mean + update, self.action_low, self.action_high)
        self._warm_mean = new_mean
        return new_mean[0]
