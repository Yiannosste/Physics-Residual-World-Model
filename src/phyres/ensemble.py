"""Physics-residual world model with a deep ensemble for epistemic uncertainty.

The world model predicts

    s_{t+1} = f_phys(s_t, a_t)  +  g_psi(s_t, a_t),

where ``f_phys`` is an analytic (possibly misspecified) prior -- any
:class:`phyres.environments.base.PhysicsPrior` -- and ``g_psi`` is a small
MLP trained on the *residual* between observed transitions and the prior's
prediction. An ensemble of K independently initialised and bootstrapped
residual networks provides an epistemic-uncertainty estimate via prediction
disagreement, following the deep-ensemble line of work
(Lakshminarayanan et al., 2017; Chua et al., 2018).

The ensemble is entirely generic over the task: it reads ``state_dim``,
``feature_dim`` and ``circular_dims`` off the prior, so the same class
serves the pendulum, cart-pole and planar-quadrotor benchmarks unchanged.
Angular state dimensions are lifted to (sin, cos) features by the prior's
``features()`` method, so the network never sees the +-pi discontinuity.

:meth:`ResidualEnsembleWorldModel.online_update` supports lightweight,
warm-started continual adaptation: transitions observed while the planner
is executing in the real world are pushed into a rolling replay buffer, and
a handful of low-learning-rate Adam steps (reusing each member's existing
optimiser state) are taken against a sample of that buffer -- letting the
residual track a slowly drifting reality gap without retraining from
scratch.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Optional

import numpy as np

from phyres.environments.base import PhysicsPrior
from phyres.nn import MLP


class ResidualEnsembleWorldModel:
    """Ensemble of residual dynamics models on top of an analytic prior."""

    def __init__(
        self,
        prior: PhysicsPrior,
        n_members: int = 5,
        hidden: tuple[int, ...] = (64, 64),
        seed: int = 0,
        replay_capacity: int = 4000,
    ) -> None:
        self.prior = prior
        self.n_members = n_members
        root = np.random.default_rng(seed)
        self.members = [
            MLP(in_dim=prior.feature_dim, hidden=hidden, out_dim=prior.state_dim,
                rng=np.random.default_rng(s))
            for s in root.integers(0, 2**31 - 1, size=n_members)
        ]
        self._boot_rng = root
        # Residual targets are normalised for stable training.
        self.res_mean = np.zeros(prior.state_dim)
        self.res_std = np.ones(prior.state_dim)
        # Rolling replay buffer for online continual adaptation.
        self._replay: Deque[tuple[np.ndarray, np.ndarray, np.ndarray]] = deque(
            maxlen=replay_capacity
        )

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def residual_targets(
        self, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray
    ) -> np.ndarray:
        """Reality gap the network must learn: observed minus prior prediction."""
        pred = self.prior.forward(states, actions)
        return self.prior.wrap_state(next_states - pred)

    def fit(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray,
        epochs: int = 300,
        **fit_kwargs,
    ) -> list[list[float]]:
        """Train every member on a bootstrap resample of the dataset."""
        x = self.prior.features(states, actions)
        res = self.residual_targets(states, actions, next_states)
        self.res_mean = res.mean(axis=0)
        self.res_std = res.std(axis=0) + 1e-8
        y = (res - self.res_mean) / self.res_std
        histories = []
        n = x.shape[0]
        for member in self.members:
            idx = self._boot_rng.integers(0, n, size=n)  # bootstrap resample
            histories.append(member.fit(x[idx], y[idx], epochs=epochs, **fit_kwargs))
        for s, a, s_next in zip(states, actions, next_states):
            self._replay.append((s, a, s_next))
        return histories

    # ------------------------------------------------------------------ #
    # Online continual adaptation
    # ------------------------------------------------------------------ #
    def online_update(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        replay_batch: int = 64,
        epochs: int = 1,
        lr: float = 1e-4,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        """Warm-started adaptation step on one freshly observed transition.

        Pushes ``(state, action, next_state)`` into the rolling replay
        buffer, then samples up to ``replay_batch`` transitions from the
        buffer (including the new one) and takes ``epochs`` low-learning-rate
        Adam steps on every ensemble member. Because :meth:`phyres.nn.MLP.fit`
        keeps its Adam moment estimates between calls, this is a warm start,
        not a restart -- exactly the "lightweight" adaptation needed to track
        a slowly drifting reality gap during closed-loop execution without
        catastrophically forgetting the offline-trained residual.

        Returns the mean training loss across members for this update.
        """
        self._replay.append((np.asarray(state), np.asarray(action), np.asarray(next_state)))
        rng = rng or np.random.default_rng()
        n = len(self._replay)
        batch = min(replay_batch, n)
        idx = rng.integers(0, n, size=batch)
        buffer = list(self._replay)
        s = np.stack([buffer[i][0] for i in idx])
        a = np.stack([buffer[i][1] for i in idx])
        s_next = np.stack([buffer[i][2] for i in idx])

        x = self.prior.features(s, a)
        res = self.residual_targets(s, a, s_next)
        y = (res - self.res_mean) / self.res_std

        losses = []
        for member in self.members:
            history = member.fit(x, y, epochs=epochs, batch_size=batch, lr=lr)
            losses.append(history[-1])
        return float(np.mean(losses))

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #
    def predict_members(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Per-member next-state predictions, shape (K, B, state_dim)."""
        base = self.prior.forward(state, action)
        x = self.prior.features(state, action)
        preds = []
        for member in self.members:
            res = member.forward(x) * self.res_std + self.res_mean
            nxt = self.prior.wrap_state(base + res)
            preds.append(nxt)
        return np.stack(preds, axis=0)

    def predict(self, state: np.ndarray, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Ensemble mean prediction and epistemic disagreement.

        Returns ``(mean, disagreement)`` where ``disagreement`` is the total
        standard deviation across members, summed over state dimensions --
        a consistent estimator of epistemic uncertainty that vanishes on
        well-covered data and grows out of distribution. Circular state
        dimensions (declared by the prior) use the circular mean/deviation
        so the ±pi boundary never introduces a spurious artefact; all other
        dimensions use the ordinary arithmetic mean/std.
        """
        preds = self.predict_members(state, action)  # (K, B, state_dim)
        circular = set(self.prior.circular_dims)
        mean = np.empty(preds.shape[1:])
        disagreement = np.zeros(preds.shape[1])
        for i in range(self.prior.state_dim):
            col = preds[..., i]
            if i in circular:
                m = np.arctan2(np.sin(col).mean(axis=0), np.cos(col).mean(axis=0))
                dev = (col - m[None, ...] + np.pi) % (2.0 * np.pi) - np.pi
                disagreement += dev.std(axis=0)
            else:
                m = col.mean(axis=0)
                disagreement += col.std(axis=0)
            mean[..., i] = m
        return mean, disagreement
