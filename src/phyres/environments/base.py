"""Abstract interfaces shared by every benchmark task in the suite.

A benchmark task is defined by a pair of objects that share a state/action
convention:

* :class:`PhysicsPrior`: a stateless, batched, analytic (and deliberately
  misspecified) model ``f_phys(s, a)`` that predicts the next state. This is
  the physics prior the hybrid world model builds a residual on top of
  (:mod:`phyres.ensemble`), and it is also what the CEM / MPPI planners roll
  forward when simulating candidate trajectories
  (:mod:`phyres.planners`).
* :class:`PhysicalSystem`: a stateful "real world" simulator that owns the
  current state and advances it with :meth:`~PhysicalSystem.step`,
  incorporating whatever unmodeled phenomena (friction, drag, ground
  effect, ...) make it differ from the prior. This is the reality gap the
  residual ensemble must learn to close.

Concrete tasks (:mod:`phyres.environments.pendulum`,
:mod:`phyres.environments.cartpole`, :mod:`phyres.environments.quadrotor2d`)
implement both halves plus a task cost function and are otherwise opaque to
the rest of the engine. :mod:`phyres.ensemble` and :mod:`phyres.planners`
only ever talk to these two ABCs, which is what makes the engine
task-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np


def wrap_angle(theta: np.ndarray) -> np.ndarray:
    """Wrap angles (radians) to the interval [-pi, pi)."""
    return (theta + np.pi) % (2.0 * np.pi) - np.pi


class PhysicsPrior(ABC):
    """Batched, analytic (possibly misspecified) transition model.

    All methods operate on batches: ``state`` has shape ``(B, state_dim)``
    and ``action`` has shape ``(B, action_dim)``. Subclasses must set
    ``state_dim``, ``action_dim``, ``feature_dim``, ``action_low`` and
    ``action_high`` (typically in ``__post_init__``) and implement
    :meth:`forward`.

    ``circular_dims`` declares which state indices are angles, so the engine
    (residual targets, ensemble disagreement, planner cost accumulation) can
    treat them on the circle rather than the real line.
    """

    state_dim: int
    action_dim: int
    feature_dim: int
    circular_dims: tuple[int, ...] = ()
    action_low: np.ndarray
    action_high: np.ndarray

    @abstractmethod
    def forward(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Predict ``s_{t+1} = f_phys(s_t, a_t)`` for a batch of transitions."""

    def features(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Build the residual network's input features from (state, action).

        Default: raw concatenation. Tasks with angular state dimensions
        should override this to feed ``(sin, cos)`` pairs instead of the
        raw angle, so the network never has to learn across the +-pi
        discontinuity.
        """
        return np.concatenate([state, action], axis=-1)

    def wrap_state(self, state: np.ndarray) -> np.ndarray:
        """Canonicalise a (batch of) state, wrapping circular dimensions.

        Applied both to raw states and to state *differences* (e.g. residual
        targets): wrapping a difference of two angles onto [-pi, pi) yields
        the correct geodesic error on the circle, so this one function
        serves both purposes.
        """
        if not self.circular_dims:
            return state
        wrapped = np.array(state, copy=True)
        for i in self.circular_dims:
            wrapped[..., i] = wrap_angle(wrapped[..., i])
        return wrapped


class PhysicalSystem(ABC):
    """Stateful ground-truth simulator ("the real world").

    Unlike :class:`PhysicsPrior`, a physical system owns a single current
    state and is advanced one step at a time via :meth:`step`, mirroring a
    real robot / gym-style environment. This is deliberately not batched: it
    is used for sequential data-collection rollouts and closed-loop control.
    """

    state_dim: int
    action_dim: int

    @abstractmethod
    def reset(
        self, *, state: Optional[np.ndarray] = None, seed: Optional[int] = None
    ) -> np.ndarray:
        """Reset to an initial state (explicit, freshly sampled, or default).

        Returns the initial observation, shape ``(state_dim,)``.
        """

    @abstractmethod
    def step(self, action: np.ndarray) -> np.ndarray:
        """Advance one step under the true (unmodeled) dynamics.

        ``action`` has shape ``(action_dim,)``. Returns the next
        observation, shape ``(state_dim,)``, and updates :attr:`state`.
        """

    @property
    @abstractmethod
    def state(self) -> np.ndarray:
        """The current internal state, shape ``(state_dim,)``."""


CostFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass
class BenchmarkTask:
    """Bundles everything the engine needs to run on one benchmark task.

    A thin convenience wrapper so experiments and the demo notebook can do
    ``task = make_task(seed=0)`` instead of wiring up prior, true system and
    cost function by hand for every task.
    """

    name: str
    prior: PhysicsPrior
    true_system: PhysicalSystem
    cost_fn: CostFn
    initial_state: np.ndarray
    plan_horizon: int
    eval_horizon: int
    planner_init_mean: Optional[np.ndarray] = None
    explore_center: Optional[np.ndarray] = None
    explore_scale: float = 1.0
    explore_episode_len: int = 120
    default_uncertainty_weight: float = 2.0
    # Task-owned planner tuning: some tasks (e.g. a torque-sensitive
    # quadrotor) need different sampling scales than the generic defaults.
    # Merged over CEMPlanner / MPPIPlanner's constructor defaults by
    # whichever runner builds the planner (see experiments/run_benchmark.py).
    cem_kwargs: dict[str, Any] = field(default_factory=dict)
    mppi_kwargs: dict[str, Any] = field(default_factory=dict)
