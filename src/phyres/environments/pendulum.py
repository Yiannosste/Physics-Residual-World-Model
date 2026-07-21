"""Pendulum swing-up -- the original PhyRes benchmark task.

The central object of study is the *reality gap*: the analytic prior
``PendulumPrior`` is a textbook rigid-body model with (deliberately)
misspecified parameters, while ``PendulumSystem`` is the "real world" -- it
integrates the same rigid-body core but adds unmodeled phenomena (quadratic
aerodynamic drag, dry Coulomb friction, actuator saturation and a parameter
offset). The learned residual in :mod:`phyres.ensemble` must absorb exactly
this gap.

State convention: ``s = (theta, omega)`` with ``theta`` the angle from the
upright position (``theta = 0`` is the unstable equilibrium we want to
stabilise) and ``omega`` the angular velocity. Actions are a scalar torque.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from phyres.environments.base import (
    BenchmarkTask,
    PhysicalSystem,
    PhysicsPrior,
    wrap_angle,
)

GRAVITY = 9.81


@dataclass
class PendulumPrior(PhysicsPrior):
    """Frictionless torque-actuated pendulum, semi-implicit Euler.

    Parameters are intentionally *wrong* relative to :class:`PendulumSystem`
    to emulate the model misspecification any real robot faces: identified
    mass and length are never exact, and friction is rarely modelled well.
    """

    mass: float = 1.0
    length: float = 1.0
    damping: float = 0.0  # the prior naively assumes no friction at all
    dt: float = 0.05
    u_max: float = 3.5

    def __post_init__(self) -> None:
        self.state_dim = 2
        self.action_dim = 1
        self.feature_dim = 4
        self.circular_dims = (0,)
        self.action_low = np.array([-self.u_max])
        self.action_high = np.array([self.u_max])

    def forward(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        theta, omega = state[..., 0], state[..., 1]
        u = action[..., 0]
        inertia = self.mass * self.length**2
        # theta measured from upright => gravity torque is +(g/l) sin(theta)
        alpha = (
            (GRAVITY / self.length) * np.sin(theta)
            - (self.damping / inertia) * omega
            + u / inertia
        )
        omega_next = omega + self.dt * alpha
        theta_next = wrap_angle(theta + self.dt * omega_next)
        return np.stack([theta_next, omega_next], axis=-1)

    def features(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        theta, omega = state[..., 0], state[..., 1]
        u = action[..., 0]
        return np.stack([np.sin(theta), np.cos(theta), omega, u], axis=-1)


@dataclass
class PendulumSystem(PhysicalSystem):
    """Ground-truth simulator ("the real world") used for data and eval.

    Differences from the prior, i.e. the reality gap the residual must
    learn:

    * heavier bob and longer arm (parameter misidentification),
    * viscous damping ``b * omega`` (unmodeled linear friction),
    * quadratic aerodynamic drag ``c_d * omega * |omega|``,
    * dry Coulomb friction ``mu * sign(omega)`` (smoothed with tanh),
    * torque saturation at ``u_max``.
    """

    mass: float = 1.25
    length: float = 1.15
    damping: float = 0.15
    quad_drag: float = 0.04
    coulomb: float = 0.08
    u_max: float = 3.5
    dt: float = 0.05
    seed: int = 0
    _rng: np.random.Generator = field(init=False, repr=False)
    _state: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.state_dim = 2
        self.action_dim = 1
        self._rng = np.random.default_rng(self.seed)
        self._state = np.zeros(2)

    @property
    def state(self) -> np.ndarray:
        return self._state.copy()

    def reset(
        self, *, state: Optional[np.ndarray] = None, seed: Optional[int] = None
    ) -> np.ndarray:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if state is not None:
            self._state = np.asarray(state, dtype=float).copy()
        else:
            theta0 = self._rng.uniform(-np.pi, np.pi)
            omega0 = self._rng.uniform(-1.0, 1.0)
            self._state = np.array([theta0, omega0])
        return self.state

    def step(self, action: np.ndarray) -> np.ndarray:
        s = self._state[None, :]
        a = np.asarray(action, dtype=float).reshape(1, -1)
        theta, omega = s[..., 0], s[..., 1]
        u = np.clip(a[..., 0], -self.u_max, self.u_max)
        inertia = self.mass * self.length**2
        friction = (
            self.damping * omega
            + self.quad_drag * omega * np.abs(omega)
            + self.coulomb * np.tanh(10.0 * omega)  # smoothed Coulomb term
        )
        alpha = (GRAVITY / self.length) * np.sin(theta) + (u - friction) / inertia
        omega_next = omega + self.dt * alpha
        theta_next = wrap_angle(theta + self.dt * omega_next)
        self._state = np.array([theta_next[0], omega_next[0]])
        return self.state


def swingup_cost(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    """Quadratic swing-up cost; zero at the upright rest state.

    c(s, a) = theta^2 + 0.1 * omega^2 + 0.001 * u^2
    """
    theta = wrap_angle(state[..., 0])
    omega = state[..., 1]
    u = action[..., 0]
    return theta**2 + 0.1 * omega**2 + 0.001 * u**2


def make_task(seed: int = 0) -> BenchmarkTask:
    """Build the pendulum swing-up benchmark task (hanging start)."""
    true_system = PendulumSystem(seed=seed)
    prior = PendulumPrior(dt=true_system.dt, u_max=true_system.u_max)
    return BenchmarkTask(
        name="pendulum",
        prior=prior,
        true_system=true_system,
        cost_fn=swingup_cost,
        initial_state=np.array([np.pi, 0.0]),
        plan_horizon=25,
        eval_horizon=250,
    )
