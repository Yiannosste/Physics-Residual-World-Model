"""Cart-pole balancing: the second PhyRes benchmark task.

State convention: ``s = (x, x_dot, theta, theta_dot)`` where ``x`` is the
cart position, and ``theta`` is the pole angle from the *upright* (unstable)
equilibrium, so ``theta = 0`` is the balanced state. Actions are a scalar
horizontal force applied to the cart.

``CartPolePrior`` integrates the textbook rigid-body cart-pole equations
(Florian, 2007) with no friction of any kind, the standard idealisation
used in most control-theory derivations. ``CartPoleSystem`` is the "real
world": it adds track (Coulomb + viscous) friction on the cart and a
dead-zone on the transmitted force that models cable slack / backlash in
the actuation linkage, plus extra viscous damping at the pole hinge. None
of these three effects are present in the prior, so together they are the
reality gap the residual ensemble must learn.
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


def _cartpole_accelerations(
    x_dot: np.ndarray,
    theta: np.ndarray,
    theta_dot: np.ndarray,
    net_force: np.ndarray,
    cart_mass: float,
    pole_mass: float,
    half_length: float,
    hinge_damping: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Shared rigid-body core (Florian, 2007), theta measured from upright."""
    total_mass = cart_mass + pole_mass
    sin_t, cos_t = np.sin(theta), np.cos(theta)
    temp = (net_force + pole_mass * half_length * theta_dot**2 * sin_t) / total_mass
    theta_acc = (
        GRAVITY * sin_t - cos_t * temp - hinge_damping * theta_dot / (pole_mass * half_length)
    ) / (half_length * (4.0 / 3.0 - pole_mass * cos_t**2 / total_mass))
    x_acc = temp - pole_mass * half_length * theta_acc * cos_t / total_mass
    return x_acc, theta_acc


@dataclass
class CartPolePrior(PhysicsPrior):
    """Idealised, frictionless cart-pole: standard rigid-body kinematics."""

    cart_mass: float = 1.0
    pole_mass: float = 0.1
    half_length: float = 0.5
    dt: float = 0.02
    u_max: float = 12.0

    def __post_init__(self) -> None:
        self.state_dim = 4
        self.action_dim = 1
        self.feature_dim = 6
        self.circular_dims = (2,)
        self.action_low = np.array([-self.u_max])
        self.action_high = np.array([self.u_max])

    def forward(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        x, x_dot, theta, theta_dot = (
            state[..., 0],
            state[..., 1],
            state[..., 2],
            state[..., 3],
        )
        force = action[..., 0]
        x_acc, theta_acc = _cartpole_accelerations(
            x_dot, theta, theta_dot, force,
            self.cart_mass, self.pole_mass, self.half_length, hinge_damping=0.0,
        )
        x_dot_next = x_dot + self.dt * x_acc
        x_next = x + self.dt * x_dot_next
        theta_dot_next = theta_dot + self.dt * theta_acc
        theta_next = wrap_angle(theta + self.dt * theta_dot_next)
        return np.stack([x_next, x_dot_next, theta_next, theta_dot_next], axis=-1)

    def features(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        x, x_dot, theta, theta_dot = (
            state[..., 0],
            state[..., 1],
            state[..., 2],
            state[..., 3],
        )
        force = action[..., 0]
        return np.stack(
            [x, x_dot, np.sin(theta), np.cos(theta), theta_dot, force], axis=-1
        )


@dataclass
class CartPoleSystem(PhysicalSystem):
    """Ground-truth cart-pole with track friction and cable slack.

    Reality gap relative to the prior:

    * Coulomb + viscous friction on the cart-track interface,
    * a force dead-zone modelling cable slack / backlash in the actuator
      linkage (forces below ``slack_force`` fail to transmit at all),
    * extra viscous damping at the pole hinge (a looser, lossier joint).
    """

    cart_mass: float = 1.0
    pole_mass: float = 0.1
    half_length: float = 0.5
    dt: float = 0.02
    u_max: float = 12.0
    track_coulomb: float = 0.35
    track_viscous: float = 0.12
    slack_force: float = 0.4
    hinge_damping: float = 0.02
    seed: int = 0
    _rng: np.random.Generator = field(init=False, repr=False)
    _state: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.state_dim = 4
        self.action_dim = 1
        self._rng = np.random.default_rng(self.seed)
        self._state = np.zeros(4)

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
            self._state = np.array(
                [
                    self._rng.uniform(-0.5, 0.5),
                    self._rng.uniform(-0.5, 0.5),
                    self._rng.uniform(-0.3, 0.3),
                    self._rng.uniform(-0.5, 0.5),
                ]
            )
        return self.state

    def _deadzone(self, force: np.ndarray) -> np.ndarray:
        """Cable-slack dead-zone: |F| below the slack threshold transmits 0."""
        magnitude = np.maximum(np.abs(force) - self.slack_force, 0.0)
        return np.sign(force) * magnitude

    def step(self, action: np.ndarray) -> np.ndarray:
        s = self._state[None, :]
        a = np.asarray(action, dtype=float).reshape(1, -1)
        x, x_dot, theta, theta_dot = s[..., 0], s[..., 1], s[..., 2], s[..., 3]
        force = np.clip(a[..., 0], -self.u_max, self.u_max)
        force_eff = self._deadzone(force)
        track_friction = self.track_coulomb * np.tanh(10.0 * x_dot) + self.track_viscous * x_dot
        net_force = force_eff - track_friction
        x_acc, theta_acc = _cartpole_accelerations(
            x_dot, theta, theta_dot, net_force,
            self.cart_mass, self.pole_mass, self.half_length,
            hinge_damping=self.hinge_damping,
        )
        x_dot_next = x_dot + self.dt * x_acc
        x_next = x + self.dt * x_dot_next
        theta_dot_next = theta_dot + self.dt * theta_acc
        theta_next = wrap_angle(theta + self.dt * theta_dot_next)
        self._state = np.array(
            [x_next[0], x_dot_next[0], theta_next[0], theta_dot_next[0]]
        )
        return self.state


def balance_cost(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    """Quadratic balancing cost; zero at the centred, upright rest state.

    c(s, a) = theta^2 + 0.1 * theta_dot^2 + 0.05 * x^2 + 0.01 * x_dot^2 + 0.001 * F^2
    """
    x = state[..., 0]
    x_dot = state[..., 1]
    theta = wrap_angle(state[..., 2])
    theta_dot = state[..., 3]
    force = action[..., 0]
    return (
        theta**2
        + 0.1 * theta_dot**2
        + 0.05 * x**2
        + 0.01 * x_dot**2
        + 0.001 * force**2
    )


def make_task(seed: int = 0) -> BenchmarkTask:
    """Build the cart-pole balancing benchmark task (small perturbation)."""
    true_system = CartPoleSystem(seed=seed)
    prior = CartPolePrior(dt=true_system.dt, u_max=true_system.u_max)
    return BenchmarkTask(
        name="cartpole",
        prior=prior,
        true_system=true_system,
        cost_fn=balance_cost,
        initial_state=np.array([0.0, 0.0, 0.3, 0.0]),
        plan_horizon=25,
        eval_horizon=250,
    )
