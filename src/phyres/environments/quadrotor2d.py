"""Planar (2D) quadrotor hover: the third PhyRes benchmark task.

State convention: ``s = (x, y, phi, x_dot, y_dot, phi_dot)`` with ``x, y``
the position of the vehicle's centre of mass, ``phi`` the roll angle
(``phi = 0`` means the thrust axis points straight up) and the trailing
three components their time derivatives. Actions are the two rotor
thrusts ``(F_left, F_right)``, each constrained to ``[0, F_max]`` (rotors
can only push).

``Quadrotor2DPrior`` assumes textbook vacuum kinematics: two point thrusts
acting through the vehicle's centre of mass, no aerodynamic interaction with
the surrounding air. ``Quadrotor2DSystem`` is the "real world": it adds
quadratic aerodynamic drag on both translational velocities and a
*ground-effect* lift boost that grows sharply as the vehicle approaches the
floor (``y -> 0``), a well known, strongly nonlinear near-ground
phenomenon that a vacuum-kinematics prior cannot represent at all. A simple
floor contact clamp prevents the true system's state from going through the
ground during exploration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from phyres.environments.base import (
    BenchmarkTask,
    CostFn,
    PhysicalSystem,
    PhysicsPrior,
    wrap_angle,
)

GRAVITY = 9.81


@dataclass
class Quadrotor2DPrior(PhysicsPrior):
    """Idealised planar quadrotor: two point thrusts, no aerodynamics."""

    mass: float = 0.5
    arm: float = 0.15
    inertia: float = 0.0023
    dt: float = 0.02
    thrust_max: float = 6.0

    def __post_init__(self) -> None:
        self.state_dim = 6
        self.action_dim = 2
        self.feature_dim = 9
        self.circular_dims = (2,)
        self.action_low = np.zeros(2)
        self.action_high = np.full(2, self.thrust_max)

    @property
    def hover_thrust(self) -> float:
        """Per-rotor thrust that exactly balances gravity."""
        return 0.5 * self.mass * GRAVITY

    def forward(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        x, y, phi, x_dot, y_dot, phi_dot = (
            state[..., 0], state[..., 1], state[..., 2],
            state[..., 3], state[..., 4], state[..., 5],
        )
        f_left, f_right = action[..., 0], action[..., 1]
        total_thrust = f_left + f_right
        x_acc = -(total_thrust / self.mass) * np.sin(phi)
        y_acc = (total_thrust / self.mass) * np.cos(phi) - GRAVITY
        phi_acc = (f_right - f_left) * self.arm / self.inertia

        x_dot_n = x_dot + self.dt * x_acc
        y_dot_n = y_dot + self.dt * y_acc
        phi_dot_n = phi_dot + self.dt * phi_acc
        x_n = x + self.dt * x_dot_n
        y_n = y + self.dt * y_dot_n
        phi_n = wrap_angle(phi + self.dt * phi_dot_n)
        return np.stack([x_n, y_n, phi_n, x_dot_n, y_dot_n, phi_dot_n], axis=-1)

    def features(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        x, y, phi, x_dot, y_dot, phi_dot = (
            state[..., 0], state[..., 1], state[..., 2],
            state[..., 3], state[..., 4], state[..., 5],
        )
        return np.stack(
            [x, y, np.sin(phi), np.cos(phi), x_dot, y_dot, phi_dot,
             action[..., 0], action[..., 1]],
            axis=-1,
        )


@dataclass
class Quadrotor2DSystem(PhysicalSystem):
    """Ground-truth planar quadrotor with drag and near-ground lift.

    Reality gap relative to the prior:

    * quadratic aerodynamic drag on both translational velocities,
    * a ground-effect multiplier on total thrust that grows the closer the
      vehicle gets to the floor (``exp(-y / ground_effect_height)``),
    * a compliant (spring-damper) floor contact force that activates on
      penetration (``y < 0``). It is smooth and continuous rather than an
      instantaneous velocity clamp, so it is (like Coulomb friction
      elsewhere in this repo) a strong but learnable nonlinearity.
    """

    mass: float = 0.5
    arm: float = 0.15
    inertia: float = 0.0023
    dt: float = 0.02
    thrust_max: float = 6.0
    drag_coeff: float = 0.15
    ground_effect_gain: float = 0.35
    ground_effect_height: float = 0.3
    contact_stiffness: float = 300.0
    contact_damping: float = 30.0
    seed: int = 0
    _rng: np.random.Generator = field(init=False, repr=False)
    _state: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.state_dim = 6
        self.action_dim = 2
        self._rng = np.random.default_rng(self.seed)
        self._state = np.array([0.0, 1.5, 0.0, 0.0, 0.0, 0.0])

    @property
    def hover_thrust(self) -> float:
        return 0.5 * self.mass * GRAVITY

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
                    self._rng.uniform(0.8, 2.0),
                    self._rng.uniform(-0.4, 0.4),
                    self._rng.uniform(-0.5, 0.5),
                    self._rng.uniform(-0.5, 0.5),
                    self._rng.uniform(-0.5, 0.5),
                ]
            )
        return self.state

    def step(self, action: np.ndarray) -> np.ndarray:
        s = self._state[None, :]
        a = np.asarray(action, dtype=float).reshape(1, -1)
        x, y, phi, x_dot, y_dot, phi_dot = (
            s[..., 0], s[..., 1], s[..., 2], s[..., 3], s[..., 4], s[..., 5],
        )
        f_left = np.clip(a[..., 0], 0.0, self.thrust_max)
        f_right = np.clip(a[..., 1], 0.0, self.thrust_max)
        total_thrust = f_left + f_right

        ground_effect = 1.0 + self.ground_effect_gain * np.exp(
            -np.maximum(y, 0.0) / self.ground_effect_height
        )
        drag_x = self.drag_coeff * x_dot * np.abs(x_dot)
        drag_y = self.drag_coeff * y_dot * np.abs(y_dot)

        # Compliant floor contact: a spring-damper force that only acts on
        # penetration (y < 0), continuous in y and y_dot. Unlike a hard
        # position/velocity clamp, this never introduces a state
        # discontinuity, so it remains a learnable residual just like the
        # other unmodeled forces here.
        penetration = np.maximum(-y, 0.0)
        contact_acc = self.contact_stiffness * penetration - self.contact_damping * y_dot * (
            penetration > 0.0
        )

        x_acc = -(total_thrust / self.mass) * np.sin(phi) - drag_x / self.mass
        y_acc = (
            (total_thrust * ground_effect / self.mass) * np.cos(phi)
            - GRAVITY
            - drag_y / self.mass
            + contact_acc
        )
        phi_acc = (f_right - f_left) * self.arm / self.inertia

        x_dot_n = x_dot + self.dt * x_acc
        y_dot_n = y_dot + self.dt * y_acc
        phi_dot_n = phi_dot + self.dt * phi_acc
        x_n = x + self.dt * x_dot_n
        y_n = np.maximum(y + self.dt * y_dot_n, -0.25)  # safety floor vs. Euler overshoot
        phi_n = wrap_angle(phi + self.dt * phi_dot_n)

        self._state = np.array(
            [x_n[0], y_n[0], phi_n[0], x_dot_n[0], y_dot_n[0], phi_dot_n[0]]
        )
        return self.state


def make_hover_cost(
    target_x: float = 0.0, target_y: float = 1.5, hover_thrust: float = 1.226
) -> CostFn:
    """Build a hover-at-(target_x, target_y) quadratic cost function."""

    def hover_cost(state: np.ndarray, action: np.ndarray) -> np.ndarray:
        x, y, phi = state[..., 0], state[..., 1], wrap_angle(state[..., 2])
        x_dot, y_dot, phi_dot = state[..., 3], state[..., 4], state[..., 5]
        f_left, f_right = action[..., 0], action[..., 1]
        return (
            2.0 * (x - target_x) ** 2
            + 2.0 * (y - target_y) ** 2
            + 4.0 * phi**2
            + 0.1 * (x_dot**2 + y_dot**2)
            + 0.05 * phi_dot**2
            + 0.02 * ((f_left - hover_thrust) ** 2 + (f_right - hover_thrust) ** 2)
        )

    return hover_cost


def make_task(seed: int = 0) -> BenchmarkTask:
    """Build the planar-quadrotor hover benchmark task (roll perturbation)."""
    true_system = Quadrotor2DSystem(seed=seed)
    prior = Quadrotor2DPrior(
        mass=true_system.mass,
        arm=true_system.arm,
        inertia=true_system.inertia,
        dt=true_system.dt,
        thrust_max=true_system.thrust_max,
    )
    target = np.array([0.0, 1.5])
    cost_fn = make_hover_cost(target[0], target[1], hover_thrust=prior.hover_thrust)
    return BenchmarkTask(
        name="quadrotor2d",
        prior=prior,
        true_system=true_system,
        cost_fn=cost_fn,
        initial_state=np.array([0.3, 1.5, 0.35, 0.0, 0.0, 0.0]),
        plan_horizon=20,
        eval_horizon=150,
        planner_init_mean=np.full(2, prior.hover_thrust),
        # Random independent per-rotor thrust is torque-violent (the roll
        # gain arm/inertia is large), so an exploration policy spanning the
        # full [0, thrust_max] range mostly just tumbles the vehicle. Centre
        # exploration on hover thrust with a reduced amplitude and shorter
        # episodes, like a real flight-test engineer would.
        explore_center=np.full(2, prior.hover_thrust),
        explore_scale=0.35,
        explore_episode_len=60,
        # A large roll-torque gain (arm / inertia) makes the default
        # full-range sampling scale far too aggressive for this task; a
        # smaller, hover-relative noise scale gives smooth, stable control.
        default_uncertainty_weight=0.2,
        mppi_kwargs={"n_samples": 400, "temperature": 0.3, "noise_std": np.full(2, 0.5)},
    )
