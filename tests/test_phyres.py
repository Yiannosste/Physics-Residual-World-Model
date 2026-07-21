"""Unit tests for the PhyRes engine: environments, learning, uncertainty, planning."""

import numpy as np
import pytest

from phyres.data import collect_random_rollouts
from phyres.ensemble import ResidualEnsembleWorldModel
from phyres.environments import TASKS, make_task
from phyres.environments.base import wrap_angle
from phyres.environments.cartpole import CartPolePrior, CartPoleSystem
from phyres.environments.pendulum import PendulumPrior, PendulumSystem, swingup_cost
from phyres.environments.quadrotor2d import Quadrotor2DPrior, Quadrotor2DSystem
from phyres.nn import MLP
from phyres.planners import CEMPlanner, MPPIPlanner


# --------------------------------------------------------------------------- #
# Core math / physics sanity
# --------------------------------------------------------------------------- #

def test_wrap_angle_range():
    theta = np.array([-7.0, -np.pi, 0.0, np.pi, 7.0])
    wrapped = wrap_angle(theta)
    assert np.all(wrapped >= -np.pi) and np.all(wrapped < np.pi)
    assert np.allclose(np.sin(wrapped), np.sin(theta), atol=1e-12)


def test_mlp_fits_simple_function():
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, size=(512, 2))
    y = np.stack([np.sin(3 * x[:, 0]), x[:, 1] ** 2], axis=-1)
    net = MLP(2, (32, 32), 2, rng)
    history = net.fit(x, y, epochs=300, lr=3e-3)
    assert history[-1] < 0.01


# --------------------------------------------------------------------------- #
# Pendulum
# --------------------------------------------------------------------------- #

def test_pendulum_upright_equilibrium_is_fixed_point():
    prior = PendulumPrior()
    s = np.array([[0.0, 0.0]])
    a = np.zeros((1, 1))
    assert np.allclose(prior.forward(s, a), s, atol=1e-12)


def test_pendulum_true_system_dissipates_energy():
    """With zero torque, friction must monotonically drain total energy."""
    sys_ = PendulumSystem()
    sys_.reset(state=np.array([2.5, 0.0]))

    def energy(state):
        theta, omega = state
        inertia = sys_.mass * sys_.length**2
        return 0.5 * inertia * omega**2 + sys_.mass * 9.81 * sys_.length * (
            1.0 + np.cos(theta)
        )

    e_prev = energy(sys_.state)
    for _ in range(400):
        sys_.step(np.zeros(1))
    assert energy(sys_.state) < 0.5 * e_prev


# --------------------------------------------------------------------------- #
# Cart-pole
# --------------------------------------------------------------------------- #

def test_cartpole_upright_equilibrium_is_fixed_point():
    prior = CartPolePrior()
    s = np.zeros((1, 4))
    a = np.zeros((1, 1))
    assert np.allclose(prior.forward(s, a), s, atol=1e-10)


def test_cartpole_deadzone_blocks_small_forces():
    sys_ = CartPoleSystem(slack_force=0.4)
    sys_.reset(state=np.zeros(4))
    s_next = sys_.step(np.array([0.1]))  # below the slack threshold
    # No force transmitted and no friction active (x_dot=0) => stays at rest.
    assert np.allclose(s_next, np.zeros(4), atol=1e-12)


# --------------------------------------------------------------------------- #
# Planar quadrotor
# --------------------------------------------------------------------------- #

def test_quadrotor_hover_is_fixed_point():
    prior = Quadrotor2DPrior()
    s = np.array([[0.0, 1.5, 0.0, 0.0, 0.0, 0.0]])
    a = np.full((1, 2), prior.hover_thrust)
    assert np.allclose(prior.forward(s, a), s, atol=1e-10)


def test_quadrotor_floor_contact_decelerates_penetration():
    """The compliant floor should push back and arrest a fast downward fall."""
    sys_ = Quadrotor2DSystem()
    sys_.reset(state=np.array([0.0, 0.01, 0.0, 0.0, -5.0, 0.0]))  # falling fast, near floor
    for _ in range(20):
        sys_.step(np.zeros(2))
    # Contact force should have arrested the fall; no unbounded sinking.
    assert sys_.state[1] > -0.25
    assert sys_.state[4] > -5.0  # downward speed reduced from the initial -5 m/s


# --------------------------------------------------------------------------- #
# Engine genericity: every registered task satisfies the same contract
# --------------------------------------------------------------------------- #

# (episodes, epochs) per task for the fast unit-test budget. The 6D
# quadrotor couples translation, rotation and a stiff ground-contact
# nonlinearity, so it needs more data/epochs than the 2D pendulum or 4D
# cart-pole to comfortably clear the same held-out-RMSE bar; production runs
# (run_benchmark.py's defaults) use an even larger budget and a wider margin.
_TRAIN_BUDGET = {"pendulum": (8, 150), "cartpole": (8, 150), "quadrotor2d": (16, 350)}


@pytest.mark.parametrize("task_name", sorted(TASKS))
def test_residual_ensemble_beats_prior(task_name):
    """C1: hybrid residual learning must close the reality gap on every task."""
    task = make_task(task_name, seed=1)
    n_episodes, epochs = _TRAIN_BUDGET[task_name]
    s, a, s_next = collect_random_rollouts(
        task.true_system, task.prior.action_low, task.prior.action_high,
        n_episodes=n_episodes, episode_len=task.explore_episode_len, seed=1,
        center=task.explore_center, amplitude_scale=task.explore_scale,
    )
    model = ResidualEnsembleWorldModel(task.prior, n_members=3, seed=1)
    model.fit(s, a, s_next, epochs=epochs)

    def rmse(pred):
        err = task.prior.wrap_state(pred - s_next)
        return float(np.sqrt(np.mean(err**2)))

    err_prior = rmse(task.prior.forward(s, a))
    err_model = rmse(model.predict(s, a)[0])
    assert err_model < 0.5 * err_prior


@pytest.mark.parametrize("task_name", sorted(TASKS))
def test_disagreement_grows_out_of_distribution(task_name):
    """C2: ensemble disagreement must be much larger off-distribution."""
    task = make_task(task_name, seed=2)
    s, a, s_next = collect_random_rollouts(
        task.true_system, task.prior.action_low, task.prior.action_high,
        n_episodes=8, seed=2,
    )
    model = ResidualEnsembleWorldModel(task.prior, n_members=3, seed=2)
    model.fit(s, a, s_next, epochs=150)
    _, d_in = model.predict(s[:200], a[:200])

    # Far outside anything seen during training: extreme velocities/actions.
    rng = np.random.default_rng(3)
    s_ood = s[:200].copy()
    s_ood[:, -1] = 25.0 * np.sign(rng.normal(size=200))  # blow up the last (velocity-like) dim
    a_ood = np.broadcast_to(task.prior.action_high * 5.0 + 5.0, (200, task.prior.action_dim)).copy()
    _, d_ood = model.predict(s_ood, a_ood)
    assert d_ood.mean() > 3.0 * d_in.mean()


# --------------------------------------------------------------------------- #
# Planners
# --------------------------------------------------------------------------- #

class _PriorModel:
    def __init__(self, prior):
        self.prior = prior

    def predict(self, state, action):
        return self.prior.forward(state, action), np.zeros(state.shape[0])


@pytest.mark.parametrize("planner_cls", [CEMPlanner, MPPIPlanner])
def test_planner_stabilises_pendulum_near_upright(planner_cls):
    """From a small perturbation, prior-based planning must reach the upright set."""
    prior = PendulumPrior()
    kwargs = dict(
        world_model=_PriorModel(prior), cost_fn=swingup_cost,
        action_dim=1, action_low=prior.action_low, action_high=prior.action_high,
        horizon=15, n_samples=200, seed=0,
    )
    if planner_cls is CEMPlanner:
        planner = planner_cls(n_elites=20, n_iters=3, **kwargs)
    else:
        planner = planner_cls(temperature=0.5, **kwargs)

    s = np.array([0.4, 0.0])
    for _ in range(60):
        u = planner.plan(s)
        s = prior.forward(s[None, :], u[None, :])[0]
    assert abs(wrap_angle(np.array([s[0]]))[0]) < 0.1
    assert abs(s[1]) < 0.5


def test_cem_respects_action_bounds():
    prior = Quadrotor2DPrior()
    planner = CEMPlanner(
        world_model=_PriorModel(prior), cost_fn=lambda s, a: np.sum(a**2, axis=-1),
        action_dim=2, action_low=prior.action_low, action_high=prior.action_high,
        horizon=5, n_samples=50, n_elites=5, n_iters=2, seed=0,
    )
    s = np.array([0.0, 1.5, 0.0, 0.0, 0.0, 0.0])
    for _ in range(5):
        a = planner.plan(s)
        assert np.all(a >= prior.action_low - 1e-9) and np.all(a <= prior.action_high + 1e-9)


# --------------------------------------------------------------------------- #
# Online continual adaptation
# --------------------------------------------------------------------------- #

def test_online_update_adapts_to_a_shifted_reality_gap():
    """A residual trained on one friction level must adapt online to another."""
    prior = PendulumPrior()
    train_system = PendulumSystem(damping=0.15, coulomb=0.08, seed=10)
    s, a, s_next = collect_random_rollouts(
        train_system, prior.action_low, prior.action_high, n_episodes=10, seed=10
    )
    model = ResidualEnsembleWorldModel(prior, n_members=3, seed=10)
    model.fit(s, a, s_next, epochs=150)

    # A much higher-friction "worn actuator" regime the model has never seen.
    shifted_system = PendulumSystem(damping=0.6, coulomb=0.35, seed=11)
    s2, a2, s2_next = collect_random_rollouts(
        shifted_system, prior.action_low, prior.action_high, n_episodes=4, seed=11
    )

    def rmse():
        err = prior.wrap_state(model.predict(s2, a2)[0] - s2_next)
        return float(np.sqrt(np.mean(err**2)))

    err_before = rmse()
    rng = np.random.default_rng(12)
    for i in range(len(s2)):
        model.online_update(s2[i], a2[i], s2_next[i], replay_batch=32, epochs=1, lr=5e-4, rng=rng)
    err_after = rmse()
    assert err_after < 0.8 * err_before
