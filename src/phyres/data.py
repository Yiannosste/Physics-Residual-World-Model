"""Data collection utilities: exploratory rollouts on the true system."""

from __future__ import annotations

import numpy as np

from phyres.environments.base import PhysicalSystem


def collect_random_rollouts(
    system: PhysicalSystem,
    action_low: np.ndarray,
    action_high: np.ndarray,
    n_episodes: int = 40,
    episode_len: int = 120,
    seed: int = 0,
    center: np.ndarray | None = None,
    amplitude_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect (s, a, s') transitions under a smoothed random policy.

    Actions follow an Ornstein-Uhlenbeck-style correlated noise process
    (independently per action dimension) rather than white noise:
    correlated actions pump energy into the system, which yields much
    broader state-space coverage than uncorrelated exploration. Works with
    any :class:`~phyres.environments.base.PhysicalSystem`, so it is shared
    by every benchmark task.

    ``center`` and ``amplitude_scale`` let a task bias exploration around a
    physically sensible operating point (e.g. hover thrust for a quadrotor,
    where the bound midpoint is a poor exploration centre and the full
    bound range is a torque-violent one) instead of the bare action bounds
    -- see :attr:`~phyres.environments.base.BenchmarkTask.explore_center`.

    Returns arrays of shape (N, state_dim), (N, action_dim), (N, state_dim).
    """
    rng = np.random.default_rng(seed)
    action_low = np.asarray(action_low, dtype=float)
    action_high = np.asarray(action_high, dtype=float)
    amplitude = amplitude_scale * 0.5 * (action_high - action_low)
    center = 0.5 * (action_high + action_low) if center is None else np.asarray(center, dtype=float)
    states, actions, next_states = [], [], []
    for _ in range(n_episodes):
        s = system.reset(seed=int(rng.integers(0, 2**31 - 1)))
        u = center.copy()
        for _ in range(episode_len):
            u = center + 0.85 * (u - center) + 0.5 * amplitude * rng.normal(size=amplitude.shape)
            a = np.clip(u, action_low, action_high)
            s_next = system.step(a)
            states.append(s)
            actions.append(a)
            next_states.append(s_next)
            s = s_next
    return np.array(states), np.array(actions), np.array(next_states)
