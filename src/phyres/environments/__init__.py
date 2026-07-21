"""Benchmark task suite for physics-residual control.

Every task exposes a matched pair of :class:`~phyres.environments.base.PhysicsPrior`
/ :class:`~phyres.environments.base.PhysicalSystem` implementations plus a
``make_task(seed)`` factory returning a :class:`~phyres.environments.base.BenchmarkTask`.
"""

from phyres.environments.base import BenchmarkTask, PhysicalSystem, PhysicsPrior, wrap_angle
from phyres.environments import cartpole, pendulum, quadrotor2d

TASKS = {
    "pendulum": pendulum.make_task,
    "cartpole": cartpole.make_task,
    "quadrotor2d": quadrotor2d.make_task,
}


def make_task(name: str, seed: int = 0) -> BenchmarkTask:
    """Build a named benchmark task by key (see :data:`TASKS`)."""
    try:
        factory = TASKS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown task {name!r}; choose from {sorted(TASKS)}") from exc
    return factory(seed=seed)


__all__ = [
    "BenchmarkTask",
    "PhysicalSystem",
    "PhysicsPrior",
    "wrap_angle",
    "TASKS",
    "make_task",
    "cartpole",
    "pendulum",
    "quadrotor2d",
]
