"""PhyRes: Physics-Residual World Models with Calibrated Epistemic Uncertainty.

A modular, fully reproducible research codebase for hybrid dynamics
learning and uncertainty-aware model-predictive control, implemented in
pure NumPy. The engine (:mod:`phyres.ensemble`, :mod:`phyres.planners`) is
task-agnostic; concrete benchmark tasks live in :mod:`phyres.environments`.
"""

from phyres.ensemble import ResidualEnsembleWorldModel
from phyres.environments import TASKS, make_task
from phyres.environments.base import BenchmarkTask, PhysicalSystem, PhysicsPrior
from phyres.planners import CEMPlanner, MPPIPlanner

__version__ = "0.2.0"

__all__ = [
    "BenchmarkTask",
    "PhysicalSystem",
    "PhysicsPrior",
    "ResidualEnsembleWorldModel",
    "CEMPlanner",
    "MPPIPlanner",
    "TASKS",
    "make_task",
]
