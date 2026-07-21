"""Sampling-based receding-horizon planners over a (possibly hybrid) world model.

Both planners share the same contract: a ``world_model`` exposing
``predict(state, action) -> (next_state, disagreement)`` (any
:class:`phyres.ensemble.ResidualEnsembleWorldModel`, or a plain analytic
prior wrapped to the same interface), a batched ``cost_fn(state, action)``,
and per-dimension action bounds. Both penalise imagined trajectories by
``uncertainty_weight * disagreement``, which is what turns point-estimate
MPC into risk-aware MPC (see SCIENCE.md §5).
"""

from phyres.planners.cem import CEMPlanner
from phyres.planners.mppi import MPPIPlanner

__all__ = ["CEMPlanner", "MPPIPlanner"]
