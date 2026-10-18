"""Generic benchmark runner: any task x any planner x optional online adaptation.

Pipeline
--------
1. Collect exploratory transitions from the *true* system of the chosen
   task under a correlated random policy.
2. Train a physics-residual deep-ensemble world model on the residuals
   between observations and the misspecified analytic prior.
3. Run receding-horizon control (CEM or MPPI) with three world models:
     a) the misspecified analytic prior alone (no learning),
     b) the residual ensemble, optimistic (no uncertainty penalty),
     c) the residual ensemble with the epistemic-uncertainty penalty.
4. Report one-step model error and closed-loop control cost.

This is the task-agnostic counterpart of ``run_swingup.py``: the pipeline
itself never mentions pendulums, cart-poles or quadrotors. It only talks
to the :class:`~phyres.environments.base.PhysicsPrior` /
:class:`~phyres.environments.base.PhysicalSystem` interface, so any task
registered in :data:`phyres.environments.TASKS` runs through it unchanged.

Examples
--------
    python experiments/run_benchmark.py --task pendulum --planner cem --seed 0
    python experiments/run_benchmark.py --task cartpole --planner mppi --seed 0
    python experiments/run_benchmark.py --task quadrotor2d --planner mppi --online-adapt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from phyres.data import collect_random_rollouts
from phyres.ensemble import ResidualEnsembleWorldModel
from phyres.environments import TASKS, make_task
from phyres.planners import CEMPlanner, MPPIPlanner

PLANNERS = {"cem": CEMPlanner, "mppi": MPPIPlanner}


def one_step_error(predict_fn, prior, states, actions, next_states) -> float:
    """RMS one-step prediction error on held-out transitions."""
    pred = predict_fn(states, actions)
    err = prior.wrap_state(pred - next_states)
    return float(np.sqrt(np.mean(err**2)))


class _PriorOnlyModel:
    """Adapter: analytic prior exposed through the world-model interface."""

    def __init__(self, prior):
        self.prior = prior

    def predict(self, state, action):
        return self.prior.forward(state, action), np.zeros(state.shape[0])


def run_episode(
    true_system,
    planner,
    episode_len: int,
    s0: np.ndarray,
    cost_fn,
    world_model: ResidualEnsembleWorldModel | None = None,
    online_adapt: bool = False,
    seed: int = 0,
):
    """Closed-loop rollout of a planner on the true system."""
    planner.reset()
    s = true_system.reset(state=s0)
    rng = np.random.default_rng(seed)
    total_cost = 0.0
    trajectory = [s.copy()]
    for _ in range(episode_len):
        a = planner.plan(s)
        total_cost += float(cost_fn(s[None, :], a[None, :])[0])
        s_next = true_system.step(a)
        if online_adapt and world_model is not None:
            world_model.online_update(s, a, s_next, rng=rng)
        s = s_next
        trajectory.append(s.copy())
    return total_cost, s, np.array(trajectory)


def build_planner(kind: str, world_model, task, uncertainty_weight: float, seed: int):
    common = dict(
        world_model=world_model,
        cost_fn=task.cost_fn,
        action_dim=task.prior.action_dim,
        action_low=task.prior.action_low,
        action_high=task.prior.action_high,
        horizon=task.plan_horizon,
        uncertainty_weight=uncertainty_weight,
        init_mean=task.planner_init_mean,
        seed=seed,
    )
    if kind == "cem":
        defaults = dict(n_samples=300, n_elites=30, n_iters=4)
        defaults.update(task.cem_kwargs)
        return CEMPlanner(**common, **defaults)
    defaults = dict(n_samples=300, temperature=1.0)
    defaults.update(task.mppi_kwargs)
    return MPPIPlanner(**common, **defaults)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=sorted(TASKS), default="pendulum")
    parser.add_argument("--planner", choices=sorted(PLANNERS), default="cem")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=40, help="data episodes")
    parser.add_argument("--epochs", type=int, default=300, help="training epochs")
    parser.add_argument("--uncertainty-weight", type=float, default=2.0)
    parser.add_argument("--online-adapt", action="store_true",
                         help="apply online_update() during the uncertainty-penalised rollout")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    task = make_task(args.task, seed=args.seed)
    print(f"[1/4] Collecting exploratory data on {task.name!r} ...")
    states, actions, next_states = collect_random_rollouts(
        task.true_system, task.prior.action_low, task.prior.action_high,
        n_episodes=args.episodes, episode_len=task.explore_episode_len, seed=args.seed,
        center=task.explore_center, amplitude_scale=task.explore_scale,
    )
    n_train = int(0.9 * len(states))
    train = (states[:n_train], actions[:n_train], next_states[:n_train])
    test = (states[n_train:], actions[n_train:], next_states[n_train:])
    print(f"      {n_train} train / {len(states) - n_train} held-out transitions")

    print("[2/4] Training the physics-residual ensemble ...")
    t0 = time.time()
    model = ResidualEnsembleWorldModel(task.prior, n_members=5, seed=args.seed)
    histories = model.fit(*train, epochs=args.epochs)
    print(f"      done in {time.time() - t0:.1f}s "
          f"(final member losses: {[round(h[-1], 5) for h in histories]})")

    err_prior = one_step_error(task.prior.forward, task.prior, *test)
    err_model = one_step_error(lambda s, a: model.predict(s, a)[0], task.prior, *test)
    print(f"      held-out one-step RMSE  prior: {err_prior:.4f}  "
          f"residual ensemble: {err_model:.4f}  "
          f"({err_prior / max(err_model, 1e-9):.1f}x reduction)")

    print(f"[3/4] Closed-loop control ({task.eval_horizon} steps, planner={args.planner}) ...")
    variants = {
        "prior_only": (_PriorOnlyModel(task.prior), 0.0, False),
        "residual_optimistic": (model, 0.0, False),
        "residual_uncertainty": (model, args.uncertainty_weight, args.online_adapt),
    }
    results: dict[str, dict] = {}
    for name, (wm, lam, adapt) in variants.items():
        planner = build_planner(args.planner, wm, task, lam, args.seed)
        t0 = time.time()
        cost, s_final, _ = run_episode(
            task.true_system, planner, task.eval_horizon, task.initial_state,
            task.cost_fn, world_model=wm if wm is model else None,
            online_adapt=adapt, seed=args.seed,
        )
        results[name] = {"closed_loop_cost": round(cost, 1)}
        print(f"      {name:22s} cost={cost:10.2f}  ({time.time() - t0:.0f}s)"
              f"{'  [online-adapted]' if adapt else ''}")

    print("[4/4] Writing results ...")
    payload = {
        "task": task.name,
        "planner": args.planner,
        "seed": args.seed,
        "one_step_rmse": {"prior": err_prior, "residual_ensemble": err_model},
        "control": results,
    }
    out = args.out or Path(f"results_{task.name}_{args.planner}.json")
    out.write_text(json.dumps(payload, indent=2))
    print(f"      saved to {out}")


if __name__ == "__main__":
    main()
