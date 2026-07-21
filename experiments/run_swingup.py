"""End-to-end experiment: pendulum swing-up under model misspecification.

This is the original, single-task PhyRes experiment, preserved for
backward-compatible reproduction of the README's headline numbers. It is a
thin, pendulum-specific instantiation of the task-agnostic pipeline in
``run_benchmark.py`` -- see that script (or ``demo.ipynb``) for the same
pipeline running on the cart-pole and planar-quadrotor benchmark tasks, and
for the MPPI planner.

Pipeline
--------
1. Collect exploratory transitions from the *true* system (unknown friction,
   drag, heavier bob) under a correlated random policy.
2. Train a physics-residual deep-ensemble world model on the residuals
   between observations and the misspecified analytic prior.
3. Run receding-horizon CEM-MPC for swing-up with three world models:
     a) the misspecified analytic prior alone (no learning),
     b) the residual ensemble, optimistic (no uncertainty penalty),
     c) the residual ensemble with the epistemic-uncertainty penalty.
4. Report one-step model error and closed-loop control cost.

Runs in a few minutes on a laptop CPU. Reproduce with:

    python experiments/run_swingup.py --seed 0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from phyres.data import collect_random_rollouts
from phyres.ensemble import ResidualEnsembleWorldModel
from phyres.environments.base import wrap_angle
from phyres.environments.pendulum import make_task
from phyres.planners import CEMPlanner


def one_step_error(predict_fn, prior, states, actions, next_states) -> float:
    """RMS one-step prediction error on held-out transitions."""
    pred = predict_fn(states, actions)
    err = prior.wrap_state(pred - next_states)
    return float(np.sqrt(np.mean(err**2)))


def run_episode(true_system, planner, episode_len: int, s0: np.ndarray, cost_fn):
    """Closed-loop rollout of a planner on the true system."""
    planner.reset()
    s = true_system.reset(state=s0)
    total_cost = 0.0
    trajectory = [s.copy()]
    for _ in range(episode_len):
        a = planner.plan(s)
        total_cost += float(cost_fn(s[None, :], a[None, :])[0])
        s = true_system.step(a)
        trajectory.append(s.copy())
    final_theta = abs(wrap_angle(np.array([s[0]]))[0])
    return total_cost, final_theta, np.array(trajectory)


class _PriorOnlyModel:
    """Adapter: analytic prior exposed through the world-model interface."""

    def __init__(self, prior):
        self.prior = prior

    def predict(self, state, action):
        return self.prior.forward(state, action), np.zeros(state.shape[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=40, help="data episodes")
    parser.add_argument("--epochs", type=int, default=300, help="training epochs")
    parser.add_argument("--eval-len", type=int, default=250, help="control steps")
    parser.add_argument("--out", type=Path, default=Path("results.json"))
    args = parser.parse_args()

    task = make_task(seed=args.seed)
    true_system, prior = task.true_system, task.prior

    print("[1/4] Collecting exploratory data from the true system ...")
    states, actions, next_states = collect_random_rollouts(
        true_system, prior.action_low, prior.action_high,
        n_episodes=args.episodes, seed=args.seed,
    )
    n_train = int(0.9 * len(states))
    train = (states[:n_train], actions[:n_train], next_states[:n_train])
    test = (states[n_train:], actions[n_train:], next_states[n_train:])
    print(f"      {n_train} train / {len(states) - n_train} held-out transitions")

    print("[2/4] Training the physics-residual ensemble ...")
    t0 = time.time()
    model = ResidualEnsembleWorldModel(prior, n_members=5, seed=args.seed)
    histories = model.fit(*train, epochs=args.epochs)
    print(f"      done in {time.time() - t0:.1f}s "
          f"(final member losses: {[round(h[-1], 5) for h in histories]})")

    err_prior = one_step_error(prior.forward, prior, *test)
    err_model = one_step_error(lambda s, a: model.predict(s, a)[0], prior, *test)
    print(f"      held-out one-step RMSE  prior: {err_prior:.4f}  "
          f"residual ensemble: {err_model:.4f}  "
          f"({err_prior / err_model:.1f}x reduction)")

    print(f"[3/4] Closed-loop swing-up (hanging start, {args.eval_len} steps) ...")
    variants = {
        "prior_only_mpc": (_PriorOnlyModel(prior), 0.0),
        "residual_optimistic_mpc": (model, 0.0),
        "residual_uncertainty_mpc": (model, 2.0),
    }
    results: dict[str, dict[str, float]] = {}
    for name, (wm, lam) in variants.items():
        planner = CEMPlanner(
            wm, task.cost_fn,
            action_dim=prior.action_dim, action_low=prior.action_low, action_high=prior.action_high,
            horizon=task.plan_horizon, n_samples=300, n_elites=30, n_iters=4,
            uncertainty_weight=lam, seed=args.seed,
        )
        t0 = time.time()
        cost, final_theta, _ = run_episode(
            true_system, planner, args.eval_len, task.initial_state, task.cost_fn
        )
        success = final_theta < 0.25
        results[name] = {
            "closed_loop_cost": round(cost, 1),
            "final_angle_rad": round(float(final_theta), 3),
            "success": bool(success),
        }
        print(f"      {name:28s} cost={cost:8.1f}  "
              f"|theta_T|={final_theta:5.3f} rad  "
              f"{'UPRIGHT' if success else 'failed'}  ({time.time() - t0:.0f}s)")

    print("[4/4] Writing results ...")
    payload = {
        "seed": args.seed,
        "one_step_rmse": {"prior": err_prior, "residual_ensemble": err_model},
        "control": results,
    }
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"      saved to {args.out}")


if __name__ == "__main__":
    main()
