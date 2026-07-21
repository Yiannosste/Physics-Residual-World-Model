# Scientific Background

**PhyRes — Physics-Residual World Models with Calibrated Epistemic Uncertainty for Safe Model-Based Control**

This document explains the scientific foundations of the repository: why hybrid
physics–neural world models matter for Physical AI, the mathematical formulation
of every component, how the pieces compose into an uncertainty-aware
control loop, and how that engine generalises into a **task-agnostic benchmark
suite** with two planners and an online continual-adaptation mechanism.
Citations to the primary literature are given throughout.

---

## 1. Motivation: the world-model bottleneck in Physical AI

Physical AI — embodied systems that perceive, predict, and act in the physical
world — has converged on a common architectural idea: the **world model**, an
internal, action-conditioned simulator of environment dynamics that an agent
can query instead of the real world (Ha & Schmidhuber, 2018; Hafner et al.,
2023; LeCun, 2022). World models underpin the current generation of
model-based reinforcement learning (Dreamer, TD-MPC2), video-generative
foundation models for robotics (NVIDIA Cosmos, Genie), and modern
model-predictive control pipelines.

Two failure modes dominate when world models meet real robots:

1. **Sample inefficiency and physical implausibility of purely learned
   models.** A neural network trained from scratch must rediscover facts that
   Newton wrote down in 1687 — energy conservation, rigid-body kinematics,
   gravity. On the low-data regimes typical of hardware, it does so poorly,
   and its errors compound exponentially over a planning horizon (Venkatraman
   et al., 2015; Lambert et al., 2022).

2. **Misspecification of purely analytic models.** First-principles models are
   sample-free but systematically wrong: identified inertial parameters carry
   error, and phenomena such as Coulomb friction, aerodynamic drag, cable
   forces, backlash, ground effect, and contact are notoriously hard to model.
   Controllers optimised against a misspecified simulator exploit its errors —
   the *sim-to-real gap* (Zhao et al., 2020).

The scientific position of this repository is that these two failure modes are
**complementary**, and that the correct object to learn is not the dynamics
but the **residual between reality and the best available analytic prior** —
together with a *calibrated estimate of what the learned component does not
know*, so that a planner can be made risk-aware rather than optimistic. To
show this position is not an artefact of any one control problem, the
repository implements the idea as a **task-agnostic engine** (§2.2) and
instantiates it on three structurally different benchmark tasks (§2.3).

---

## 2. Problem formulation

We consider a discrete-time controlled dynamical system with state
$s_t \in \mathbb{R}^n$ and action $a_t \in \mathbb{R}^m$:

$$
s_{t+1} = F^\*(s_t, a_t), \qquad F^\* \text{ unknown},
$$

from which we may sample a modest dataset of transitions
$\mathcal{D} = \{(s_i, a_i, s_i')\}_{i=1}^N$ collected by an exploratory
policy. The goal is receding-horizon optimal control:

$$
\min_{a_{0:H-1}} \mathbb{E}\left[\sum_{t=0}^{H-1} c(s_t, a_t)\right]
\quad \text{s.t.} \quad s_{t+1} = F^\*(s_t, a_t),
$$

where the expectation is over model uncertainty, since $F^\*$ must be replaced
by a learned surrogate $\hat{F}$.

### 2.1 The abstract task contract

Every benchmark task is defined by a matched pair of objects
(`src/phyres/environments/base.py`):

* a **`PhysicsPrior`** — a stateless, batched, analytic (and deliberately
  misspecified) map $f_{\text{phys}}(s, a)$, together with a declaration of
  which state dimensions are angular (`circular_dims`) and how $(s, a)$ are
  featurised for the residual network (`features`, default: raw
  concatenation, overridden to feed $(\sin, \cos)$ pairs for angular states);
* a **`PhysicalSystem`** — a stateful "real world" simulator, advanced one
  step at a time via `step(action)`, that shares the prior's state/action
  convention but adds whatever unmodeled phenomena constitute the reality
  gap.

Neither the residual ensemble (§3–4) nor either planner (§5) ever inspects a
concrete task: they are written once against this contract and are correct
for any task that implements it, which is what makes "benchmark suite" more
than a figure of speech here — adding a fourth task is adding a fourth pair of
classes, not touching the engine.

### 2.2 The testbed systems

The repository instantiates the contract on three canonical underactuated /
nonlinear control problems, chosen to be structurally different (2, 4 and 6
state dimensions; 1 and 2 actuators; one, one and two kinds of unmodeled
nonlinearity respectively) so that the hybrid-model claims are not an
artefact of any single system's structure.

**Pendulum swing-up** (`environments/pendulum.py`) — the minimal system that
is (i) nonlinear, (ii) underactuated (the torque limit $u_{\max}$ is below
the gravity torque, so the controller must *pump energy* through multiple
swings), and (iii) equipped with a genuinely hard reality gap. With $\theta$
the angle from the upright equilibrium, the **true** system is

$$
m^\* \ell^{\*2} \ddot{\theta} = m^\* g \ell^\* \sin\theta +
u - \underbrace{b \dot\theta}_{\text{viscous}} -
\underbrace{c_d \dot\theta |\dot\theta|}_{\text{quadratic drag}} -
\underbrace{\mu \mathrm{sgn}(\dot\theta)}_{\text{Coulomb}} ,
\qquad |u| \le u_{\max},
$$

while the **analytic prior** is the frictionless textbook model with
*misidentified* parameters $m \ne m^\*$, $\ell \ne \ell^\*$:
$m \ell^2 \ddot{\theta} = m g \ell \sin\theta + u$.

**Cart-pole balancing** (`environments/cartpole.py`) — a 4-state, 1-actuator
system ($x, \dot x, \theta, \dot\theta$, $\theta = 0$ upright) whose prior is
the textbook rigid-body cart-pole (Florian, 2007), integrated with *no*
friction anywhere. The **true** system adds three unmodeled effects absent
from that idealisation: Coulomb + viscous friction at the cart-track
interface, a force *dead-zone* modelling cable slack / backlash in the
actuation linkage (forces below a slack threshold fail to transmit at all —
a sharp but bounded nonlinearity, structurally similar to the pendulum's
Coulomb term), and extra viscous damping at the pole hinge.

**Planar quadrotor hover** (`environments/quadrotor2d.py`) — a 6-state
($x, y, \phi, \dot x, \dot y, \dot\phi$), 2-actuator (per-rotor thrust)
system whose prior assumes textbook vacuum kinematics: two point thrusts
acting through the centre of mass, no aerodynamic interaction with the
surrounding air. The **true** system adds quadratic aerodynamic drag on both
translational velocities and a *ground-effect* lift boost that grows sharply
as the vehicle approaches the floor — a well-documented, strongly nonlinear
near-ground phenomenon in real rotorcraft (Powers et al., 2013) that a
vacuum-kinematics prior has no mechanism to represent — plus a compliant
(spring-damper) floor contact force. This is the highest-dimensional, most
actuator-coupled testbed in the suite (the roll-torque gain
`arm / inertia` is large, so exploration and planning noise scales must be
tuned relative to hover thrust rather than the raw actuator range; see
`explore_center` / `mppi_kwargs` on the task).

All three systems are discretised with semi-implicit (symplectic) Euler,
which preserves the qualitative energy behaviour of the continuous system far
better than explicit Euler at the same step size (Stewart, 2000). In every
case, the discrepancy between prior and true system — misidentified
parameters plus one or more unmodeled nonlinear force/torque terms — is the
*reality gap* that learning must close.

---

## 3. Physics-residual world models

### 3.1 Formulation

Rather than learning $F^\*$ from scratch, we posit the **hybrid decomposition**

$$
\hat{F}(s, a) = f_{\text{phys}}(s, a) + g_\psi(s, a),
$$

where $f_{\text{phys}}$ is the task's analytic prior and $g_\psi$ is a small
multilayer perceptron trained on the **residual targets**

$$
r_i = s_i' - f_{\text{phys}}(s_i, a_i),
$$

wrapped through the prior's `wrap_state` on any circular dimension so the
target is the geodesic error on the circle rather than the raw arithmetic
difference. Residual targets are z-score normalised; training minimises
mean-squared error with Adam (Kingma & Ba, 2015), implemented with manual
backpropagation (`src/phyres/nn.py`) — no framework autograd.

This *residual physics* idea has repeatedly proven to be one of the most
effective forms of physical inductive bias: it appears as residual dynamics
learning for control (Zeng et al., 2020, "TossingBot"; Ajay et al., 2018),
grey-box system identification (Ljung, 2010), hybrid ODE modelling
(Yin et al., 2021, "APHYNITY"), and simulator-gap correction
(Golemo et al., 2018).

### 3.2 Why the decomposition helps: an inductive-bias argument

Let $F^\* = f_{\text{phys}} + r^\*$ define the true residual. Three properties
make $r^\*$ a strictly easier learning target than $F^\*$:

- **Smaller magnitude and complexity.** The prior already accounts for the
  dominant terms (gravity, inertia, vacuum kinematics). What remains —
  friction, drag, cable slack, ground effect, parameter offsets — is
  lower-amplitude and, empirically, lower-frequency as a function of
  $(s, a)$. Standard generalisation bounds scale with the target's norm in
  the hypothesis space; shrinking the target shrinks the sample complexity.
- **Graceful out-of-distribution behaviour.** Where data is absent, a
  weight-decayed MLP's output decays toward a constant, so $\hat F$ falls
  back to $f_{\text{phys}} + \text{const}$ — a physically sensible
  extrapolation. A from-scratch network falls back to *nothing*.
- **Compounding-error mitigation.** In $H$-step rollouts, one-step errors
  compound multiplicatively (Venkatraman et al., 2015). Keeping the
  high-gain part of the map analytic means learned error is injected only
  through the (small) residual channel.

A useful, falsifiable corollary of this argument shows up empirically on the
quadrotor task: the prior already predicts the roll dynamics
($\phi, \dot\phi$) *exactly* (the true system adds no unmodeled torque), and
the residual ensemble learns essentially zero correction there while making
a large correction on the translational dimensions where the true drag and
ground-effect terms live — i.e. the residual is selective, not a blanket
correction (see `demo.ipynb`, §4).

### 3.3 Relation to alternative physics priors

Residual modelling is one point in a spectrum of physics-informed learning:
Lagrangian/Hamiltonian neural networks hard-code energy structure
(Greydanus et al., 2019; Cranmer et al., 2020), PINNs impose PDE residuals as
soft losses (Raissi et al., 2019), and differentiable simulators make the
prior's parameters trainable (de Avila Belbute-Peres et al., 2018). The
residual decomposition is distinguished by being *non-invasive*: it accepts
any black-box prior — including an off-the-shelf simulator — and never has to
fight it, because the network can represent exactly the part of reality the
prior misses.

---

## 4. Epistemic uncertainty via deep ensembles

### 4.1 Aleatoric vs. epistemic

A model-based agent faces two kinds of uncertainty (Kendall & Gal, 2017):
**aleatoric** (irreducible stochasticity of the system) and **epistemic**
(reducible ignorance due to finite data). For control, epistemic uncertainty
is the safety-critical one: it flags *where the model may be arbitrarily
wrong*, which is precisely where a planner must not go.

### 4.2 Ensembles as approximate posteriors

We train $K$ residual networks $\{g_{\psi_k}\}_{k=1}^K$ that differ in random
initialisation and in bootstrap resamples of $\mathcal{D}$. Deep ensembles of
this form are the empirically strongest practical approximation to Bayesian
posterior predictive uncertainty at small $K$ (Lakshminarayanan et al., 2017;
Osband et al., 2016; Fort et al., 2019), and are the de-facto standard in
model-based RL through PETS (Chua et al., 2018) and MBPO (Janner et al.,
2019). The ensemble's **disagreement** is accumulated per state dimension,
using the circular mean/deviation on any dimension the task declares angular
and the ordinary arithmetic mean/std elsewhere:

$$
d(s, a) = \sum_{j=1}^{n} \mathrm{dev}_k\big[\hat s^{(k)}_j(s,a)\big],
\qquad
\mathrm{dev}_k[\cdot] =
\begin{cases}
\text{circular std (via } \mathrm{atan2}\text{ of the member outputs)} & j \in \text{circular dims} \\
\text{ordinary std} & \text{otherwise,}
\end{cases}
$$

which reduces to the pendulum-specific formula used in earlier revisions of
this repository (angular deviation for $\theta$, ordinary deviation for
$\omega$) as the $n=2$, `circular_dims=(0,)` special case, and generalises
unchanged to the cart-pole's one angular dimension and the quadrotor's one
angular dimension embedded in a 6D state. On well-covered data all members
agree (they interpolate the same targets), while off-distribution their
extrapolations diverge; a unit test parametrised over all three tasks
verifies the defining property directly: disagreement on far-out-of-distribution
queries exceeds in-distribution disagreement by a large factor
(`test_disagreement_grows_out_of_distribution`).

An important subtlety: because every member shares the *same* physics prior,
disagreement measures uncertainty **only about the residual** — the prior is
treated as trusted structure. This is exactly the desired semantics for a
hybrid model: the agent is certain about Newton, uncertain about friction (or
drag, or ground effect, or cable slack).

---

## 5. Planning: two risk-aware sampling-based MPC schemes

### 5.1 Cross-Entropy Method planning

Control is receding-horizon: at each step the agent optimises an $H$-step
action sequence, executes the first action, and re-plans. `CEMPlanner`
(`src/phyres/planners/cem.py`) implements the Cross-Entropy Method
(Rubinstein & Kroese, 2004; Botev et al., 2013), a derivative-free
evolutionary scheme that iterates:

1. sample $N$ action sequences from a Gaussian
   $\mathcal{N}(\mu_{1:H}, \sigma_{1:H}^2)$, clipped to the task's
   per-dimension action bounds;
2. roll each sequence through the world model and score it;
3. refit $(\mu, \sigma)$ to the top-$E$ *elite* sequences.

CEM-MPC over learned models is the backbone of PETS (Chua et al., 2018),
PlaNet (Hafner et al., 2019), and TD-MPC (Hansen et al., 2022); it handles
non-convex, non-differentiable objectives and box-constrained actions
natively. The sampling mean is warm-started from the previous step's shifted
solution, which turns re-planning into a cheap refinement.

### 5.2 Model Predictive Path Integral control

`MPPIPlanner` (`src/phyres/planners/mppi.py`) implements Model Predictive
Path Integral control (Williams, Aldrich & Theodorou, 2017), an
information-theoretic alternative to CEM's hard elite selection. Rather than
discarding all but the top-$E$ samples, every one of the $N$ sampled
trajectories contributes to the updated nominal sequence in proportion to
$\exp(-\text{cost} / \tau)$ for a temperature $\tau$:

$$
\mu_{1:H} \leftarrow \mu_{1:H} + \sum_{n=1}^{N} w_n \varepsilon_n,
\qquad
w_n = \frac{\exp\big(-J_n / \tau\big)}{\sum_{n'} \exp\big(-J_{n'} / \tau\big)},
$$

where $\varepsilon_n$ is sample $n$'s (post-clipping) perturbation from the
nominal sequence and $J_n$ its rolled-out cost. Because *every* sample pulls
the mean a little rather than only a truncated elite set redefining both mean
and variance each iteration, the resulting control sequence has markedly
lower step-to-step jerk than CEM's — the practical motivation for offering
both planners in the suite rather than picking one. `demo.ipynb` §6
visualises this directly on the quadrotor's rotor-thrust commands.

### 5.3 Risk-aware objective

Both planners optimise the same objective, evaluated by rolling the world
model's ensemble-mean prediction forward and accumulating disagreement along
the way:

$$
J(a_{0:H-1}) = \sum_{t=0}^{H-1}
   c(\hat{s}_t, a_t) + \lambda d(\hat{s}_t, a_t).
$$

The penalty term implements **pessimism in the face of epistemic
uncertainty**: trajectories that pass through poorly-modelled regions are
charged extra, so the optimiser prefers plans the model family collectively
vouches for. This is the *opposite* sign convention from exploration bonuses
(curiosity rewards disagreement; Pathak et al., 2019) and aligns with
conservative/robust model-based control (Yu et al., 2020, MOPO; Kidambi
et al., 2020, MOReL), where an uncertainty-penalised surrogate lower-bounds
the true return under assumptions on the uncertainty estimator's calibration.
Setting $\lambda = 0$ recovers the standard optimistic planner and exposes
the failure mode this line of work exists to prevent: *model exploitation*,
in which the optimiser finds actions that are excellent inside the model and
poor in reality — demonstrated directly by the `prior_only` variant in
every task's results, which plans against $f_{\text{phys}}$ alone
($\lambda$ irrelevant, since the prior has no disagreement signal) and fails.

Task-specific cost functions are quadratic regulator costs around the goal
state (upright pendulum, centred upright cart-pole, target-altitude
level-attitude quadrotor hover), each with a small control-effort term.

---

## 6. Online continual adaptation

### 6.1 Motivation

The ensemble in §3–4 is trained once, offline, on an exploratory dataset. A
deployed controller, however, keeps observing new transitions while it
executes — and the reality gap it is compensating for is not necessarily
stationary (tyre wear, a battery-voltage-dependent motor constant, a
temperature-dependent friction coefficient). Ignoring these observations
wastes information the agent already has; naively retraining from scratch on
them is both computationally wasteful and, without care, invites
*catastrophic forgetting* of the offline-trained residual (McCloskey &
Cohen, 1989).

### 6.2 Warm-started replay updates

`ResidualEnsembleWorldModel.online_update` (`src/phyres/ensemble.py`)
implements a lightweight middle ground, in the spirit of online model
adaptation for real-world control (Nagabandi et al., 2019) and
replay-buffer-based continual learning (Rolnick et al., 2019):

1. each freshly observed transition $(s_t, a_t, s_{t+1})$ is pushed into a
   fixed-capacity rolling replay buffer, alongside the offline training data
   already folded in by `fit`;
2. a minibatch is sampled from that buffer (new transitions plus a random
   slice of history, so old and new evidence both influence the update);
3. every ensemble member takes a small number of low-learning-rate Adam
   steps against that minibatch, **reusing its existing Adam moment
   estimates** (`MLP.fit` persists `_adam_state` across calls) — a warm
   start, not a restart.

Because the update is warm-started, low-learning-rate, and mixes in replayed
history rather than training exclusively on the newest transition, it tracks
a slowly drifting reality gap without discarding what the offline phase
already learned. `test_online_update_adapts_to_a_shifted_reality_gap`
verifies this directly: an ensemble trained on one friction regime is shown a
handful of transitions from a substantially higher-friction regime it has
never seen, and `online_update` measurably closes the resulting prediction
gap. The residual normalisation statistics (`res_mean`, `res_std`) are kept
fixed at their offline values — a deliberate simplicity/robustness trade-off
documented as a limitation below.

---

## 7. Experimental design and falsifiable claims

The experiments (`experiments/run_swingup.py` for the original single-task
pipeline; `experiments/run_benchmark.py --task <name> --planner
<cem|mppi>` for any task/planner combination; `demo.ipynb` for a narrated
walkthrough on the quadrotor) are built around four pre-registered,
falsifiable claims, each backed by a unit test or a reported metric:

- **C1 — Residual learning closes the reality gap, on every task.** The
  hybrid model's held-out one-step RMSE is a large factor below the
  misspecified prior's, on the pendulum, cart-pole *and* quadrotor.
  *(Metric: one-step RMSE, prior vs. ensemble; test:
  `test_residual_ensemble_beats_prior`, parametrised over all three tasks.)*
- **C2 — Disagreement detects distribution shift, on every task.** Ensemble
  disagreement on out-of-distribution state-actions exceeds the
  in-distribution level severalfold, on every task in the suite.
  *(Test: `test_disagreement_grows_out_of_distribution`, parametrised.)*
- **C3 — The hybrid model, not the prior, supports control on tasks that
  require sustained, systematically correct actuation.** On the pendulum
  (which must pump energy through several swings against a prior that
  underestimates dissipation) and the cart-pole (where CEM against the prior
  alone is measurably worse than against the residual ensemble, which the
  uncertainty penalty improves on further), closed-loop cost is a clear,
  ordered function of model quality: prior-only worst, residual-optimistic
  better, residual-uncertainty best or tied. *(Metric: closed-loop cost per
  planner variant, reported by `run_benchmark.py`; unit-tested directly for
  the pendulum's near-upright stabilisation under both planners via
  `test_planner_stabilises_pendulum_near_upright`.)* Section 7's limitations
  discuss an instructive counter-example on the quadrotor task.
- **C4 — Online adaptation recovers accuracy under a shifted reality gap.**
  A residual ensemble trained under one set of unmodeled-friction parameters
  measurably reduces its prediction error on a shifted-friction regime after
  a handful of warm-started `online_update` steps, without retraining from
  scratch. *(Test: `test_online_update_adapts_to_a_shifted_reality_gap`.)*

Reproducibility is treated as a first-class constraint: the entire pipeline
is deterministic given `--seed`, dependency-minimal (NumPy only for the
engine, with manual backpropagation — no framework autograd; `matplotlib` /
`seaborn` / `jupyter` are demo-only, opt-in via the `notebook` extra), and
small enough to audit line by line.

### Limitations and outlook

**One-step accuracy does not automatically buy closed-loop reward, and the
quadrotor task is an instructive counter-example.** Despite a 2-2.5x
held-out one-step RMSE reduction (C1) and correctly structured epistemic
uncertainty (C2), the hybrid model's closed-loop hover-recovery cost is
statistically indistinguishable from — sometimes marginally worse than — the
vacuum-kinematics prior alone, whether the vehicle starts near the target
altitude or deliberately close to the ground where the ground-effect gap is
largest. The likely explanation is that hover recovery from a small
perturbation is a well-conditioned regulation problem: with MPC replanning
against the *true* observed state every 20 ms, a systematic one-step bias
gets corrected by feedback before it can compound, so the prior's error
never gets the chance to accumulate the way it does across the pendulum's
multi-swing, precisely-timed energy pumping (where the prior's dissipation
error is directly why it fails to swing up at all). This is a useful, honest
negative result, not a hidden one: it shows C1/C2 (representation quality)
and C3 (closed-loop payoff) are genuinely separate claims that do not imply
each other, and that the payoff of a better model is largest exactly where
the task is *least* forgiving of model error — open-loop-flavoured,
long-horizon, precisely-timed control, not short-horizon regulation near a
fixed point.

The testbeds are deliberately low-dimensional; the point is scientific
isolation of the hybrid-model question across structurally different
systems, not benchmark coverage at robotics scale. The disagreement penalty
uses a fixed weight $\lambda$ (and, for MPPI, a fixed temperature and noise
scale) rather than a calibrated risk level or an adaptive schedule; ensembles
quantify uncertainty about the residual but not about the prior's *structure*
(e.g. a missing contact mode); mean-propagation through the horizon ignores
aleatoric noise (trajectory-sampling schemes as in PETS would capture it);
and the quadrotor's large roll-torque gain means its planner hyperparameters
are hand-tuned relative to hover thrust rather than derived automatically
from the task (`Quadrotor2DPrior`/`Quadrotor2DSystem` expose the values that
matter — `hover_thrust`, `arm`, `inertia` — for exactly this reason). Online
adaptation (§6) keeps the residual normalisation statistics fixed at their
offline values, so it can track a moderate drift but is not designed for
regime changes large enough to invalidate that normalisation. Natural
extensions: contact-rich systems where $f_{\text{phys}}$ is a differentiable
simulator, adaptive/calibrated $\lambda$ and MPPI temperature, trajectory
sampling for aleatoric uncertainty, and periodic re-estimation of residual
normalisation statistics during online adaptation.

---

## 8. References

- Ajay, A., et al. (2018). Augmenting physical simulators with stochastic neural networks. *IROS*.
- Botev, Z., et al. (2013). The cross-entropy method for optimization. *Handbook of Statistics*, 35.
- Chua, K., Calandra, R., McAllister, R., Levine, S. (2018). Deep RL in a handful of trials using probabilistic dynamics models (PETS). *NeurIPS*.
- Cranmer, M., et al. (2020). Lagrangian neural networks. *ICLR workshop*.
- de Avila Belbute-Peres, F., et al. (2018). End-to-end differentiable physics for learning and control. *NeurIPS*.
- Florian, R. V. (2007). Correct equations for the dynamics of the cart-pole system. *Technical report, Center for Cognitive and Neural Studies*.
- Fort, S., Hu, H., Lakshminarayanan, B. (2019). Deep ensembles: a loss landscape perspective. *arXiv:1912.02757*.
- Golemo, F., et al. (2018). Sim-to-real transfer with neural-augmented robot simulation. *CoRL*.
- Greydanus, S., Dzamba, M., Yosinski, J. (2019). Hamiltonian neural networks. *NeurIPS*.
- Ha, D., Schmidhuber, J. (2018). World models. *NeurIPS*.
- Hafner, D., et al. (2019). Learning latent dynamics for planning from pixels (PlaNet). *ICML*.
- Hafner, D., et al. (2023). Mastering diverse domains through world models (DreamerV3). *arXiv:2301.04104*.
- Hansen, N., Wang, X., Su, H. (2022). Temporal difference learning for model predictive control (TD-MPC). *ICML*.
- Janner, M., et al. (2019). When to trust your model: model-based policy optimization (MBPO). *NeurIPS*.
- Kendall, A., Gal, Y. (2017). What uncertainties do we need in Bayesian deep learning? *NeurIPS*.
- Kidambi, R., et al. (2020). MOReL: model-based offline reinforcement learning. *NeurIPS*.
- Kingma, D., Ba, J. (2015). Adam: a method for stochastic optimization. *ICLR*.
- Lakshminarayanan, B., Pritzel, A., Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. *NeurIPS*.
- Lambert, N., et al. (2022). Investigating compounding prediction errors in learned dynamics models. *arXiv:2203.09637*.
- LeCun, Y. (2022). A path towards autonomous machine intelligence. *OpenReview*.
- Ljung, L. (2010). Perspectives on system identification. *Annual Reviews in Control*, 34(1).
- McCloskey, M., Cohen, N. J. (1989). Catastrophic interference in connectionist networks: the sequential learning problem. *Psychology of Learning and Motivation*, 24.
- Nagabandi, A., Clavera, I., Liu, S., Fearing, R. S., Abbeel, P., Levine, S., Finn, C. (2019). Learning to adapt in dynamic, real-world environments through meta-reinforcement learning. *ICLR*.
- Osband, I., et al. (2016). Deep exploration via bootstrapped DQN. *NeurIPS*.
- Pathak, D., Gandhi, D., Gupta, A. (2019). Self-supervised exploration via disagreement. *ICML*.
- Powers, C., Mellinger, D., Kushleyev, A., Kothmann, B., Kumar, V. (2013). Influence of aerodynamics and proximity effects in quadrotor flight. *ISER*.
- Raissi, M., Perdikaris, P., Karniadakis, G. (2019). Physics-informed neural networks. *J. Comp. Physics*, 378.
- Rolnick, D., Ahuja, A., Schwarz, J., Lillicrap, T., Wayne, G. (2019). Experience replay for continual learning. *NeurIPS*.
- Rubinstein, R., Kroese, D. (2004). *The Cross-Entropy Method*. Springer.
- Stewart, D. (2000). Rigid-body dynamics with friction and impact. *SIAM Review*, 42(1).
- Venkatraman, A., Hebert, M., Bagnell, J. A. (2015). Improving multi-step prediction of learned time series models. *AAAI*.
- Williams, G., Aldrich, A., Theodorou, E. A. (2017). Model predictive path integral control: from theory to parallel computation. *Journal of Guidance, Control, and Dynamics*, 40(2).
- Yin, Y., et al. (2021). Augmenting physical models with deep networks for complex dynamics forecasting (APHYNITY). *ICLR*.
- Yu, T., et al. (2020). MOPO: model-based offline policy optimization. *NeurIPS*.
- Zeng, A., et al. (2020). TossingBot: learning to throw arbitrary objects with residual physics. *IEEE T-RO*.
- Zhao, W., Queralta, J. P., Westerlund, T. (2020). Sim-to-real transfer in deep reinforcement learning for robotics: a survey. *SSCI*.
