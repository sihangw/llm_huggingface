import numpy as np
from poly_theta_generator import PolyThetaConfig, PolyFromTheta
from poly_theta_sampler import ThetaSampler, ThetaSamplerConfig
from CompositeScore import CompositeScoreConfig, CompositeTrajectoryScorer
from itertools import combinations_with_replacement

def make_cost_fn(theta_inst):
    """
    Build a per-step cost function cost_fn(x, u, u_prev=None, terminal=False)
    from a theta_inst sampled by ThetaSampler.
    """

    meta = theta_inst["meta"]
    init_goal = theta_inst["init_goal"]

    X_GOAL = np.asarray(init_goal["x_end"], dtype=float)  # target state
    U_MAX  = float(meta["u_max"])                         # control scale / limit
    R_SAFE = float(meta["r"])                             # safe radius in state space

    # --- fixed weights (v0 baseline; can later be tuned / LLM-tuned) ---
    W_S_TRACK   = 1.0
    W_S_SAFE    = 5.0
    W_S_U       = 0.1
    W_S_SMOOTH  = 0.1

    W_T_TRACK   = 10.0
    W_T_SAFE    = 20.0
    W_T_U       = 0.1
    W_T_SMOOTH  = 0.0  # usually don't care about smoothness at terminal

    def cost_fn(x, u, u_prev=None, terminal=False):
        """
        x: (n,) ndarray
        u: (m,) ndarray
        u_prev: (m,) ndarray or None
        terminal: bool
        """
        x = np.asarray(x, dtype=float)
        u = np.asarray(u, dtype=float)

        # Tracking error (quadratic)
        e_track = np.sum((x - X_GOAL) ** 2)

        # Control magnitude, normalized by U_MAX
        e_u = np.sum((u / U_MAX) ** 2)

        # Smoothness (quadratic in u - u_prev)
        if u_prev is None:
            e_smooth = 0.0
        else:
            u_prev = np.asarray(u_prev, dtype=float)
            e_smooth = np.sum((u - u_prev) ** 2)

        # Safe-region violation: ||x||_inf > R_SAFE → quadratic penalty
        inf_norm = np.max(np.abs(x))
        gamma = max(0.0, inf_norm - R_SAFE)
        e_safe = gamma ** 2

        # Choose weights for stage vs terminal
        if terminal:
            w_track   = W_T_TRACK
            w_safe    = W_T_SAFE
            w_u       = W_T_U
            w_smooth  = W_T_SMOOTH
        else:
            w_track   = W_S_TRACK
            w_safe    = W_S_SAFE
            w_u       = W_S_U
            w_smooth  = W_S_SMOOTH

        cost = (
            w_track  * e_track +
            w_safe   * e_safe +
            w_u      * e_u +
            w_smooth * e_smooth
        )
        return float(cost)

    return cost_fn

def rollout_cost(theta_inst, cost_fn, x0, U_seq):
    """
    Compute total cost of a control sequence U_seq under theta_inst dynamics.

    theta_inst : dict from ThetaSampler.sample_instance()
    cost_fn    : per-step cost, from make_cost_fn(...)
    x0         : initial state, shape (n,)
    U_seq      : array of controls, shape (N, m)

    Returns
    -------
    total_cost : float
    X_traj     : array of states, shape (N+1, n)
    """

    step = theta_inst["step"]   # closure: step(x, u) -> x_next
    N    = theta_inst["meta"]["N"]

    U_seq = np.asarray(U_seq, dtype=float)
    assert U_seq.shape[0] == N

    n = theta_inst["meta"]["n"]
    x = np.asarray(x0, dtype=float).reshape(n)

    X_traj = np.zeros((N + 1, n), dtype=float)
    X_traj[0] = x

    total_cost = 0.0
    u_prev = None

    for k in range(N):
        u = U_seq[k]
        terminal = (k == N - 1)
        total_cost += cost_fn(x, u, u_prev=u_prev, terminal=terminal)

        # step the dynamics
        x = step(x, u)
        X_traj[k + 1] = x
        u_prev = u

    return float(total_cost), X_traj

def mppi_update(
    theta_inst,
    cost_fn,
    x0,
    U_mean,
    num_samples=256,
    lambda_=1.0,
    noise_sigma=0.5,
    u_min=None,
    u_max=None,
    rng=None,
):
    """
    Perform one MPPI update of the control sequence U_mean at state x0.

    Parameters
    ----------
    theta_inst : dict
        Instance (contains step, meta, etc.).
    cost_fn : callable
        Per-step cost function.
    x0 : ndarray, shape (n,)
        Current state.
    U_mean : ndarray, shape (H, m)
        Current mean control sequence.
    num_samples : int
        Number of sampled trajectories.
    lambda_ : float
        MPPI temperature (control noise inverse temperature).
    noise_sigma : float or ndarray
        Std of additive Gaussian noise on controls.
    u_min, u_max : float or ndarray or None
        Control bounds; if None, inferred from theta_inst['meta']['u_max'].
    rng : np.random.Generator or None
        Random generator.

    Returns
    -------
    U_new : ndarray, shape (H, m)
        Updated mean control sequence.
    costs : ndarray, shape (num_samples,)
        Costs of sampled trajectories.
    """
    if rng is None:
        rng = np.random.default_rng()

    U_mean = np.asarray(U_mean, dtype=float)
    H, m = U_mean.shape

    meta = theta_inst["meta"]
    if u_max is None:
        u_max = float(meta["u_max"])
    if u_min is None:
        u_min = -float(meta["u_max"])

    u_max_arr = np.full((m,), u_max, dtype=float)
    u_min_arr = np.full((m,), u_min, dtype=float)

    # ensure noise_sigma has shape (H, m)
    noise_sigma_arr = np.ones_like(U_mean) * noise_sigma

    costs = np.zeros(num_samples, dtype=float)
    noises = np.zeros((num_samples, H, m), dtype=float)

    for i in range(num_samples):
        # sample noise
        eps = rng.normal(loc=0.0, scale=1.0, size=(H, m))
        dU = noise_sigma_arr * eps
        U_i = U_mean + dU
        U_i = np.clip(U_i, u_min_arr, u_max_arr)

        J_i, _ = rollout_cost(theta_inst, cost_fn, x0, U_i)
        costs[i] = J_i
        noises[i] = dU

    # MPPI weights
    J_min = np.min(costs)
    weights = np.exp(-(costs - J_min) / lambda_)
    weights_sum = np.sum(weights) + 1e-12
    weights = weights / weights_sum  # normalize

    # weighted update of controls
    # U_new = U_mean + sum_i w_i * dU_i
    dU_mean = np.tensordot(weights, noises, axes=(0, 0))
    U_new = U_mean + dU_mean

    # clip again just in case
    U_new = np.clip(U_new, u_min_arr, u_max_arr)

    return U_new, costs

def mppi_mpc(
    theta_inst,
    cost_fn,
    x0,
    T_outer=25,
    H=25,
    num_samples=256,
    num_iters=3,
    lambda_=1.0,
    noise_sigma=0.5,
    u_min=None,
    u_max=None,
    rng=None,
):
    """
    Closed-loop sampling-based MPC using MPPI.

    Parameters
    ----------
    theta_inst : dict
        Instance from ThetaSampler.sample_instance().
    cost_fn : callable
        Per-step cost.
    x0 : ndarray, shape (n,)
        Initial state.
    T_outer : int
        Number of MPC outer steps (length of closed-loop simulation).
    H : int
        Planning horizon (<= meta['N'] typically).
    num_samples : int
        Number of trajectories sampled per MPPI iteration.
    num_iters : int
        Number of MPPI iterations per MPC step.
    lambda_ : float
        MPPI temperature parameter.
    noise_sigma : float
        Std of control noise.
    u_min, u_max : float or None
        Control bounds; default uses +/- meta['u_max'].
    rng : np.random.Generator or None
        Random generator.

    Returns
    -------
    X_cl : ndarray, shape (T_outer+1, n)
        Closed-loop state trajectory.
    U_cl : ndarray, shape (T_outer, m)
        Applied control sequence.
    """
    if rng is None:
        rng = np.random.default_rng()

    meta = theta_inst["meta"]
    step = theta_inst["step"]
    n, m = meta["n"], meta["m"]

    if u_max is None:
        u_max = float(meta["u_max"])
    if u_min is None:
        u_min = -float(meta["u_max"])

    # initial guess for control sequence: zeros
    U_mean = np.zeros((H, m), dtype=float)

    X_cl = np.zeros((T_outer + 1, n), dtype=float)
    U_cl = np.zeros((T_outer, m), dtype=float)

    x = np.asarray(x0, dtype=float).reshape(n)
    X_cl[0] = x

    for t in range(T_outer):
        # MPPI inner loop to optimize U_mean from current state x
        for _ in range(num_iters):
            U_mean, costs = mppi_update(
                theta_inst,
                cost_fn,
                x,
                U_mean,
                num_samples=num_samples,
                lambda_=lambda_,
                noise_sigma=noise_sigma,
                u_min=u_min,
                u_max=u_max,
                rng=rng,
            )

        # apply first control
        u_apply = U_mean[0].copy()
        U_cl[t] = u_apply

        # step system
        x = step(x, u_apply)
        X_cl[t + 1] = x

        # shift control sequence for receding horizon
        # last control is repeated or set to zero
        U_mean[:-1] = U_mean[1:]
        U_mean[-1] = 0.0

    return X_cl, U_cl

from mppi_mpc import MPPIConfig, run_closed_loop_mppi

cfg = PolyThetaConfig(n=3, m=1, d_f=2, d_G=2, dt=0.1, N=25)
sampler = ThetaSampler(cfg, ThetaSamplerConfig(), seed=23)
inst  = sampler.sample_instance()  # directly solvable instance


gen = PolyFromTheta(cfg)
x0 =  inst['init_goal']['x0']
drift, inp = inst["drift"], inst["input"]
U = inst["witness_controls"]
X = gen.rollout(drift, inp, x0, U)
x_end = X[-1]
print(x_end)

f = inst['step']      # f(x, u) -> x_next
x = x0.copy()
for k in range(25):
    x = f(x, U[k])
print(x)

print("Instance x_end:", inst["init_goal"]["x_end"])
print("Diff:", x - inst["init_goal"]["x_end"])

cost_fn = make_cost_fn(inst)
x0      = inst["init_goal"]["x0"]  # or your current state

# U_seq: sampled control sequence of shape (N, m)
J, X = rollout_cost(inst, cost_fn, x0, U)
print(J)
print(X[-1])


# Assuming inst and cost_fn already created
x0 = np.asarray(inst["init_goal"]["x0"], dtype=float)

X_cl, U_cl = mppi_mpc(
    theta_inst=inst,
    cost_fn=cost_fn,
    x0=x0,
    T_outer=25,         # simulate 25 steps
    H=25,               # plan 25-step horizon
    num_samples=256,
    num_iters=3,
    lambda_=1.0,
    noise_sigma=0.3,
)

print("Final state:", X_cl[-1])
print("Goal state: ", inst["init_goal"]["x_end"])


# Example phi_target: ball of radius rho around x_goal, using Euclidean distance.
x_goal = np.asarray(inst["init_goal"]["x_end"], dtype=float)
rho    = float(inst["init_goal"]["goal_radius"])

def phi_target(x):
    # negative inside target set
    return np.linalg.norm(x - x_goal) - rho

def constraint_excess(x, u, t):
    # Example: no extra constraints -> always satisfy
    return -1.0  # <= 0 => no violation
cfg_score = CompositeScoreConfig(
    dt=inst["meta"]["dt"],
    D=inst["meta"]["rho"],          # characteristic length scale (e.g., target radius)
    V_ref=1.0,      # or something like baseline violation percentile
    beta1=0.5,
    beta2=0.2,
    beta6=0.2,
)

scorer = CompositeTrajectoryScorer(phi_target, constraint_excess, cfg_score)

result = scorer.score(X, U)

print("Composite score S:", result["Y"])
print("Success:", result["success"])
print("m_min:", result["m_min"])
print("tau_star:", result["tau_star"])
print("violations:", result["violations"])

meta = inst["meta"]
H = meta["N"]
m = meta["m"]
u_max = meta["u_max"]
rng = np.random.default_rng(0)
# Trajectory 1: "reasonable" small controls drifting toward zero
U1 = rng.uniform(-0.3 * u_max, 0.3 * u_max, size=(H, m))
# Trajectory 2: "aggressive" larger, noisy controls
U2 = rng.uniform(-u_max, u_max, size=(H, m))
# Trajectory 3:
U3 = rng.uniform(-0.5 * u_max, 0.5 * u_max, size=(H, m))

J1, X1 = rollout_cost(inst, cost_fn, x0, U1)
score1 = scorer.score(X1, U1)
J2, X2 = rollout_cost(inst, cost_fn, x0, U2)
score2 = scorer.score(X2, U2)
J3, X3 = rollout_cost(inst, cost_fn, x0, U3)
score3 = scorer.score(X3, U3)
score_cl = scorer.score(X_cl, U_cl)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.patches import Circle

def plot_2d_trajectories(theta_inst, X_list, labels=None, show_safe_region=True):
    """
    Plot multiple trajectories in 2D (first two state components),
    together with initial and target points.

    Parameters
    ----------
    theta_inst : dict
        Instance returned by ThetaSampler.sample_instance().
        Used to get x0, x_goal, and safe radius r.
    X_list : list of np.ndarray
        Each element is an array of shape (T+1, n) for one trajectory.
    labels : list of str or None
        Optional labels for each trajectory.
    show_safe_region : bool
        If True, draw a circle of radius r around origin (if n >= 2).
    """
    init_goal = theta_inst["init_goal"]
    x0    = np.asarray(init_goal["x0"], dtype=float)
    x_goal = np.asarray(init_goal["x_end"], dtype=float)
    # r_safe = float(meta["r"])

    if labels is None:
        labels = [f"traj {i}" for i in range(len(X_list))]

    fig, ax = plt.subplots(figsize=(6, 6))

    # Trajectories
    for X, label in zip(X_list, labels):
        X = np.asarray(X, dtype=float)
        assert X.shape[1] >= 2, "Need at least 2 state dimensions to make 2D plot."
        ax.plot(X[:, 0], X[:, 1], label=label)

    # Initial point (same for all trajectories if you used same x0)
    ax.scatter(x0[0], x0[1], c="green", s=80, marker="o", zorder=5, label="start")

    # Target point
    ax.scatter(x_goal[0], x_goal[1], c="red", s=100, marker="*", zorder=6, label="target")

    # # Optional safe region (ball in first two coords)
    # if show_safe_region:
    #     circ = Circle((0.0, 0.0), r_safe, edgecolor="gray", facecolor="none", linestyle="--", alpha=0.6)
    #     ax.add_patch(circ)
    #     ax.text(0.05, 0.95, f"R_SAFE = {r_safe:.2f}", transform=ax.transAxes,
    #             ha="left", va="top", fontsize=9, color="gray")

    ax.set_xlabel("x[0]")
    ax.set_ylabel("x[1]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    ax.set_title("2D trajectories (first two state components)")

    plt.tight_layout()
    plt.show()

plot_2d_trajectories(
    theta_inst=inst,
    X_list=[X1, X2, X3],
    labels=["candidate traj 1", "candidate traj 2", "candidate traj 3"],
)


def plot_2d_trajectories_with_scores(theta_inst, X_list, scores=None, labels=None):
    """
    Plot multiple trajectories in 2D (first two state components),
    together with initial and target points and an optional scalar score
    for each trajectory.

    Parameters
    ----------
    theta_inst : dict
        Instance returned by ThetaSampler.sample_instance().
        Used to get x0 and x_goal.
    X_list : list of np.ndarray
        Each element is an array of shape (T+1, n) for one trajectory.
    scores : list of float or None
        One scalar score per trajectory (e.g. normalized score).
        If None, no scores are shown.
    labels : list of str or None
        Optional labels for each trajectory.
    """
    init_goal = theta_inst["init_goal"]
    x0     = np.asarray(init_goal["x0"], dtype=float)
    x_goal = np.asarray(init_goal["x_end"], dtype=float)

    num_traj = len(X_list)

    if labels is None:
        labels = [f"traj {i}" for i in range(num_traj)]

    if scores is not None:
        assert len(scores) == num_traj, "scores must have same length as X_list"

    fig, ax = plt.subplots(figsize=(6, 6))

    for i, (X, label) in enumerate(zip(X_list, labels)):
        X = np.asarray(X, dtype=float)
        assert X.shape[1] >= 2, "Need at least 2 state dimensions to make 2D plot."

        # Plot trajectory line (matplotlib assigns a color automatically)
        (line,) = ax.plot(X[:, 0], X[:, 1], label=label)

        # End point marker
        end_color = line.get_color()
        ax.scatter(X[-1, 0], X[-1, 1], c=[end_color], s=60, marker="X", zorder=5)

        # Annotate scalar score near end point, if provided
        if scores is not None:
            s_val = scores[i]
            ax.text(
                X[-1, 0],
                X[-1, 1],
                f"  score={s_val:.3f}",
                fontsize=8,
                va="center",
                ha="left",
                color=end_color,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
            )

    # Initial point (assumed same for all)
    ax.scatter(x0[0], x0[1], c="green", s=80, marker="o", zorder=6, label="start")

    # Target point
    ax.scatter(x_goal[0], x_goal[1], c="red", s=100, marker="*", zorder=7, label="target")

    ax.set_xlabel("x[0]")
    ax.set_ylabel("x[1]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    ax.set_title("2D trajectories with scalar scores")

    plt.tight_layout()
    plt.show()


plot_2d_trajectories_with_scores(
    theta_inst=inst,
    X_list=[X1, X2, X3, X_cl],
    scores=[score1["Y"], score2["Y"], score3["Y"], score_cl["Y"]],
    labels=["candidate traj 1", "candidate traj 2", "candidate traj 3", "candidate traj llm"],
)


print(score1["Y"])
print(score2["Y"])
print(score3["Y"])
print(score_cl["Y"])



