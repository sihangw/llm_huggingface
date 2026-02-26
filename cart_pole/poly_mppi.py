import numpy as np
from poly_theta_generator import PolyThetaConfig, PolyFromTheta
from poly_theta_sampler import ThetaSampler, ThetaSamplerConfig
from mppi_mpc import MPPIConfig, run_closed_loop_mppi

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

cfg = PolyThetaConfig(n=3, m=1, d_f=2, d_G=2, dt=0.1, N=25)
sampler = ThetaSampler(cfg, ThetaSamplerConfig(), seed=342)
inst  = sampler.sample_instance()  # directly solvable instance

step_fn = inst["step"]
cost_fn = make_cost_fn(inst)

x0 = inst["init_goal"]["x0"]
m = inst["meta"]["m"]
u_max = inst["meta"]["u_max"]

cfg_mppi = MPPIConfig(
    H=5,
    num_samples=256,
    num_iters=3,
    lambda_=1.0,
    noise_sigma=0.3,
    u_min=-u_max,
    u_max= u_max,
    shift_fill="zero",
)

X_cl, U_cl = run_closed_loop_mppi(
    step_fn=step_fn,
    cost_fn=cost_fn,
    x0=x0,
    T_outer=10,
    cfg=cfg_mppi,
    m=m,
)
print("Final:", X_cl[-1], "Goal:", inst["init_goal"]["x_end"])


# Example phi_target: ball of radius rho around x_goal, using Euclidean distance.
x_goal = np.asarray(inst["init_goal"]["x_end"], dtype=float)
rho    = float(inst["init_goal"]["goal_radius"])

def phi_target(x):
    # negative inside target set
    return np.linalg.norm(x - x_goal) - rho

def constraint_excess(x, u, t):
    # Example: no extra constraints -> always satisfy
    return -1.0  # <= 0 => no violation


u_max = float(inst["meta"]["u_max"])
cfg_score = CompositeScoreConfig(
    dt=inst["meta"]["dt"],
    D=inst["meta"]["rho"],
    beta_margin=1,
    beta_time=1,
    beta_u=1,
    beta_du=1,
    U_ref=u_max,
    dU_ref=0.25*u_max,
)

scorer = CompositeTrajectoryScorer(phi_target, cfg_score)
result = scorer.score(X_cl, U_cl)

print("Final Score:", result["score"])
print("Success:", result["success"])
print("effort_score:", result["effort_score"])
print("smooth_score:", result["smooth_score"])
print("margin_score:", result["margin_score"])
print("time_score:", result["time_score"])

from weights_search_mppi import run_weight_search_for_instance

best = run_weight_search_for_instance(
    inst=inst,
    scorer=scorer,
    run_closed_loop_mppi=run_closed_loop_mppi,
    mppi_cfg_base=cfg_mppi,   # your MPPIConfig
    K=20,
    T_outer=30,
    rng=np.random.default_rng(3333),
    keep_top=5,
    early_stop_Y=0.99,
    verbose=True,
)

print("Best Y:", best["Y"])
print("Best w:", best["w"])
