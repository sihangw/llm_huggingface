import numpy as np
from poly_theta_generator import PolyThetaConfig, PolyFromTheta
from poly_theta_sampler import ThetaSampler, ThetaSamplerConfig
from mppi_mpc import MPPIConfig, run_closed_loop_mppi

from CompositeScore import CompositeScoreConfig, CompositeTrajectoryScorer
from itertools import combinations_with_replacement
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # must be set before torch initializes CUDA
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
torch.cuda.set_device(0)
import argparse, json, re, time
from datasets import load_dataset
from typing import List, Dict, Any, Tuple, Optional
# from vllm import LLM, SamplingParams
import dill
from ilqr_mpc import PolyDynamicsWithJac, QuadCostConfig
from instance_sampler import ExperimentSampler, ExperimentSamplerConfig
from weights_search_ilqr import run_weight_search_for_instance_fast_ilqr, make_fast_cost_from_w
from CompositeScore import CompositeScoreConfig, CompositeTrajectoryScorer
from instance_prompt import build_instance_prompt, build_beta_prompt_simple
from ilqr_mpc import ILQRMPCConfig, FastILQRMPC, run_closed_loop_fast_ilqr_mpc, FastQuadraticCost


import numpy as np
import cvxpy as cp

# ---------------------------
# Nonlinear system: unicycle
# x = [px, py, theta], u = [v, w]
# ---------------------------
def f_unicycle(x, u, dt):
    px, py, th = x
    v, w = u
    return np.array([
        px + dt * v * np.cos(th),
        py + dt * v * np.sin(th),
        th + dt * w
    ], dtype=float)

def rollout_unicycle(x0, U, dt):
    H = U.shape[0]
    X = np.zeros((H + 1, 3), dtype=float)
    X[0] = x0
    for t in range(H):
        X[t + 1] = f_unicycle(X[t], U[t], dt)
    return X

# ---------------------------
# Linearize dynamics along a nominal (Xn, Un):
# x_{k+1} ≈ A_k x_k + B_k u_k + c_k
# ---------------------------
def linearize_unicycle(Xn, Un, dt):
    N = Un.shape[0]
    A = np.zeros((N, 3, 3), dtype=float)
    B = np.zeros((N, 3, 2), dtype=float)
    c = np.zeros((N, 3), dtype=float)

    for k in range(N):
        px, py, th = Xn[k]
        v, w = Un[k]

        Ak = np.eye(3)
        Ak[0, 2] = -dt * v * np.sin(th)
        Ak[1, 2] =  dt * v * np.cos(th)

        Bk = np.zeros((3, 2), dtype=float)
        Bk[0, 0] = dt * np.cos(th)  # dpx/dv
        Bk[1, 0] = dt * np.sin(th)  # dpy/dv
        Bk[2, 1] = dt               # dtheta/dw

        fk = f_unicycle(Xn[k], Un[k], dt)
        ck = fk - Ak @ Xn[k] - Bk @ Un[k]

        A[k], B[k], c[k] = Ak, Bk, ck

    return A, B, c

# ---------------------------
# One SQP-like MPC step (successive linearization)
# ---------------------------
def mpc_solve_step(x_init, x_goal, dt, N,
                   v_max=1.0, w_max=1.5,
                   Q=np.diag([30.0, 30.0, 0.5]),
                   R=np.diag([0.2, 0.05]),
                   QN=np.diag([80.0, 80.0, 1.0]),
                   sqp_iters=3,
                   u_init=None):
    """
    Returns a control sequence U*(N,2) and predicted X*(N+1,3)
    """

    # Nominal initialization
    if u_init is None:
        Un = np.zeros((N, 2), dtype=float)
        Un[:, 0] = 0.7  # nominal forward speed
        Un[:, 1] = 0.0
    else:
        Un = u_init.copy()

    # Build nominal state rollout
    def rollout_nom(x0, U):
        X = np.zeros((N + 1, 3), dtype=float)
        X[0] = x0
        for k in range(N):
            X[k + 1] = f_unicycle(X[k], U[k], dt)
        return X

    Xn = rollout_nom(x_init, Un)

    for _ in range(sqp_iters):
        A, B, c = linearize_unicycle(Xn, Un, dt)

        # Decision variables
        X = cp.Variable((N + 1, 3))
        U = cp.Variable((N, 2))

        constraints = [X[0, :] == x_init]

        for k in range(N):
            constraints += [
                X[k + 1, :] == A[k] @ X[k, :] + B[k] @ U[k, :] + c[k]
            ]
            constraints += [
                cp.abs(U[k, 0]) <= v_max,
                cp.abs(U[k, 1]) <= w_max
            ]

        # Objective
        obj = 0
        for k in range(N):
            dx = X[k, :] - x_goal
            obj += cp.quad_form(dx, Q) + cp.quad_form(U[k, :], R)
        obj += cp.quad_form(X[N, :] - x_goal, QN)

        prob = cp.Problem(cp.Minimize(obj), constraints)

        # Use OSQP by default (works well for QP). If you have MOSEK, swap solver="MOSEK".
        prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)

        if U.value is None or X.value is None:
            # fallback: return nominal if solver fails
            return Un, Xn

        Un = U.value
        Xn = rollout_nom(x_init, Un)

    return Un, Xn

# ---------------------------
# Receding-horizon MPC rollout (this is your "best" trajectory)
# ---------------------------
def run_mpc(x0, x_goal, dt=0.1, H=30, N=20, v_max=1.0, w_max=1.5):
    X = np.zeros((H + 1, 3), dtype=float)
    U = np.zeros((H, 2), dtype=float)
    X[0] = x0

    u_warm = None
    for t in range(H):
        U_star, _ = mpc_solve_step(
            x_init=X[t],
            x_goal=x_goal,
            dt=dt,
            N=N,
            v_max=v_max,
            w_max=w_max,
            sqp_iters=3,
            u_init=u_warm
        )
        u = U_star[0]
        U[t] = u
        X[t + 1] = f_unicycle(X[t], u, dt)

        # warm start: shift controls
        u_warm = np.vstack([U_star[1:], U_star[-1:]])

    return X, U

def run_mpc_with_weights(x0, x_goal, dt, H, N, v_max, w_max, Q, QN, R,
                         mpc_solve_step_fn, f_step_fn):
    """
    Receding-horizon MPC rollout using provided weight matrices.
    mpc_solve_step_fn must accept (x_init, x_goal, dt, N, v_max, w_max, Q, R, QN, u_init)
    f_step_fn is the true nonlinear step x_{t+1} = f(x_t, u_t)
    """
    X = np.zeros((H + 1, 3), dtype=float)
    U = np.zeros((H, 2), dtype=float)
    X[0] = x0

    u_warm = None
    for t in range(H):
        U_star, _ = mpc_solve_step_fn(
            x_init=X[t],
            x_goal=x_goal,
            dt=dt,
            N=N,
            v_max=v_max,
            w_max=w_max,
            Q=Q, R=R, QN=QN,
            u_init=u_warm
        )
        u = U_star[0]
        U[t] = u
        X[t + 1] = f_step_fn(X[t], u, dt)
        u_warm = np.vstack([U_star[1:], U_star[-1:]])  # shift warm start

    return X, U

# ---------------------------
# Three random-but-meaningfully-different policies
# ---------------------------
def sample_piecewise(H, m, rng, umin, umax, n_segments=6):
    edges = np.linspace(0, H, n_segments + 1, dtype=int)
    U = np.zeros((H, m), dtype=float)
    for s in range(n_segments):
        u = rng.uniform(umin, umax, size=(m,))
        U[edges[s]:edges[s+1]] = u
    return U

def sample_smooth(H, m, rng, umin, umax, smooth=0.9):
    U = np.zeros((H, m), dtype=float)
    u = np.zeros(m)
    for t in range(H):
        u = smooth * u + (1 - smooth) * rng.normal(size=m)
        U[t] = u
    # scale each dim to bounds
    U = U / (np.max(np.abs(U), axis=0, keepdims=True) + 1e-9)
    mid = 0.5 * (umin + umax)
    rad = 0.5 * (umax - umin)
    return mid + rad * U

def sample_pulses(H, m, rng, umin, umax, n_pulses=4, width=5):
    U = np.zeros((H, m), dtype=float)
    for _ in range(n_pulses):
        t0 = int(rng.integers(0, max(1, H - width)))
        j = int(rng.integers(0, m))
        amp = rng.uniform(umin[j], umax[j])
        U[t0:t0+width, j] = amp
    return np.clip(U, umin, umax)

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    dt = 0.1
    H = 40          # total rollout steps you want to SHOW
    N = 20          # MPC horizon
    x0 = np.array([0.0, 0.0, 0.0])
    x_goal = np.array([2.0, 2.0, 0.0])   # goal includes theta; you can ignore theta in your own score
    target_xy = x_goal[:2]
    rho = 0.1
    u_max = 1.5

    def phi_target(x):
        return np.linalg.norm(x - x_goal) - rho  # negative inside target


    cfg_score = CompositeScoreConfig(
        dt=dt,
        D=rho,
        beta_margin=0.3,
        beta_time=0.5,
        beta_u=0.4,
        beta_du=0.2,
        U_ref=u_max,
        dU_ref=0.25 * u_max,
    )
    scorer = CompositeTrajectoryScorer(phi_target, cfg_score)

    v_max = u_max
    w_max = u_max
    umin = np.array([-v_max, -w_max])
    umax = np.array([ v_max,  w_max])
    #
    # Q_base = np.diag([30.0, 30.0, 0.5])
    # R_base = np.diag([0.2, 0.05])
    #
    # configs = [
    #     {"label": "Q=pos30, P=pos80", "Q": np.diag([30, 30, 0.5]), "QN": np.diag([80, 80, 1.0])},
    #     {"label": "Q=pos10, P=pos200", "Q": np.diag([5, 10, 10]), "QN": np.diag([10, 20, 2.0])},
    #     # {"label": "Q=pos80, P=pos80", "Q": np.diag([1, 80, 8]), "QN": np.diag([20, 80, 10])},
    #     {"label": "Q=pos30, P=pos400", "Q": np.diag([0.5, 0.5, 30]), "QN": np.diag([400, 400, 4.0])},
    # ]
    #
    # X_list, U_list, labels, scores = [], [], [], []
    #
    # for cfg in configs:
    #     X, U = run_mpc_with_weights(
    #         x0=x0, x_goal=x_goal, dt=dt, H=H, N=N,
    #         v_max=v_max, w_max=w_max,
    #         Q=cfg["Q"], QN=cfg["QN"], R=R_base,
    #         mpc_solve_step_fn=mpc_solve_step,  # from your MPC code
    #         f_step_fn=f_unicycle  # from your dynamics code
    #     )
    #
    #     # use YOUR score(X,U). Example:
    #     s = scorer.score(X, U)["score"]
    #
    #     X_list.append(X)
    #     U_list.append(U)
    #     labels.append(cfg["label"])
    #     scores.append(s)
    #
    # print(scores)




    # 1) MPC trajectory (this should be the "best" for reach-to-goal)
    X_mpc, U_mpc = run_mpc(x0, x_goal, dt=dt, H=H, N=N, v_max=v_max, w_max=w_max)

    # 2-4) Three different random policies
    U_rand1 = sample_piecewise(H, 2, rng, umin, umax, n_segments=6)
    U_rand2 = sample_smooth(H, 2, rng, umin, umax, smooth=0.92)
    U_rand3 = sample_pulses(H, 2, rng, umin, umax, n_pulses=5, width=4)

    X_rand1 = rollout_unicycle(x0, U_rand1, dt)
    X_rand2 = rollout_unicycle(x0, U_rand2, dt)
    X_rand3 = rollout_unicycle(x0, U_rand3, dt)

    # Package outputs exactly as "4 sets of X and U"
    X_list = [X_mpc, X_rand1, X_rand2, X_rand3]
    U_list = [U_mpc, U_rand1, U_rand2, U_rand3]
    # names  = ["mpc_best", "rand_piecewise", "rand_smooth", "rand_pulses"]

    S_mpc = scorer.score(X_mpc, U_mpc)
    S_1 = scorer.score(X_rand1, U_rand1)
    S_2 = scorer.score(X_rand2, U_rand2)
    S_3 = scorer.score(X_rand3, U_rand3)

    print(S_mpc, S_1, S_2, S_3)







    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors
    from matplotlib.patches import Circle


    def plot_xy_trajectories_with_scores(
            x0, xg, X_list, scores=None, labels=None,
            annotate="closest",  # "final" or "closest"
            show_start=True, show_goal=True,
            show_end_marker=True
    ):
        """
        Plot multiple trajectories in XY only (X[:,0], X[:,1]).
        Optionally annotate a scalar score at either the final point or the point
        closest to the goal in XY.

        Parameters
        ----------
        x0, xg : array-like (>=2,)
            Start and goal states. Only first 2 entries used.
        X_list : list of arrays, each (T+1, n)
        scores : list of floats or None
        labels : list of str or None
        annotate : str
            "final" -> annotate at last state
            "closest" -> annotate at point with min XY distance to goal
        """

        x0 = np.asarray(x0, dtype=float).ravel()
        xg = np.asarray(xg, dtype=float).ravel()
        x0_xy, xg_xy = x0[:2], xg[:2]

        num_traj = len(X_list)
        if labels is None:
            labels = [f"traj {i}" for i in range(num_traj)]
        if scores is not None:
            assert len(scores) == num_traj, "scores must match X_list length"

        fig, ax = plt.subplots(figsize=(6.5, 6.0))

        # Track bounds for nicer autoscaling
        all_xy = [x0_xy, xg_xy]

        for i, (X, label) in enumerate(zip(X_list, labels)):
            X = np.asarray(X, dtype=float)
            assert X.shape[1] >= 2, "Need at least 2 state dimensions to plot XY."

            xy = X[:, :2]
            all_xy.append(xy)

            (line,) = ax.plot(xy[:, 0], xy[:, 1], label=label, linewidth=2.0)
            color = line.get_color()

            # End marker
            if show_end_marker:
                ax.scatter(xy[-1, 0], xy[-1, 1], c=[color], s=60, marker="X", zorder=5)

            # Choose annotation point
            if scores is not None:
                if annotate == "final":
                    idx = -1
                elif annotate == "closest":
                    d = np.linalg.norm(xy - xg_xy[None, :], axis=1)
                    idx = int(np.argmin(d))
                else:
                    raise ValueError("annotate must be 'final' or 'closest'")

                px, py = xy[idx, 0], xy[idx, 1]
                s_val = float(scores[i])

                ax.text(
                    px, py,
                    f" score={s_val:.3f}",
                    fontsize=9,
                    va="center", ha="left",
                    color=color,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.75, edgecolor="none"),
                    zorder=10
                )

                # Optional marker at annotation point if it's not the end
                if annotate == "closest" and idx != len(xy) - 1:
                    ax.scatter(px, py, c=[color], s=40, marker="o", zorder=6, alpha=0.9)

        # Start & goal
        if show_start:
            ax.scatter(x0_xy[0], x0_xy[1], c="green", s=90, marker="o", zorder=8, label="start")
        if show_goal:
            ax.scatter(xg_xy[0], xg_xy[1], c="red", s=140, marker="*", zorder=9, label="target")

        # Pretty axis limits with margin
        # Concatenate all points
        pts = []
        for item in all_xy:
            arr = np.asarray(item)
            if arr.ndim == 1:
                pts.append(arr[None, :])
            else:
                pts.append(arr)
        pts = np.vstack(pts)
        xmin, ymin = pts.min(axis=0)
        xmax, ymax = pts.max(axis=0)
        dx, dy = xmax - xmin, ymax - ymin
        pad = 0.08 * max(dx, dy, 1e-6)
        ax.set_xlim(xmin - pad, xmax + pad)
        ax.set_ylim(ymin - pad, ymax + pad)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        ax.set_title("Scores by different trajectories")

        plt.tight_layout()
        plt.show()
        # X_mpc, X_rand1, X_rand2, X_rand3


    # plot_xy_trajectories_with_scores(
    #     x0=x0, xg=x_goal,
    #     X_list=X_list,
    #     scores=scores,
    #     labels=["candidate 1", "candidate 2", "candidate 3", "candidate llm"],
    #     annotate="closest"
    # )

    plot_xy_trajectories_with_scores(
        x0=x0,
        xg=x_goal,
        X_list=[X_rand1, X_rand2, X_rand3, X_mpc],
        scores=[S_1["score"], S_2["score"], S_3["score"], S_mpc["score"]],
        labels=["random 1", "random 2", "random 3", "llm"],
        annotate="closest"  # better when “final” isn’t what matters
    )














# def rollout(drift: Dict, inp: Dict, x0: np.ndarray, U_seq: np.ndarray,n, N, step) -> np.ndarray:
#     X = np.zeros((N + 1, n))
#     X[0] = x0;
#     x = x0.copy()
#     for k in range(N):
#         x = step(x, U_seq[k], drift, inp)
#         X[k + 1] = x
#     return X
#
#
# sampler = ExperimentSampler(ExperimentSamplerConfig(
#     dt=0.1, N=30, K=10, max_iters=5,
#     n_range=(2, 2), m_range=(1, 2), d_f_range=(1, 3), d_G_range=(1, 3),
#     H_range=(3, 5),
#     T_outer_slack_frac_range=(0.3, 1.1),  # <N or >N allowed
#     R_safe=(2.0, 100.0),  # sample safe radius relative to r
#     u_max_range=(0.1, 50.0),
#     prob_mode=(0.1, 0.5, 0.1),
#     master_seed=400,
# ))
# # #
# # # trial = sampler.sample_trial()
# # # inst = trial["inst"]
# # # p = trial["params"]
# # # LOAD instances directly for reward
# # def load_instances(path: str) -> dict:
# #     with open(path, "rb") as f:
# #         instances = dill.load(f)
# #     print(f"Loaded {len(instances)} instances")
# #     return instances
# #
# # ds = load_dataset("json", data_files="data_10000_dt05_N20_grpo/dataset_all.jsonl", split="train")
# # ds = ds.train_test_split(0.2, seed=24)
# # train_ds = ds["train"]
# # eval_ds  = ds["test"]
# # instances = load_instances(path="data_10000_dt05_N20_grpo/instances.dill")
# #
# # data_pt = train_data = train_ds[122]
# # idx = data_pt["trial_id"]
# # p = data_pt["params"]
# # inst = instances[idx]
# #
# # def read_jsonl_row(path, idx: int):
# #     with open(path, "r", encoding="utf-8") as f:
# #         for i, line in enumerate(f):
# #             if i == idx:
# #                 return json.loads(line)
# #     raise IndexError(f"Index {idx} out of range")
# #
# # row = read_jsonl_row("data_10000_dt05_N20_grpo/dataset_all.jsonl", idx)   # 11th row
# # w_best = row["best_w"]
# # print(w_best)
#
#
# trial = sampler.sample_trial()
# inst = trial["inst"]
# p = trial["params"]
#
# x_0 = np.asarray(inst["init_goal"]["x0"], dtype=float)
# x_goal = np.asarray(inst["init_goal"]["x_end"], dtype=float)
# rho = float(inst["init_goal"]["goal_radius"])
# u_max = float(inst["meta"]["u_max"])
#
#
# def phi_target(x):
#     return np.linalg.norm(x - x_goal) - rho  # negative inside target
#
#
# cfg_score = CompositeScoreConfig(
#     dt=float(inst["meta"]["dt"]),
#     D=float(inst["meta"]["rho"]),
#     beta_margin=float(p["beta_margin"]),
#     beta_time=float(p["beta_time"]),
#     beta_u=float(p["beta_u"]),
#     beta_du=float(p["beta_du"]),
#     U_ref=u_max,
#     dU_ref=0.25 * u_max,
# )
# scorer = CompositeTrajectoryScorer(phi_target, cfg_score)
#
# # ----- MPC config for evaluation -----
# cfg_ilqr = ILQRMPCConfig(
#     H=int(p["H"]),
#     max_iters=int(p["max_iters"]),
#     u_min=-u_max,
#     u_max=u_max,
#     shift_fill="zero",
# )
#
# def mpc_builder(inst_local, cost_obj, cfg_ilqr_local):
#     dyn = PolyDynamicsWithJac(cfg=inst_local["cfg"], drift=inst_local["drift"], inp=inst_local["input"])
#     return FastILQRMPC(dyn=dyn, cost=cost_obj, cfg=cfg_ilqr_local)
#
#
# best = run_weight_search_for_instance_fast_ilqr(
#     inst=inst,
#     scorer=scorer,
#     run_closed_loop_fast_ilqr_mpc=run_closed_loop_fast_ilqr_mpc,
#     mpc_builder=mpc_builder,
#     ilqr_cfg_base=cfg_ilqr,
#     QuadCostConfig=QuadCostConfig,
#     FastQuadraticCost=FastQuadraticCost,
#     K=int(p["K"]),
#     T_outer=int(p["T_outer"]),
#     rng=trial["rng_search"],
#     keep_top=int(p.get("keep_top", 5)),
#     early_stop_Y=float(p.get("early_stop_Y", 0.999)),
#     verbose=True,
# )
#
# Y_oracle = float(best["Y"])
# w_oracle = np.asarray(best["w"], dtype=float).reshape(-1)
#
# cost_obj = make_fast_cost_from_w(inst, w_oracle, QuadCostConfig, FastQuadraticCost)
# mpc_llm = mpc_builder(inst, cost_obj, cfg_ilqr)
# X_llm, U_llm = run_closed_loop_fast_ilqr_mpc(
#     mpc=mpc_llm,
#     step_fn_true=inst["step"],
#     x0=x_0,
#     T_outer=int(p["T_outer"]),
# )
# score_dict = scorer.score(X_llm, U_llm)
# S_LLM = float(score_dict["score"])
# succ_llm = bool(score_dict["success"])
# print(S_LLM)
#
# w_rand = np.array([0.5, 100, 20, 0, 1, 40, 20])   # 7-dim, uniform in [0, 1)
# cost_obj = make_fast_cost_from_w(inst, w_rand, QuadCostConfig, FastQuadraticCost)
# mpc_llm = mpc_builder(inst, cost_obj, cfg_ilqr)
# X_llm, U_llm = run_closed_loop_fast_ilqr_mpc(
#     mpc=mpc_llm,
#     step_fn_true=inst["step"],
#     x0=x_0,
#     T_outer=int(p["T_outer"]),
# )
# score_dict = scorer.score(X_llm, U_llm)
# S_LLM = float(score_dict["score"])
# succ_llm = bool(score_dict["success"])
# print(S_LLM)


# meta = inst["meta"]
# H = meta["N"]
# m = meta["m"]
# u_max = meta["u_max"]
# rng = np.random.default_rng(0)
# # # Trajectory 1: "reasonable" small controls drifting toward zero
# # U1 = rng.uniform(-0.01 * u_max, 0.01 * u_max, size=(H, m))
# # # Trajectory 2: "aggressive" larger, noisy controls
# # U2 = rng.uniform(-u_max, u_max, size=(H, m))
# # # Trajectory 3:
# # U3 = rng.uniform(-0.5 * u_max, 0.5 * u_max, size=(H, m))
# # print(U1)
# # print(U2)
# # print(U3)
#
# def sample_piecewise(H, m, u_max, rng, n_segments=5):
#     seg_edges = np.linspace(0, H, n_segments+1, dtype=int)
#     U = np.zeros((H, m))
#     for s in range(n_segments):
#         u = rng.uniform(-u_max, u_max, size=(m,))
#         U[seg_edges[s]:seg_edges[s+1]] = u
#     return U
#
# def sample_pulses(H, m, u_max, rng, n_pulses=3, width=5):
#     U = np.zeros((H, m))
#     for _ in range(n_pulses):
#         t0 = rng.integers(0, max(1, H-width))
#         dim = rng.integers(0, m)
#         amp = rng.choice([-1.0, 1.0]) * rng.uniform(0.5, 1.0) * u_max[dim] if np.ndim(u_max)>0 else rng.uniform(0.5,1.0)*u_max
#         U[t0:t0+width, dim] = amp
#     return np.clip(U, -u_max, u_max)
#
# def sample_sin(H, m, u_max, rng, dt=1.0):
#     t = np.arange(H) * dt
#     U = np.zeros((H, m))
#     for j in range(m):
#         A = rng.uniform(0.2, 1.0) * (u_max[j] if np.ndim(u_max)>0 else u_max)
#         f = rng.uniform(0.05, 0.5)  # cycles per step unit-ish
#         phi = rng.uniform(0, 2*np.pi)
#         U[:, j] = A * np.sin(2*np.pi*f*t + phi)
#     return np.clip(U, -u_max, u_max)
#
# U1 = sample_piecewise(H, m, u_max, rng)
# U2 = sample_pulses(H, m, u_max, rng)
# U3 = sample_sin(H, m, u_max, rng)
#
#
# drift = inst["drift"]
# inp = inst["input"]
# step = inst["step"]
# n = p["n"]
# N = p["N"]
# x0 = inst["init_goal"]["x0"]
# print(x0)
#
# X1 = rollout(drift, inp, x0, U1, n, N, step)
# S_1 = scorer.score(X1, U1)
# X2 = rollout(drift, inp, x0, U2, n, N, step)
# S_2 = scorer.score(X2, U2)
# X3 = rollout(drift, inp, x0, U3, n, N, step)
# S_3 = scorer.score(X3, U3)




# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import cm, colors
# from matplotlib.patches import Circle
#
#
# def plot_2d_trajectories_with_scores(theta_inst, X_list, scores=None, labels=None):
#     """
#     Plot multiple trajectories in 2D (first two state components),
#     together with initial and target points and an optional scalar score
#     for each trajectory.
#
#     Parameters
#     ----------
#     theta_inst : dict
#         Instance returned by ThetaSampler.sample_instance().
#         Used to get x0 and x_goal.
#     X_list : list of np.ndarray
#         Each element is an array of shape (T+1, n) for one trajectory.
#     scores : list of float or None
#         One scalar score per trajectory (e.g. normalized score).
#         If None, no scores are shown.
#     labels : list of str or None
#         Optional labels for each trajectory.
#     """
#     init_goal = theta_inst["init_goal"]
#     x0     = np.asarray(init_goal["x0"], dtype=float)
#     x_goal = np.asarray(init_goal["x_end"], dtype=float)
#
#     num_traj = len(X_list)
#
#     if labels is None:
#         labels = [f"traj {i}" for i in range(num_traj)]
#
#     if scores is not None:
#         assert len(scores) == num_traj, "scores must have same length as X_list"
#
#     fig, ax = plt.subplots(figsize=(6, 6))
#
#     for i, (X, label) in enumerate(zip(X_list, labels)):
#         X = np.asarray(X, dtype=float)
#         assert X.shape[1] >= 2, "Need at least 2 state dimensions to make 2D plot."
#
#         # Plot trajectory line (matplotlib assigns a color automatically)
#         (line,) = ax.plot(X[:, 0], X[:, 1], label=label)
#
#         # End point marker
#         end_color = line.get_color()
#         ax.scatter(X[-1, 0], X[-1, 1], c=[end_color], s=60, marker="X", zorder=5)
#
#         # Annotate scalar score near end point, if provided
#         if scores is not None:
#             s_val = scores[i]
#             ax.text(
#                 X[-1, 0],
#                 X[-1, 1],
#                 f"  score={s_val:.3f}",
#                 fontsize=8,
#                 va="center",
#                 ha="left",
#                 color=end_color,
#                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
#             )
#
#     # Initial point (assumed same for all)
#     ax.scatter(x0[0], x0[1], c="green", s=80, marker="o", zorder=6, label="start")
#
#     # Target point
#     ax.scatter(x_goal[0], x_goal[1], c="red", s=100, marker="*", zorder=7, label="target")
#
#     ax.set_xlabel("x[0]")
#     ax.set_ylabel("x[1]")
#     ax.set_aspect("equal", adjustable="box")
#     ax.grid(True, alpha=0.3)
#     ax.legend(loc="best")
#     ax.set_title("2D trajectories with scalar scores")
#
#     plt.tight_layout()
#     plt.show()
#
#
# plot_2d_trajectories_with_scores(
#     theta_inst=inst,
#     X_list=[X1, X2, X3, X_llm],
#     scores=[S_1["score"], S_2["score"], S_3["score"], S_LLM],
#     labels=["candidate traj 1", "candidate traj 2", "candidate traj 3", "candidate traj llm"],
# )




# print(score1["score"])
# print(score2["score"])
# print(score3["score"])
# print(score_cl["score"])