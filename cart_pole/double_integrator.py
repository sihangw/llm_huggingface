#!/usr/bin/env python3
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

# =======================
#  Double integrator (discrete)
#  x = [p, v],   p_{k+1} = p_k + dt * v_k
#                v_{k+1} = v_k + dt * u_k
# =======================
def double_integrator_discretized(dt: float):
    A = np.array([[1.0, dt],
                  [0.0, 1.0]], dtype=float)
    B = np.array([[0.5*dt**2],
                  [dt]], dtype=float)
    c = np.zeros(2, dtype=float)
    return A, B, c

# =======================
#  Rollout + cost
#  - Quadratic tracking on (p, v) with weights (q_p, q_v)
#  - Quadratic actuation r * u^2
#  - Optional input-rate smoothing lam_du * (u_k - u_{k-1})^2
#  - Soft state-limit penalties on p, v
# =======================
# def rollout_cost(x0, U, A, B, c,
#                  p_ref=0.0, v_ref=0.0,
#                  q_p=1.0, q_v=0.1, r=0.01, q_pN=10.0, q_vN=1.0,
#                  u_min=-2.0, u_max=+2.0,
#                  p_lim=5.0, v_lim=3.0,
#                  lam_du=0.0,
#                  penalty_w=1e3):
#     x = np.array(x0, dtype=float)
#     J = 0.0
#     u_prev = None
#     N = U.shape[0]
#
#     def pen_state(x):
#         p, v = x
#         pen = 0.0
#         if abs(p) > p_lim: pen += (abs(p) - p_lim)**2
#         if abs(v) > v_lim: pen += (abs(v) - v_lim)**2
#         return penalty_w * pen
#
#     for k in range(N):
#         u = float(np.clip(U[k, 0], u_min, u_max))
#         p, v = x
#         J += q_p*(p - p_ref)**2 + q_v*(v - v_ref)**2 + r*(u**2) + pen_state(x)
#         if u_prev is not None and lam_du > 0.0:
#             J += lam_du * (u - u_prev)**2
#
#         # dynamics
#         x = A @ x + B.flatten()*u + c
#         u_prev = u
#
#     # terminal cost
#     p, v = x
#     J += q_pN*(p - p_ref)**2 + q_vN*(v - v_ref)**2 + pen_state(x)
#     return float(J)

# def cost_fn(x, u, u_prev=None, terminal=False):
#     """Self-contained double-integrator cost.
#     Args: x=[p,v], u (float), k (int, unused), u_prev (float|None), terminal (bool).
#     Returns (cost, u_clipped).
#     """
#     x = np.asarray(x, float); p, v = x
#     # refs/limits (local to this function)
#     p_ref_ = 0.0; v_ref_ = 0.0; p_lim_ = 5.0; v_lim_ = 3.0; u_min_, u_max_ = -2.0, 2.0
#     # weights
#     q_p_, q_v_, r_ = 1.0, 0.1, 0.01
#     q_pN_, q_vN_, lam_du_, w_ = 10.0, 1.0, 0.1, 1e3
#     u = float(np.clip(u, u_min_, u_max_))
#     # soft state penalty
#     pen = w_*(max(0.0, abs(p)-p_lim_)**2 + max(0.0, abs(v)-v_lim_)**2)
#     if terminal:
#         return q_pN_*(p-p_ref_)**2 + q_vN_*(v-v_ref_)**2 + pen, u
#     J = q_p_*(p-p_ref_)**2 + q_v_*(v-v_ref_)**2 + r_*(u**2) + pen
#     if u_prev is not None and lam_du_ > 0.0:
#         J += lam_du_*(u - float(u_prev))**2
#     return J, u


def cost_fn(x, u, u_prev=None, terminal=False):
    """
    Self-contained double-integrator cost.

    Args:
    x (list): [position, velocity] at time t
    u (float): control input
    u_prev (float, optional): previous control input. Defaults to None.
    terminal (bool, optional): whether the end of the simulation is reached. Defaults to False.

    Returns:
    (float, float): cost and clipped control input
    """
    # Calculate the derivative of the cost function
    dxdt = x[1]
    dvdt = x[0]

    # Calculate the cost function
    cost = 0.5 * np.sum(dxdt ** 2 + dvdt ** 2)

    # Clip the control input to the range [-2, 2]
    u_clipped = np.clip(u, -2, 2)

    # If the end of the simulation is reached, return the cost and clipped control input
    if terminal:
        return cost, u_clipped
    else:
        return cost, u_clipped

def rollout_cost(x0, U, A, B, c):
    x = np.array(x0, float); J = 0.0; u_prev = None
    N = U.shape[0]
    for k in range(N):
        Jk, u_clip = cost_fn(x, U[k,0], u_prev=u_prev, terminal=False)
        J += Jk
        x = A @ x + B.flatten()*u_clip + c
        u_prev = u_clip
    JN, _ = cost_fn(x, 0.0, u_prev=u_prev, terminal=True)
    return float(J + JN)

# =======================
#  CEM sampling MPC (1D input)
#  - piecewise-constant control blocks for smoothness & fewer vars
# =======================
def cem_mpc_step(
    x0, A, B, c,
    N=30,
    u_min=-2.0, u_max=+2.0,
    K=256, Ne=32, iters=3,
    block=2,
    init_mean=0.0, init_std=1.0,
    alpha=1.0,  # elite smoothing (1.0 = overwrite with elites)
    seed=None,
    # # cost parameters passed through:
    # p_ref=0.0, v_ref=0.0,
    # q_p=1.0, q_v=0.1, r=0.01, q_pN=10.0, q_vN=1.0,
    # p_lim=5.0, v_lim=3.0, lam_du=0.0, penalty_w=1e3,
    warm_mean_blocks: np.ndarray | None = None
):
    rng = np.random.default_rng(seed)

    # block parameterization
    Nb = int(np.ceil(N / block))
    m = 1

    if warm_mean_blocks is None:
        mean_blocks = np.full((Nb, m), init_mean, dtype=float)
    else:
        mean_blocks = np.array(warm_mean_blocks, dtype=float)
        if mean_blocks.shape != (Nb, m):
            # reshape / pad if horizon changed
            mean_blocks = np.full((Nb, m), init_mean, dtype=float)

    std_blocks  = np.full((Nb, m), init_std, dtype=float)

    def expand_blocks(Ub):
        # repeat each block and trim to N
        return np.repeat(Ub, repeats=block, axis=0)[:N]

    bestJ = np.inf
    Ubest = np.zeros((N, m), dtype=float)

    for _ in range(iters):
        # sample
        S = mean_blocks[None, :, :] + std_blocks[None, :, :] * rng.standard_normal((K, Nb, m))
        S = np.clip(S, u_min, u_max)  # respect input limits during sampling

        # evaluate
        costs = np.empty(K, dtype=float)
        for k in range(K):
            U = expand_blocks(S[k])
            # J = rollout_cost(
            #     x0, U, A, B, c,
            #     p_ref=p_ref, v_ref=v_ref,
            #     q_p=q_p, q_v=q_v, r=r, q_pN=q_pN, q_vN=q_vN,
            #     u_min=u_min, u_max=u_max,
            #     p_lim=p_lim, v_lim=v_lim,
            #     lam_du=lam_du, penalty_w=penalty_w
            # )
            J = rollout_cost(
                x0, U, A, B, c)
            costs[k] = J
            if J < bestJ:
                bestJ = J; Ubest = U

        # elite update
        elite_idx = np.argpartition(costs, Ne)[:Ne]
        elites = S[elite_idx]  # (Ne, Nb, 1)
        e_mean = elites.mean(axis=0)
        e_std  = elites.std(axis=0) + 1e-6

        mean_blocks = alpha*e_mean + (1 - alpha)*mean_blocks
        std_blocks  = alpha*e_std  + (1 - alpha)*std_blocks
        std_blocks  = np.maximum(std_blocks, 0.05)  # avoid collapse

    # control to apply
    u0 = float(np.clip(Ubest[0, 0], u_min, u_max))
    # warm-start for next step: shift left, keep last block
    warm_next = np.vstack([mean_blocks[1:], mean_blocks[-1:]])
    return u0, Ubest, warm_next, bestJ

# =======================
#  Demo
#  - drive to (p_ref, v_ref) = (0, 0) from an initial offset
# =======================
def simulate_double_integrator():
    dt = 0.05
    A, B, c = double_integrator_discretized(dt)

    # horizons & limits
    N = 30
    u_min, u_max = -2.0, 2.0
    p_lim, v_lim = 5.0, 3.0

    # cost weights
    q_p, q_v, r = 1.0, 0.1, 0.01
    q_pN, q_vN = 10.0, 1.0
    lam_du = 0.1  # small input-rate smoothing

    # initial state: start away from target
    x = np.array([+4.0, -1.0], dtype=float)  # (p, v)
    p_ref, v_ref = 0.0, 0.0

    T = 120  # simulate for T steps
    xs, us = [x.copy()], []
    warm_mean_blocks = None

    for t in range(T):
        u0, Uplan, warm_mean_blocks, J = cem_mpc_step(
            x0=x, A=A, B=B, c=c, N=N,
            u_min=u_min, u_max=u_max,
            K=256, Ne=32, iters=3, block=2,
            init_mean=0.0, init_std=1.0, alpha=1.0, seed=None,
            # p_ref=p_ref, v_ref=v_ref,
            # q_p=q_p, q_v=q_v, r=r, q_pN=q_pN, q_vN=q_vN,
            # p_lim=p_lim, v_lim=v_lim, lam_du=lam_du, penalty_w=1e3,
            warm_mean_blocks=warm_mean_blocks
        )

        # apply first control and evolve true dynamics
        u = float(np.clip(u0, u_min, u_max))
        x = A @ x + B.flatten()*u + c

        xs.append(x.copy()); us.append(u)

    xs = np.array(xs)  # shape (T+1, 2)
    us = np.array(us)  # shape (T,)

    t_grid = np.arange(T+1) * dt
    fig, ax = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    ax[0].plot(t_grid, xs[:, 0]); ax[0].axhline(+p_lim, ls="--", alpha=0.5); ax[0].axhline(-p_lim, ls="--", alpha=0.5)
    ax[0].set_ylabel("p (pos)")
    ax[0].set_title("Double Integrator — Sampling MPC (CEM)")

    ax[1].plot(t_grid, xs[:, 1]); ax[1].axhline(+v_lim, ls="--", alpha=0.5); ax[1].axhline(-v_lim, ls="--", alpha=0.5)
    ax[1].set_ylabel("v (vel)")

    ax[2].plot(t_grid[:-1], us)
    ax[2].axhline(+u_min, ls="--", alpha=0.5); ax[2].axhline(+u_max, ls="--", alpha=0.5)
    ax[2].set_ylabel("u"); ax[2].set_xlabel("time (s)")
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    simulate_double_integrator()
