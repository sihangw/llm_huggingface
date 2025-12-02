from __future__ import annotations
import numpy as np

def linearize_discretize_upright(dt: float, params: dict):
    m = float(params["m"]); M = float(params["M"])
    l = float(params["l"]); g = float(params["g"])

    Ac = np.array([
        [0.0, 0.0,                    1.0, 0.0],
        [0.0, 0.0,                    0.0, 1.0],
        [((m+M)*g)/(l*M), 0.0,        0.0, 0.0],
        [-(m*g)/M,        0.0,        0.0, 0.0],
    ], dtype=float)

    Bc = np.array([[0.0],
                   [0.0],
                   [-1.0/(l*M)],
                   [ 1.0/ M     ]], dtype=float)

    Ad = np.eye(4) + dt * Ac
    Bd = dt * Bc
    cd = np.zeros(4)
    return Ad, Bd, cd

def _rollout_cost(
    x0: np.ndarray,
    U: np.ndarray,                 # (N, m=1)
    A: np.ndarray, B: np.ndarray, c: np.ndarray,
    cost: dict, cons: dict
):
    """Quadratic stage + terminal cost with soft state box penalties."""
    Q = np.asarray(cost["Q"]); R = np.asarray(cost["R"]); P = np.asarray(cost["P"])
    x_ref = np.asarray(cost.get("x_ref", np.zeros(4)))
    u_ref = np.asarray(cost.get("u_ref", np.zeros(1)))

    x_min = np.asarray(cons["x_min"]); x_max = np.asarray(cons["x_max"])
    u_min = float(cons["u_min"]);      u_max = float(cons["u_max"])

    N = U.shape[0]; n = 4
    x = np.array(x0, dtype=float)
    J = 0.0

    # penalties for state constraint violation (hinge)
    def box_violation_penalty(xk):
        # weight can be tuned; larger => closer to hard constraints
        w = 1e4
        vio_lo = np.maximum(0.0, (x_min - xk))
        vio_hi = np.maximum(0.0, (xk    - x_max))
        return w * (vio_lo @ vio_lo + vio_hi @ vio_hi)

    for k in range(N):
        u = float(np.clip(U[k, 0], u_min, u_max))
        dx = x - x_ref
        du = np.array([u]) - u_ref
        J += float(dx @ Q @ dx + du @ R @ du) + box_violation_penalty(x)
        # dynamics
        x = A @ x + B.flatten() * u + c

    # terminal
    dxN = x - x_ref
    J += float(dxN @ P @ dxN) + box_violation_penalty(x)
    return J

def solve_mpc_sampling(
    x0: np.ndarray,
    N: int,
    dt: float,
    params: dict,
    cost: dict,
    cons: dict,
    warm: dict | None = None,
    *,
    K: int = 512,            # number of samples
    Ne: int = 64,            # elites
    iters: int = 4,          # CEM iterations
    init_std: float = 3.0,   # initial std for controls
    block: int = 1,          # piecewise constant block size for u
    seed: int | None = None,
    alpha: float = 1.0        # smoothing in CEM (1.0 = overwrite)
):
    """
    Returns:
      u0 (float), sol = {"U": U_best, "mean": mean_blocks, "std": std_blocks, "status": "optimal"}
    """
    rng = np.random.default_rng(seed)
    A, B, c = linearize_discretize_upright(dt, params)

    # Input box
    u_min = float(cons["u_min"]); u_max = float(cons["u_max"])

    # block-constant parameterization (reduces search dimension)
    Nb = int(np.ceil(N / block))
    m = 1
    if warm is not None and "mean_blocks" in warm:
        mean_blocks = np.array(warm["mean_blocks"], dtype=float)  # (Nb, m)
    elif warm is not None and "U" in warm:
        # derive blocks from previous U plan
        Ub = np.mean(warm["U"].reshape(-1, block, m)[:Nb], axis=1)
        mean_blocks = Ub
    else:
        mean_blocks = np.zeros((Nb, m))
    std_blocks  = np.full((Nb, m), init_std)

    def expand_blocks(Ub):
        # Repeat each block for 'block' steps, then trim to horizon N
        return np.repeat(Ub, repeats=block, axis=0)[:N]

    best_J = np.inf
    U_best = np.zeros((N, m))

    for _ in range(iters):
        # sample around current mean/std
        samples = mean_blocks[None, :, :] + std_blocks[None, :, :] * rng.standard_normal((K, Nb, m))
        samples = np.clip(samples, u_min, u_max)  # safe sampling
        costs = np.empty(K)

        # evaluate
        for k in range(K):
            Ub = samples[k]           # (Nb,1)
            U  = expand_blocks(Ub)    # (N,1)
            costs[k] = _rollout_cost(x0, U, A, B, c, cost, cons)
            if costs[k] < best_J:
                best_J = costs[k]; U_best = U

        # elites & refit
        elite_idx = np.argpartition(costs, Ne)[:Ne]
        elites = samples[elite_idx]              # (Ne, Nb, 1)
        e_mean = elites.mean(axis=0)
        e_std  = elites.std(axis=0) + 1e-6

        # smooth updating (alpha=1 => pure elite stats)
        mean_blocks = alpha * e_mean + (1 - alpha) * mean_blocks
        std_blocks  = alpha * e_std  + (1 - alpha) * std_blocks
        # Prevent collapse too early
        std_blocks = np.maximum(std_blocks, 0.05)

    # control to apply now
    u0 = float(np.clip(U_best[0, 0], u_min, u_max))

    # warm-start for next step: shift blocks left, keep last
    shifted = np.vstack([mean_blocks[1:], mean_blocks[-1:]])
    warm_next = {"mean_blocks": shifted, "U": U_best}

    sol = {"status": "optimal", "obj": float(best_J), "U": U_best,
           "mean": mean_blocks, "std": std_blocks, "warm_next": warm_next}
    return u0, sol


