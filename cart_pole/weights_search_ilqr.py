# weight_search_fast_ilqr.py
# from __future__ import annotations

import numpy as np
from dataclasses import replace
from typing import Any, Callable, Dict, List, Optional, Tuple


# -------------------------
# Default weight sampling
# -------------------------

def default_weight_sampler(inst: Dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    """
    Sample nonnegative weights that sum to 1 (Dirichlet).
    """
    # alpha = np.ones(7)  # uniform over simplex; increase alpha for less spiky
    # w = rng.dirichlet(alpha).astype(float)
    # w = rng.random(7).astype(float)  # each in [0,1)
    w = 10 ** rng.uniform(-3.0, 3.0, size=7)
    return w

# def default_weight_sampler(inst: Dict[str, Any], rng: np.random.Generator) -> np.ndarray:
#     """
#     Sample a 7-dim nonnegative weight vector:
#       w = [Ws_track, Ws_safe, Ws_u, Ws_smooth, Wt_track, Wt_safe, Wt_u]
#     """
#     meta = inst["meta"]
#     rho = float(meta.get("rho", inst["init_goal"].get("goal_radius", 1.0)))
#
#     def logu(lo: float, hi: float) -> float:
#         return float(10.0 ** rng.uniform(lo, hi))
#
#     Ws_track  = logu(-1.5,  2.0)
#     Ws_safe   = logu(-1.5,  2.0)
#     Ws_u      = logu(-3.0,  0.5)
#     Ws_smooth = logu(-3.0,  0.5)
#
#     if rho > 0:
#         Ws_track *= float(np.clip(0.2 / rho, 0.5, 5.0))
#
#     Wt_track  = Ws_track * logu(0.5, 1.5)
#     Wt_safe   = Ws_safe  * logu(0.5, 1.5)
#     Wt_u      = Ws_u     * logu(-0.5, 0.5)
#
#     w = np.array([Ws_track, Ws_safe, Ws_u, Ws_smooth,
#                   Wt_track, Wt_safe, Wt_u], dtype=float)
#     return np.clip(w, 0.0, 1e6)


# -------------------------
# Build FastQuadraticCost from w
# -------------------------

def make_fast_cost_from_w(
    inst: Dict[str, Any],
    w: np.ndarray,
    QuadCostConfig: Any,
    FastQuadraticCost: Any,
) -> Any:
    """
    Build the derivative-aware cost object used by FastILQRMPC.
    Requires the QuadCostConfig and FastQuadraticCost classes from your iLQR code.
    """
    meta = inst["meta"]
    init_goal = inst["init_goal"]

    X_GOAL = np.asarray(init_goal["x_end"], dtype=float)
    U_MAX  = float(meta["u_max"])
    R_SAFE = float(meta["r"])

    w = np.asarray(w, dtype=float).reshape(-1)
    if w.size != 7:
        raise ValueError(f"Expected w of length 7, got {w.size}")

    (Ws_track, Ws_safe, Ws_u, Ws_smooth,
     Wt_track, Wt_safe, Wt_u) = [float(x) for x in w]

    qc = QuadCostConfig(
        x_goal=X_GOAL,
        u_max=U_MAX,
        r_safe=R_SAFE,
        W_S_TRACK=Ws_track,
        W_S_SAFE=Ws_safe,
        W_S_U=Ws_u,
        W_S_SMOOTH=Ws_smooth,
        W_T_TRACK=Wt_track,
        W_T_SAFE=Wt_safe,
        W_T_U=Wt_u,
        W_T_SMOOTH=0.0,   # your convention
    )
    return FastQuadraticCost(qc)


# -------------------------
# Main: weight search for one instance (fast iLQR MPC)
# -------------------------

def run_weight_search_for_instance_fast_ilqr(
    inst: Dict[str, Any],
    scorer: Any,
    run_closed_loop_fast_ilqr_mpc: Callable,
    mpc_builder: Callable[[Dict[str, Any], Any, Any], Any],
    ilqr_cfg_base: Any,
    QuadCostConfig: Any,
    FastQuadraticCost: Any,
    K: int = 64,
    T_outer: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    weight_sampler: Optional[Callable[[Dict[str, Any], np.random.Generator], np.ndarray]] = None,
    keep_top: int = 5,
    early_stop_Y: Optional[float] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    For ONE instance:
      - sample K weight vectors w
      - build cost object from w
      - build FastILQRMPC via mpc_builder(inst, cost_obj)
      - run closed-loop using run_closed_loop_fast_ilqr_mpc
      - score and return best + top-N

    Inputs you must provide:
      - mpc_builder(inst, cost_obj) -> FastILQRMPC
        (this function constructs dyn + FastILQRMPC using inst-specific dynamics)
    """
    if rng is None:
        rng = np.random.default_rng()
    if weight_sampler is None:
        weight_sampler = default_weight_sampler

    meta = inst["meta"]
    x0 = np.asarray(inst["init_goal"]["x0"], dtype=float)
    step_fn_true = inst["step"]

    if T_outer is None:
        T_outer = int(meta.get("N", 25))

    # Make sure bounds match this instance
    u_max = float(meta["u_max"])
    try:
        cfg_run = replace(ilqr_cfg_base, u_min=-u_max, u_max=u_max)
    except Exception:
        cfg_run = ilqr_cfg_base
        if hasattr(cfg_run, "u_min"):
            cfg_run.u_min = -u_max
        if hasattr(cfg_run, "u_max"):
            cfg_run.u_max = u_max

    best: Optional[Dict[str, Any]] = None
    top: List[Dict[str, Any]] = []

    for k in range(int(K)):
        w = weight_sampler(inst, rng)
        cost_obj = make_fast_cost_from_w(inst, w, QuadCostConfig, FastQuadraticCost)

        # Build a fresh MPC object for this weight vector
        mpc = mpc_builder(inst, cost_obj, ilqr_cfg_base)

        # Ensure its cfg matches per-instance bounds if your builder didn't already
        if hasattr(mpc, "cfg") and hasattr(mpc.cfg, "u_min"):
            mpc.cfg.u_min = getattr(cfg_run, "u_min", mpc.cfg.u_min)
            mpc.cfg.u_max = getattr(cfg_run, "u_max", mpc.cfg.u_max)

        X_cl, U_cl = run_closed_loop_fast_ilqr_mpc(
            mpc=mpc,
            step_fn_true=step_fn_true,
            x0=x0,
            T_outer=T_outer,
        )

        score_dict = scorer.score(X_cl, U_cl)
        Y = float(score_dict["score"])
        success = score_dict["success"]

        item = {"w": w, "Y": Y, "score": score_dict, "X": X_cl, "U": U_cl, "success": success}

        if best is None or Y > float(best["Y"]):
            best = item
            if verbose:
                w_str = np.array2string(w, precision=3, suppress_small=True)
                print(f"[{k+1:03d}/{K}] new best Y={Y:.4f} w={w_str}")

        top.append(item)
        top.sort(key=lambda d: float(d["Y"]), reverse=True)
        if len(top) > int(keep_top):
            top = top[: int(keep_top)]

        if early_stop_Y is not None and Y >= float(early_stop_Y):
            if verbose:
                print(f"Early stop at k={k+1}: Y={Y:.4f} >= {early_stop_Y}")
            break

    if best is None:
        raise RuntimeError("Weight search evaluated no candidates (K=0?)")

    out = dict(best)
    out["top"] = top
    return out