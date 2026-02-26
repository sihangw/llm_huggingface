# weight_search.py
from __future__ import annotations

import numpy as np
from dataclasses import replace
from typing import Any, Callable, Dict, List, Optional, Tuple


# -------------------------
# Default weight sampling
# -------------------------

def default_weight_sampler(inst: Dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    """
    Sample an 8-dim nonnegative weight vector:
      w = [Ws_track, Ws_safe, Ws_u, Ws_smooth, Wt_track, Wt_safe, Wt_u, Wt_smooth]

    Uses log-uniform ranges to explore orders of magnitude.
    """
    meta = inst["meta"]
    rho = float(meta.get("rho", inst["init_goal"].get("goal_radius", 1.0)))

    def logu(lo: float, hi: float) -> float:
        return float(10.0 ** rng.uniform(lo, hi))

    # Stage weights
    Ws_track  = logu(-1.5,  2.0)   # ~[0.03, 100]
    Ws_safe   = logu(-1.5,  2.0)
    Ws_u      = logu(-3.0,  0.5)   # ~[0.001, 3.16]
    Ws_smooth = logu(-3.0,  0.5)

    # Optional instance-aware nudge: tiny goal tolerance => need stronger tracking
    if rho > 0:
        Ws_track *= float(np.clip(0.2 / rho, 0.5, 5.0))

    # Terminal weights (usually stronger on goal + safety)
    Wt_track  = Ws_track * logu(0.5, 1.5)   # multiply by ~[3.16, 31.6]
    Wt_safe   = Ws_safe  * logu(0.5, 1.5)
    Wt_u      = Ws_u     * logu(-0.5, 0.5)
    # Wt_smooth = 0.0

    w = np.array([Ws_track, Ws_safe, Ws_u, Ws_smooth,
                  Wt_track, Wt_safe, Wt_u], dtype=float)
    return np.clip(w, 0.0, 1e6)


# -------------------------
# Default cost template (from w)
# -------------------------

def make_cost_fn_from_w(inst: Dict[str, Any], w: np.ndarray) -> Callable:
    """
    Build cost_fn(x,u,u_prev=None,terminal=False) from an 8-dim weight vector w.

    Terms:
      e_track  = ||x - X_GOAL||_2^2
      e_safe   = max(0, ||x||_inf - R_SAFE)^2
      e_u      = ||u / U_MAX||_2^2
      e_smooth = ||u - u_prev||_2^2   (0 if u_prev is None)

    Stage vs terminal weights use first 4 vs last 4 entries of w.
    """
    meta = inst["meta"]
    init_goal = inst["init_goal"]

    X_GOAL = np.asarray(init_goal["x_end"], dtype=float)
    U_MAX  = float(meta["u_max"])
    R_SAFE = float(meta["r"])

    w = np.asarray(w, dtype=float).reshape(-1)
    if w.size != 7:
        raise ValueError(f"Expected w of length 8, got {w.size}")

    (Ws_track, Ws_safe, Ws_u, Ws_smooth,
     Wt_track, Wt_safe, Wt_u) = [float(x) for x in w]

    eps = 1e-12

    def cost_fn(x, u, u_prev=None, terminal: bool = False) -> float:
        x = np.asarray(x, dtype=float)
        u = np.asarray(u, dtype=float)

        e_track = float(np.sum((x - X_GOAL) ** 2))
        e_u     = float(np.sum((u / (U_MAX + eps)) ** 2))

        if u_prev is None:
            e_smooth = 0.0
        else:
            u_prev = np.asarray(u_prev, dtype=float)
            e_smooth = float(np.sum((u - u_prev) ** 2))

        inf_norm = float(np.max(np.abs(x)))
        e_safe = float(max(0.0, inf_norm - R_SAFE) ** 2)

        if terminal:
            J = (Wt_track * e_track +
                 Wt_safe  * e_safe  +
                 Wt_u     * e_u     )
        else:
            J = (Ws_track * e_track +
                 Ws_safe  * e_safe  +
                 Ws_u     * e_u     +
                 Ws_smooth* e_smooth)

        return float(J)

    return cost_fn


# -------------------------
# Helper: call run_closed_loop_mppi with/without rng
# -------------------------

def _run_closed_loop_mppi_wrapper(
    run_closed_loop_mppi: Callable,
    step_fn: Callable,
    cost_fn: Callable,
    x0: np.ndarray,
    T_outer: int,
    cfg: Any,
    m: int,
    rng: Optional[np.random.Generator],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Some versions of run_closed_loop_mppi accept rng=..., others don't.
    This wrapper tries both.
    """
    try:
        return run_closed_loop_mppi(
            step_fn=step_fn, cost_fn=cost_fn, x0=x0, T_outer=T_outer, cfg=cfg, m=m, rng=rng
        )
    except TypeError:
        return run_closed_loop_mppi(
            step_fn=step_fn, cost_fn=cost_fn, x0=x0, T_outer=T_outer, cfg=cfg, m=m
        )


# -------------------------
# Main: weight search for one instance
# -------------------------

def run_weight_search_for_instance(
    inst: Dict[str, Any],
    scorer: Any,
    run_closed_loop_mppi: Callable,
    mppi_cfg_base: Any,
    K: int = 64,
    T_outer: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    weight_sampler: Optional[Callable[[Dict[str, Any], np.random.Generator], np.ndarray]] = None,
    cost_fn_builder: Optional[Callable[[Dict[str, Any], np.ndarray], Callable]] = None,
    keep_top: int = 5,
    early_stop_Y: Optional[float] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    For ONE theta instance:
      - sample K candidate weight vectors w
      - run closed-loop MPPI for each
      - score trajectory with scorer.score(X,U)["Y"]
      - return best and top-N candidates

    Returns dict:
      best = {
        "w": best_w,
        "Y": best_Y,
        "score": score_dict,
        "X": X_cl,
        "U": U_cl,
        "top": [ ...top candidates... ],
      }
    """
    if rng is None:
        rng = np.random.default_rng()

    if weight_sampler is None:
        weight_sampler = default_weight_sampler

    if cost_fn_builder is None:
        cost_fn_builder = make_cost_fn_from_w

    meta = inst["meta"]
    step_fn = inst["step"]
    x0 = np.asarray(inst["init_goal"]["x0"], dtype=float)
    m = int(meta["m"])
    u_max = float(meta["u_max"])

    if T_outer is None:
        T_outer = int(meta["N"])

    # Ensure MPPI bounds match this instance
    try:
        cfg_run = replace(mppi_cfg_base, u_min=-u_max, u_max=u_max)
    except Exception:
        cfg_run = mppi_cfg_base
        if hasattr(cfg_run, "u_min"):
            cfg_run.u_min = -u_max
        if hasattr(cfg_run, "u_max"):
            cfg_run.u_max = u_max

    best: Optional[Dict[str, Any]] = None
    top: List[Dict[str, Any]] = []

    for k in range(int(K)):
        w = weight_sampler(inst, rng)
        cost_fn = cost_fn_builder(inst, w)

        X_cl, U_cl = _run_closed_loop_mppi_wrapper(
            run_closed_loop_mppi=run_closed_loop_mppi,
            step_fn=step_fn,
            cost_fn=cost_fn,
            x0=x0,
            T_outer=T_outer,
            cfg=cfg_run,
            m=m,
            rng=rng,
        )

        score_dict = scorer.score(X_cl, U_cl)
        Y = float(score_dict["score"])
        # print("score")
        # print(Y)

        item = {"w": w, "Y": Y, "score": score_dict, "X": X_cl, "U": U_cl}

        # maintain best
        if best is None or Y > float(best["Y"]):
            best = item
            if verbose:
                w_str = np.array2string(w, precision=3, suppress_small=True)
                print(f"[{k+1:03d}/{K}] new best Y={Y:.4f} w={w_str}")

        # maintain top-N list
        top.append(item)
        top.sort(key=lambda d: float(d["Y"]), reverse=True)
        if len(top) > int(keep_top):
            top = top[: int(keep_top)]

        # early stop if "good enough"
        if early_stop_Y is not None and Y >= float(early_stop_Y):
            if verbose:
                print(f"Early stop at k={k+1}: Y={Y:.4f} >= {early_stop_Y}")
            break

    if best is None:
        raise RuntimeError("Weight search evaluated no candidates (K=0?)")

    out = dict(best)
    out["top"] = top
    return out