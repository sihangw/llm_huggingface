import numpy as np
from typing import Dict, Any, Optional, Tuple, List

def build_instance_prompt(
    inst: Dict[str, Any],
    H: int,                               # MPC horizon in steps
    T_mpc: Optional[int] = None,          # overall MPC/rollout length in steps(if None, use inst["meta"]["N"])
    R_safe: Optional[float] = None,       # safe radius in your cost (if None, use inst["meta"]["r"])
    eps: float = 1e-12,
) -> Tuple[Dict[str, float], str]:
    """
    Returns:
      feat: dict of scalar features (your writeup items 1-6)
      prompt: formatted string containing the same info (for LLM / logging)
    """
    meta = inst["meta"]
    drift = inst["drift"]
    inp = inst["input"]  # NOTE: your generator uses key "input" (not "inp")

    n = int(meta["n"])
    m = int(meta["m"])
    dt = float(meta["dt"])
    r = float(meta["r"])
    u_max = float(meta["u_max"])
    rho_goal = float(meta["rho"])

    if T_mpc is None:
        T_mpc = int(meta["N"])
    if R_safe is None:
        R_safe = r

    x0 = np.asarray(inst["init_goal"]["x0"], dtype=float).reshape(n)
    xg = np.asarray(inst["init_goal"]["x_end"], dtype=float).reshape(n)

    # (2) goal difficulty
    d0 = float(np.linalg.norm(x0 - xg, ord=2))
    d0_norm = float(d0 / (R_safe + eps))

    # (3) linear stability
    A = np.asarray(drift["A"], dtype=float).reshape(n, n)
    Ad = np.eye(n) + dt * A
    eigvals = np.linalg.eigvals(Ad)
    rho_lin = float(np.max(np.abs(eigvals)))
    Ad_2 = float(np.linalg.norm(Ad, ord=2))

    # (4) control authority
    G0 = np.asarray(inp["G0"], dtype=float).reshape(n, m)
    g_norm = float(np.linalg.norm(G0, 2))

    # (5) nonlinearity summary
    # drift["C"] has k>=2 blocks of shape (M_k, n)
    # input["D"] has k>=1 blocks of shape (M_k, n, m)
    eta_f = 0.0
    for k, Ck in drift["C"].items():
        if Ck is None:
            continue
        Ck = np.asarray(Ck, dtype=float)
        if Ck.size == 0:
            continue
        eta_f += float(np.linalg.norm(Ck, ord="fro")) * (r ** (int(k) - 1))

    eta_G = 0.0
    for k, Dk in inp["D"].items():
        if Dk is None:
            continue
        Dk = np.asarray(Dk, dtype=float)
        if Dk.size == 0:
            continue
        eta_G += float(np.linalg.norm(Dk.reshape(Dk.shape[0], -1), ord="fro")) * (r ** (int(k) - 1))

    eta_nl = float(eta_f + u_max * eta_G)

    # (6) reachability difficulty ratio (rough “is goal far given horizon + control strength?”)
    T = float(T_mpc * dt)
    reach_ratio = float(d0 / (T * u_max * (g_norm + eps)))

    feat = {
        # (1)
        "n": float(n),
        "m": float(m),
        "dt": float(dt),
        "H": int(H),
        "T_mpc": float(T_mpc),
        "u_max": float(u_max),
        "R_safe": float(R_safe),
        "rho_goal": float(rho_goal),
        # (2)
        "d0_norm": float(d0_norm                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               ),
        # (3)
        "rho_lin": float(rho_lin),
        # (4)
        "g_norm": float(g_norm),
        # (5)
        # "eta_f": float(eta_f),
        # "eta_G": float(eta_G),
        "eta_nl": float(eta_nl),
        # (6)
        "reach_ratio": float(reach_ratio),
    }

    # Prompt with explanations (human/LLM-friendly)
    prompt = (
        "SYSTEM SUMMARY:"
        f"- Dimensions: n={n} (state dimension, x in R^{n}), m={m} (control dimension, u in R^{m})"
        f"- Discretization: dt={dt:.4g} (time step), H={H} (planning horizon per MPC in steps), T_mpc={T_mpc} (overall planning horizon of MPC in steps)"
        f"- Initial state: x0={np.array2string(x0, precision=3, separator=',')} and our target state:  x_goal={np.array2string(xg, precision=3, separator=',')}"
        f"- Constraints/region: u_max={u_max:.4g} (control bound: |u|<=u_max), R_safe={R_safe:.4g} (safe-region radius), rho_goal={rho_goal:.4g} (goal tolerance radius)"
        f"- Goal difficulty: normalized Euclidean distance d0_norm=||x0-x_goal||2/(R_safe+eps)={d0_norm:.4g}"
        f"- Linear stability (discrete linear part Ad=I+dt*A)): rho_lin=max|eig(I+dt*A)|={rho_lin:.4g}"
        f"- Control authority at x=0: g_norm=||G0||2={g_norm:.4g} (bigger => easier to steer)"
        # f"- Nonlinearity level: eta_nl={eta_nl:.4g} (combined drift+input nonlinearity on box, larger => more nonlinear/harder)\n"
        f"- Nonlinearity score: eta_nl={eta_nl:.4g} (heuristic proxy for magnitude/sensitivity of higher-order polynomial terms on |x|_inf <= r under ||u||_inf <= u_max; larger means more nonlinear/harder)"
        f"- Initial state: x0={np.array2string(x0, precision=3, separator=',')}"
        f"- Target state: x_goal={np.array2string(xg, precision=3, separator=',')}"
        f"- Goal difficulty: d0_norm=||x0-x_goal||2/(R_safe+eps)={d0_norm:.4g}"
        f"- Reachability ratio: reach_ratio=d0/(T*u_max*(g_norm+eps))={reach_ratio:.4g} (smaller => easier to reach within horizon)"
    )
    # prompt = (
    #     "INSTANCE SUMMARY (features)\n"
    #     f"(1) task: n={n}, m={m}, dt={dt:.4g}, H={H_mpc}, u_max={u_max:.4g}, R_safe={R_safe:.4g}, rho={rho_goal:.4g}\n"
    #     f"(2) goal difficulty: d0=||x0-xg||2={d0:.4g}, d0/(R_safe+eps)={d0_norm:.4g}\n"
    #     f"(3) linear stability: rho_lin=max|eig(I+dt*A)|={rho_lin:.4g}\n"
    #     f"(4) control authority: g_F=||G0||F={g_F:.4g}, sigma_min(G0)={sigma_min:.4g}\n"
    #     f"(5) nonlinearity: eta_f={eta_f:.4g}, eta_G={eta_G:.4g}, eta_nl=eta_f+u_max*eta_G={eta_nl:.4g}\n"
    #     f"(6) reachability ratio: T=H*dt={T:.4g}, reach_ratio=d0/(T*u_max*(sigma_min+eps))={reach_ratio:.4g}\n"
    # )

    return feat, prompt

def build_beta_prompt_simple(
    beta_margin: float,
    beta_time: float,
    beta_u: float,
    beta_du: float,
) -> Tuple[Dict[str, float], str]:
    feat = {
        "beta_margin": float(beta_margin),
        "beta_time": float(beta_time),
        "beta_u": float(beta_u),
        "beta_du": float(beta_du),
    }

    prompt = (
        "SCORE METRICS (4 coefficients)"
        "- The total score Y is a weighted average of 4 components:"
        "  Y = (beta_margin*margin_score + beta_time*time_score + beta_u*effort_score + beta_du*smooth_score)/ (beta_margin + beta_time + beta_u + beta_du)"
        "- Meanings (larger weight => that component matters more):"
        f"  beta_margin={beta_margin:.4g}: prioritize reaching/being inside the target set (goal satisfaction)."
        f"  beta_time={beta_time:.4g}: prioritize reaching the target earlier (time-to-hit)."
        f"  beta_u={beta_u:.4g}: prioritize smaller control energy/effort (||u||^2)."
        f"  beta_du={beta_du:.4g}: prioritize smoother control (small changes delta_u)."
        "- All weights are assumed non-negative."
    )
    return feat, prompt