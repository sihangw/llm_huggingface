# import numpy as np
# from dataclasses import dataclass
# from typing import Callable, Optional, Dict, Any
#
# @dataclass
# class CompositeScoreConfig:
#     dt: float                      # time step
#     D: float = 1.0                 # characteristic state/target scale (e.g. target diameter)
#     V_ref: float = 1.0             # normalization for violations
#     beta1: float = 0.5             # weight on exp(-m_tilde_plus)
#     beta2: float = 0.2             # weight on exp(-min(tau*,T)/T)
#     beta6: float = 0.2             # weight on violations / V_ref (subtracted!)
#
#
# class CompositeTrajectoryScorer:
#     """
#     Simplified composite score (no relaxation terms S_alpha, S_lambda, S_T).
#
#     Returns:
#     - S_raw : unnormalized composite score (can be <0 or >1),
#     - Y     : normalized score in (0,1),
#     - L     : loss = 1 - Y in (0,1).
#     """
#
#     def __init__(
#         self,
#         phi_target: Callable[[np.ndarray], float],
#         constraint_excess: Optional[Callable[[np.ndarray, np.ndarray, float], float]] = None,
#         config: CompositeScoreConfig = CompositeScoreConfig(dt=0.1),
#     ):
#         self.phi_target = phi_target
#         self.constraint_excess = constraint_excess
#         self.cfg = config
#
#     def score(self, X: np.ndarray, U: np.ndarray) -> Dict[str, Any]:
#         """
#         Compute composite score for discrete trajectory (X,U).
#
#         X : (H+1, n)
#         U : (H, m)
#         """
#         X = np.asarray(X, dtype=float)
#         U = np.asarray(U, dtype=float)
#         H = U.shape[0]
#         assert X.shape[0] == H + 1
#
#         dt = self.cfg.dt
#         T_total = H * dt
#
#         # 1) m_min = min_t phi_T(x(t))
#         phi_vals = np.array([self.phi_target(x_k) for x_k in X])
#         m_min = float(np.min(phi_vals))
#
#         # success indicator
#         success = m_min <= 0.0
#
#         # scaled margin
#         m_tilde = m_min / self.cfg.D
#         m_tilde_plus = max(m_tilde, 0.0)
#
#         # 2) tau* = first time we enter target set, or +inf if never
#         hit_indices = np.where(phi_vals <= 0.0)[0]
#         if hit_indices.size > 0:
#             first_idx = int(hit_indices[0])
#             tau_star = first_idx * dt
#         else:
#             tau_star = float("inf")
#
#         if np.isfinite(tau_star):
#             tau_over_T = min(tau_star, T_total) / T_total
#         else:
#             tau_over_T = 1.0  # never hit → treat as T
#
#         # 3) violations integral approx
#         if self.constraint_excess is None:
#             violations = 0.0
#         else:
#             viol_sum = 0.0
#             for k in range(H):
#                 t_k = k * dt
#                 x_k = X[k]
#                 u_k = U[k]
#                 excess = float(self.constraint_excess(x_k, u_k, t_k))
#                 viol_sum += max(0.0, excess) * dt
#             violations = viol_sum
#
#         # components
#         margin_score    = self.cfg.beta1 * np.exp(-m_tilde_plus)
#         time_score      = self.cfg.beta2 * np.exp(-tau_over_T)
#         violation_score = -self.cfg.beta6 * (violations / self.cfg.V_ref)
#
#         # raw composite score (unbounded in principle)
#         S_raw = (
#             (1.0 if success else 0.0)
#             + margin_score
#             + time_score
#             + violation_score
#         )
#
#         # normalized score in (0,1) via logistic squashing
#         Y = 1.0 / (1.0 + np.exp(-S_raw))
#
#         # corresponding loss (like in the paper): L(x) = 1 - Y(x) ∈ (0,1)
#         L = 1.0 - Y
#
#         return {
#             "S_raw": float(S_raw),
#             "Y": float(Y),
#             "L": float(L),
#             "success": bool(success),
#             "m_min": m_min,
#             "tau_star": tau_star,
#             "violations": float(violations),
#             "margin_score": float(margin_score),
#             "time_score": float(time_score),
#             "violation_score": float(violation_score),
#         }

import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional, Dict, Any

@dataclass
class CompositeScoreConfig:
    dt: float
    D: float = 1.0

    # original terms
    beta_margin: float = 0.5   # margin term weight
    beta_time: float = 0.2   # time term weight
    # beta6: float = 0.2   # violations weight (subtracted)


    # new: control regularization terms (optional, but helps differentiate)
    beta_u: float = 0.05         # penalize control effort
    beta_du: float = 0.02        # penalize control rate (smoothness)
    U_ref: float = 1.0           # normalization for u (e.g., u_max)
    dU_ref: float = 1.0          # normalization for delta-u


class CompositeTrajectoryScorer:
    """
    Simple, differentiating score.

    S = - w_goal * terminal_dist
        - w_time * time_to_hit
        - w_viol * violations
        - w_u    * effort
        - w_du   * smoothness

    Returns:
      S_raw, success, tau_star, violations, effort, smoothness, terminal_dist
    """
    def __init__(
        self,
        phi_target: Callable[[np.ndarray], float],
        # constraint_excess: Optional[Callable[[np.ndarray, np.ndarray, float], float]] = None,
        config: CompositeScoreConfig = CompositeScoreConfig(dt=0.1),
    ):
        self.phi_target = phi_target
        # self.constraint_excess = constraint_excess
        self.cfg = config

    def score(self, X: np.ndarray, U: np.ndarray) -> Dict[str, Any]:
        X = np.asarray(X, dtype=float)
        U = np.asarray(U, dtype=float)
        H = U.shape[0]
        assert X.shape[0] == H + 1

        dt = self.cfg.dt
        T_total = H * dt

        # 1) phi values and min margin
        phi_vals = np.array([float(self.phi_target(x_k)) for x_k in X], dtype=float)
        m_min = float(np.min(phi_vals))
        m_plus = max(m_min, 0.0)
        # original-ish components
        margin_raw = 1.0 / (1.0 + m_plus / (self.cfg.D  + 1e-12))
        margin_score = np.clip(margin_raw, 0.0, 1.0)


        # 2) tau* time-to-hit
        hit_indices = np.where(phi_vals <= 0.0)[0]
        if hit_indices.size > 0:
            first_idx = int(hit_indices[0])  # 0..H
            tau_frac = first_idx / float(H)  # in [0,1], fraction of horizon
            # time_score = 1.0 - (tau_frac ** gamma)  # in (0,1], 1 best
            tau_star = float(first_idx * dt)
        else:
            tau_frac = 1.0
            # time_score = 0.0  # never hit -> worst
            tau_star = float("inf")

        gamma = 2.0  # >1 emphasizes early hits
        time_raw = 1.0 - (tau_frac ** gamma)
        time_score = np.clip(time_raw, 0.0, 1.0)

        # # 3) violations integral approx
        # if self.constraint_excess is None:
        #     violations = 0.0
        # else:
        #     viol_sum = 0.0
        #     for k in range(H):
        #         t_k = k * dt
        #         x_k = X[k]
        #         u_k = U[k]
        #         excess = float(self.constraint_excess(x_k, u_k, t_k))
        #         viol_sum += max(0.0, excess) * dt
        #     violations = float(viol_sum)
        #
        # V_max = 0.5 * T_total  # example
        # violation_score = -self.cfg.beta6 * min(violations / (V_max + 1e-12), 1.0)

        # 4) control effort + smoothness (to break ties between equally successful runs)
        # effort: integral ||u||^2 dt
        u2 = float(np.sum(U * U) * dt)
        u_max = self.cfg.U_ref
        m = U.shape[1]
        effort_raw = u2 / (T_total * m * u_max ** 2 + 1e-12)
        effort_score = 1.0 - np.clip(effort_raw, 0.0, 1)

        # smoothness: integral ||Δu||^2 dt
        dU = U[1:] - U[:-1]
        du2 = float(np.sum(dU * dU) * dt)
        smooth_raw = du2 / ((self.cfg.dU_ref ** 2) * max(T_total, 1e-12) + 1e-12)
        smooth_score = 1.0 - np.clip(smooth_raw, 0.0, 1.0)


        scores = np.array([margin_score, time_score, effort_score, smooth_score], dtype=float)
        a = np.array([self.cfg.beta_margin, self.cfg.beta_time, self.cfg.beta_u, self.cfg.beta_du], dtype=float)

        a = np.maximum(a, 0.0)
        Y = float(np.dot(a, scores) / (np.sum(a) + 1e-12))

        return {
            "success": bool(m_min <= 0.0),
            "score": float(Y),
            "margin_score": float(margin_score),
            "time_score": float(time_score),
            "effort_score": float(effort_score),
            "smooth_score": float(smooth_score),
        }