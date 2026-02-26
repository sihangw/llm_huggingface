from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Dict, Any
import numpy as np


StepFn = Callable[[np.ndarray, np.ndarray], np.ndarray]
CostFn = Callable[[np.ndarray, np.ndarray, Optional[np.ndarray], bool], float]


@dataclass
class MPPIConfig:
    H: int = 25                 # planning horizon
    num_samples: int = 256      # trajectories per iteration
    num_iters: int = 3          # MPPI iterations per MPC step
    lambda_: float = 1.0        # temperature
    noise_sigma: float = 0.3    # stddev of Gaussian noise (scalar or per-dim supported below)
    u_min: Optional[np.ndarray] = None
    u_max: Optional[np.ndarray] = None

    # warm start / shifting behavior
    shift_fill: str = "zero"    # "zero" or "repeat_last"

    # numerical stability
    eps: float = 1e-12


class MPPIController:
    """
    MPPI-based receding-horizon MPC controller.

    Usage:
        ctrl = MPPIController(step_fn, cost_fn, cfg, m=..., rng=...)
        u = ctrl.act(x)  # returns first control of optimized sequence
        ctrl.shift_after_apply()
    """

    def __init__(
        self,
        step_fn: StepFn,
        cost_fn: CostFn,
        cfg: MPPIConfig,
        m: int,
        rng: Optional[np.random.Generator] = None,
        u_init: Optional[np.ndarray] = None,
    ):
        self.step_fn = step_fn
        self.cost_fn = cost_fn
        self.cfg = cfg
        self.m = int(m)
        self.rng = rng if rng is not None else np.random.default_rng()

        H = int(cfg.H)
        if u_init is None:
            self.U_mean = np.zeros((H, self.m), dtype=float)
        else:
            u_init = np.asarray(u_init, dtype=float)
            assert u_init.shape == (H, self.m)
            self.U_mean = u_init.copy()

        # normalize bounds to arrays (m,)
        self.u_min, self.u_max = self._normalize_bounds(cfg.u_min, cfg.u_max)

    def _normalize_bounds(
        self,
        u_min: Optional[np.ndarray],
        u_max: Optional[np.ndarray],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if u_min is None and u_max is None:
            return None, None

        def as_vec(v, name):
            if v is None:
                return None
            v = np.asarray(v, dtype=float)
            if v.ndim == 0:
                return np.full((self.m,), float(v), dtype=float)
            if v.shape == (self.m,):
                return v
            raise ValueError(f"{name} must be scalar or shape (m,)")

        u_min_v = as_vec(u_min, "u_min")
        u_max_v = as_vec(u_max, "u_max")
        return u_min_v, u_max_v

    def _clip_u(self, U: np.ndarray) -> np.ndarray:
        if self.u_min is None and self.u_max is None:
            return U
        lo = self.u_min if self.u_min is not None else -np.inf
        hi = self.u_max if self.u_max is not None else np.inf
        return np.clip(U, lo, hi)

    def rollout_cost(self, x0: np.ndarray, U_seq: np.ndarray) -> (float, np.ndarray):
        """
        Roll out for H steps (planning horizon) and accumulate cost.
        """
        x = np.asarray(x0, dtype=float).copy()
        U_seq = np.asarray(U_seq, dtype=float)
        assert U_seq.shape == (self.cfg.H, self.m)

        n = x.shape[0]
        X_traj = np.zeros((self.cfg.H + 1, n), dtype=float)
        X_traj[0] = x

        total = 0.0
        u_prev = None

        for k in range(self.cfg.H):
            u = U_seq[k]
            terminal = (k == self.cfg.H - 1)
            total += float(self.cost_fn(x, u, u_prev=u_prev, terminal=terminal))
            x = self.step_fn(x, u)
            X_traj[k + 1] = x
            u_prev = u

        return float(total), X_traj

    def update(self, x0: np.ndarray) -> Dict[str, Any]:
        """
        One MPPI update step that refines self.U_mean around current state x0.
        Returns diagnostics.
        """
        H, m = self.cfg.H, self.m
        U_mean = self.U_mean

        # noise sigma can be scalar or (m,) or (H,m)
        sig = self.cfg.noise_sigma
        if np.isscalar(sig):
            noise_sigma = np.full((H, m), float(sig), dtype=float)
        else:
            sig = np.asarray(sig, dtype=float)
            if sig.shape == (m,):
                noise_sigma = np.tile(sig.reshape(1, m), (H, 1))
            elif sig.shape == (H, m):
                noise_sigma = sig
            else:
                raise ValueError("noise_sigma must be scalar, shape (m,), or shape (H,m)")

        costs = np.zeros(self.cfg.num_samples, dtype=float)
        noises = np.zeros((self.cfg.num_samples, H, m), dtype=float)

        # sample & evaluate
        for i in range(self.cfg.num_samples):
            eps = self.rng.normal(0.0, 1.0, size=(H, m))
            dU = noise_sigma * eps
            U_i = self._clip_u(U_mean + dU)

            costs[i], _ = self.rollout_cost(x0, U_i)
            noises[i] = dU

        # importance weights (softmin)
        J_min = float(np.min(costs))
        w = np.exp(-(costs - J_min) / max(self.cfg.lambda_, self.cfg.eps))
        w_sum = float(np.sum(w)) + self.cfg.eps
        w = w / w_sum

        # weighted noise update
        dU_mean = np.tensordot(w, noises, axes=(0, 0))  # (H,m)
        self.U_mean = self._clip_u(U_mean + dU_mean)

        return {
                "costs": costs,
                "weights": w,
                "J_min": J_min,
                "U_mean": self.U_mean.copy(),
        }

    def optimize(self, x0: np.ndarray) -> Dict[str, Any]:
        """
        Run num_iters MPPI iterations and return last diagnostics.
        """
        info = {}
        for _ in range(self.cfg.num_iters):
            info = self.update(x0)
        return info

    def act(self, x: np.ndarray) -> np.ndarray:
        """
        Optimize from current state and return the first control to apply.
        """
        self.optimize(x)
        return self.U_mean[0].copy()

    def shift_after_apply(self):
        """
        Receding horizon shift: drop first control, shift left, fill last.
        """
        if self.cfg.H <= 1:
            return
        if self.cfg.shift_fill == "repeat_last":
            last = self.U_mean[-1].copy()
        else:
            last = np.zeros((self.m,), dtype=float)

        self.U_mean[:-1] = self.U_mean[1:]
        self.U_mean[-1] = last


def run_closed_loop_mppi(
    step_fn: StepFn,
    cost_fn: CostFn,
    x0: np.ndarray,
    T_outer: int,
    cfg: MPPIConfig,
    m: int,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience wrapper: closed-loop MPC simulation for T_outer steps.
    """
    rng = rng if rng is not None else np.random.default_rng()
    ctrl = MPPIController(step_fn=step_fn, cost_fn=cost_fn, cfg=cfg, m=m, rng=rng)

    x = np.asarray(x0, dtype=float).copy()
    n = x.shape[0]

    X_cl = np.zeros((T_outer + 1, n), dtype=float)
    U_cl = np.zeros((T_outer, m), dtype=float)
    X_cl[0] = x

    for t in range(T_outer):
        u = ctrl.act(x)
        U_cl[t] = u
        x = step_fn(x, u)
        X_cl[t + 1] = x
        ctrl.shift_after_apply()

    return X_cl, U_cl
