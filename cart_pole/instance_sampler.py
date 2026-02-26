from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
from poly_theta_generator import PolyThetaConfig
from poly_theta_sampler import ThetaSampler, ThetaSamplerConfig
import numpy as np


def _randint_inclusive(rng: np.random.Generator, lo: int, hi: int) -> int:
    if lo > hi:
        raise ValueError(f"bad int range [{lo},{hi}]")
    return int(rng.integers(lo, hi + 1))


def _randfloat_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    if lo > hi:
        raise ValueError(f"bad float range [{lo},{hi}]")
    return float(rng.uniform(lo, hi))


def _randfloat_loguniform(rng: np.random.Generator, log10_lo: float, log10_hi: float) -> float:
    if log10_lo > log10_hi:
        raise ValueError(f"bad log10 range [{log10_lo},{log10_hi}]")
    return float(10.0 ** rng.uniform(log10_lo, log10_hi))


@dataclass
class ExperimentSamplerConfig:
    # -------- fixed hyperparams --------
    dt: float = 0.1
    N: int = 25                 # default time length for instance generation
    K: int = 64                 # weight-search samples
    max_iters: int = 10         # iLQR iterations

    # -------- sampled system sizes --------
    n_range: Tuple[int, int] = (2, 5)
    m_range: Tuple[int, int] = (1, 3)
    d_f_range: Tuple[int, int] = (1, 3)
    d_G_range: Tuple[int, int] = (1, 3)

    # -------- sampled system sizes --------
    u_max_range: Tuple[float, float] = (0.5, 5.0)
    mu_range: Tuple[float, float] = (0.15, 0.7)

    # -------- sampled task mode --------
    prob_mode: Tuple[float, float, float] = (0.3, 0.5, 0.2)

    # -------- sampled horizons --------
    H_range: Tuple[int, int] = (3, 10)  # MPC horizon (steps); will be clamped to <= N

    # T_outer can be less or greater than N
    # e.g. (0.8, 1.5) => T_outer in [ceil(0.8N), floor(1.5N)]
    T_outer_slack_frac_range: Tuple[float, float] = (0.8, 1.5)

    # -------- sampled safe radius used by cost/prompt --------
    # Interpret as R_safe = r_safe_frac * r_box (inst["meta"]["r"])
    # Keeps it consistent with the generator's state box scale.
    R_safe: Tuple[float, float] = (2.0, 10.0)

    # -------- sampled score betas (NOT fixed) --------
    # log-uniform in base-10: beta = 10^U(log10_lo, log10_hi)
    beta_log10_ranges: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]] = (
        (-0.3, 0.7),  # beta_margin ~ [0.5, 5]
        (-0.3, 0.7),  # beta_time   ~ [0.5, 5]
        (-1.0, 0.5),  # beta_u      ~ [0.1, 3.16]
        (-1.0, 0.5),  # beta_du     ~ [0.1, 3.16]
    )
    beta_zero_prob: float = 0.0  # optional: with this prob, set a beta to 0 (sparsify). keep 0.0 if you don't want.

    # -------- seeds --------
    master_seed: int = 34
    inst_seed_range: Tuple[int, int] = (0, 2**31 - 1)
    search_seed_range: Tuple[int, int] = (0, 2**31 - 1)

    # pass your ThetaSamplerConfig() here if you want non-default
    theta_sampler_cfg: Any = None


class ExperimentSampler:
    """
    Samples one trial with:
      - fixed dt, N, K, max_iters
      - sampled n,m,d_f,d_G
      - sampled H, T_outer (can be <N or >N)
      - sampled R_safe (safe radius for cost/prompt)
      - sampled score betas (log-uniform)
      - sampled instance seed + weight-search seed
    """

    def __init__(self, cfg: ExperimentSamplerConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.master_seed)

    def _sample_betas(self) -> Tuple[float, float, float, float]:
        c = self.cfg
        (m0, m1), (t0, t1), (u0, u1), (du0, du1) = c.beta_log10_ranges

        betas = [
            _randfloat_loguniform(self.rng, m0, m1),
            _randfloat_loguniform(self.rng, t0, t1),
            _randfloat_loguniform(self.rng, u0, u1),
            _randfloat_loguniform(self.rng, du0, du1),
        ]

        # Optional sparsification: randomly zero out some betas
        if c.beta_zero_prob > 0.0:
            for i in range(4):
                if self.rng.uniform() < c.beta_zero_prob:
                    betas[i] = 0.0

        # Ensure not all zero
        if sum(betas) <= 0.0:
            betas[0] = 1.0

        return float(betas[0]), float(betas[1]), float(betas[2]), float(betas[3])

    def sample_trial(self) -> Dict[str, Any]:
        c = self.cfg

        # ---- sample system sizes ----
        n = _randint_inclusive(self.rng, *c.n_range)
        # m_lo = max(1, n - 2)  # clamp so it's valid when n=1,2
        # m_hi = n  # highest is n
        # m = _randint_inclusive(self.rng, m_lo, m_hi)
        m = _randint_inclusive(self.rng, *c.m_range)
        d_f = _randint_inclusive(self.rng, *c.d_f_range)
        d_G = _randint_inclusive(self.rng, *c.d_G_range)

        # ---- sample max control input ----
        u_max_lo, u_max_hi = c.u_max_range
        u_max = _randfloat_uniform(self.rng, u_max_lo, u_max_hi)
        mu_lo, mu_hi = c.mu_range
        mu = _randfloat_uniform(self.rng, mu_lo, mu_hi)
        # mu = u_max

        # ---- fixed dt, N ----
        dt = float(c.dt)
        N = int(c.N)

        # ---- sample H (MPC horizon) ----
        H = _randint_inclusive(self.rng, *c.H_range)
        H = max(1, min(H, N))  # clamp to [1, N]

        # ---- sample T_outer (closed-loop length) ----
        slack_lo, slack_hi = c.T_outer_slack_frac_range
        if slack_lo <= 0 or slack_hi <= 0:
            raise ValueError("T_outer_slack_frac_range must be positive.")
        T_lo = max(1, int(np.ceil(slack_lo * N)))
        T_hi = max(T_lo, int(np.floor(slack_hi * N)))
        T_outer = _randint_inclusive(self.rng, T_lo, T_hi)

        # ---- sample R_safe ----
        R_lo, R_hi = c.R_safe
        R_safe = _randfloat_uniform(self.rng, R_lo, R_hi)

        # ---- sample seeds ----
        inst_seed = _randint_inclusive(self.rng, *c.inst_seed_range)
        search_seed = _randint_inclusive(self.rng, *c.search_seed_range)
        rng_search = np.random.default_rng(search_seed)

        # ---- build instance ----
        sys_cfg = PolyThetaConfig(n=n, m=m, d_f=d_f, d_G=d_G, dt=dt, N=N, r=R_safe, u_max=u_max, mu=mu, prob_mode=c.prob_mode)
        ts_cfg = c.theta_sampler_cfg if c.theta_sampler_cfg is not None else ThetaSamplerConfig()
        sampler = ThetaSampler(sys_cfg, ts_cfg, seed=inst_seed)
        inst = sampler.sample_instance()

        # # ---- sample score betas ----
        beta_margin, beta_time, beta_u, beta_du = self._sample_betas()

        return {
            "inst": inst,
            "rng_search": rng_search,
            "params": {
                # system
                "n": n, "m": m, "d_f": d_f, "d_G": d_G,
                "dt": dt, "N": N,
                "instance_seed": int(inst_seed),

                # horizons
                "H": int(H),
                "T_outer": int(T_outer),

                # Safe Radius
                "R_safe": R_safe,
                "rho": float(inst["meta"]["rho"]),

                # search
                "K": int(c.K),
                "max_iters": int(c.max_iters),
                "search_seed": int(search_seed),

                # score betas
                "beta_margin": float(beta_margin),
                "beta_time": float(beta_time),
                "beta_u": float(beta_u),
                "beta_du": float(beta_du),
            },
        }
