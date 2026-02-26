"""
Sampler for Θ = {C, D, x0, u_max, mu, rho, seed}
================================================

This module samples a *continuous* parameter vector Θ for the polynomial,
control‑affine generator provided in `poly_theta_generator.py`.

Usage (script):
    python poly_theta_sampler.py --num 5 --seed 0

Programmatic:
    from poly_theta_generator import PolyThetaConfig, PolyFromTheta
    from poly_theta_sampler import ThetaSampler, ThetaSamplerConfig

    cfg = PolyThetaConfig(n=3, m=1, d_f=2, d_G=2, dt=0.1, N=25)
    scfg = ThetaSamplerConfig()
    sampler = ThetaSampler(cfg, scfg, seed=0)

    Theta = sampler.sample_theta()          # just Θ
    inst  = sampler.sample_instance()       # Θ → full solvable instance (via generator)

Design choices
--------------
- We avoid α/ϕ/backbones entirely. Θ holds raw polynomial coefficients.
- We keep coefficients small and degree‑scaled. A later *projection & scale*
  inside the generator ensures the one‑step map is contractive on a box.
- We encourage stability by sampling the linear drift A around a negative diagonal.
- Sparsity knobs let you control how many monomials are active.

All randomness is controlled by two seeds:
  - `Theta.seed`  : used by the generator only to create the witness control; it
                    is drawn from the sampler RNG so each Θ has a unique witness.
  - `ThetaSampler(seed=...)` : controls how Θ itself is sampled.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np

from poly_theta_generator import PolyThetaConfig, PolyFromTheta

# ------------------------------
# Helper: monomial counts
# ------------------------------

def n_monomials(n: int, k: int) -> int:
    """Number of monomials of total degree k in n variables = C(n+k-1, k)."""
    from math import comb
    return comb(n + k - 1, k)

# ------------------------------
# Sampler configuration
# ------------------------------
@dataclass
class ThetaSamplerConfig:
    # Linear drift A sampling
    diag_range: tuple = (0.15, 1.2)   # sample negative diagonals in this range
    offdiag_std_frac: float = 0.15    # off-diag std as a fraction of mean |diag|

    # Constant terms
    f0_std: float = 0.05              # std for constant drift f0
    G0_std: float = 0.2               # std for constant input map G0

    # Higher-degree coefficient scales (base std at degree 1), decays with degree
    drift_base_std: float = 0.08      # base std for drift polynomial coeffs
    input_base_std: float = 0.06      # base std for input polynomial coeffs
    deg_decay: float = 0.6            # multiply std by (deg_decay)^(k-1) for degree k

    # Sparsity (probability of *keeping* a monomial coefficient block)
    keep_prob_deg2: float = 1.0
    keep_prob_deg3p: float = 1.0

    # State and control sampling
    x0_box_frac: float = 0.95          # sample x0 uniformly in [-r*x0_box_frac, r*x0_box_frac]^n
    x_end_box_frac: float = 0.95       # sample x_end uniformly in [-r*x0_box_frac, r*x0_box_frac]^n
    # u_max_range: tuple = (0.1, 5.0)
    mu_min_frac: float = 0.6          # μ ∈ [mu_min_frac * u_max, u_max]
    rho_frac: float = 0.01            # goal radius as fraction of r

# ------------------------------
# Main Sampler
# ------------------------------
class ThetaSampler:
    def __init__(self, cfg: PolyThetaConfig, scfg: ThetaSamplerConfig, seed: Optional[int] = None):
        self.cfg = cfg
        self.scfg = scfg
        self.rng = np.random.default_rng(seed)

    def _sample_linear_A(self) -> np.ndarray:
        """Sample a reasonably stable A: negative diagonal + small off-diagonal noise."""
        n = self.cfg.n
        lo, hi = self.scfg.diag_range
        diag = -self.rng.uniform(lo, hi, size=n)          # negative diagonals
        A = np.diag(diag)
        std = self.scfg.offdiag_std_frac * np.mean(np.abs(diag))
        if n > 1 and std > 0:
            noise = self.rng.normal(0.0, std, size=(n, n))
            np.fill_diagonal(noise, 0.0)
            A = A + noise
        return A

    def _mask_keep(self, shape, keep_prob):
        return (self.rng.uniform(size=shape) < keep_prob).astype(float)

    def _sample_C(self) -> Dict[int, np.ndarray]:
        n = self.cfg.n
        C: Dict[int, np.ndarray] = {}
        # degree 0: constant drift
        C[0] = self.rng.normal(0.0, self.scfg.f0_std, size=(1, n))
        # degree 1: linear drift (C[1].T is A)
        A = self._sample_linear_A()
        C[1] = A.T.copy()
        # degree >=2
        for k in range(2, self.cfg.d_f + 1):
            Mk = n_monomials(n, k)
            std = self.scfg.drift_base_std * (self.scfg.deg_decay ** (k - 1))
            block = self.rng.normal(0.0, std, size=(Mk, n))
            keep = self._mask_keep((Mk, 1), self.scfg.keep_prob_deg2 if k == 2 else self.scfg.keep_prob_deg3p)
            C[k] = block * keep  # broadcast keeps/deletes entire monomial blocks
        return C

    def _sample_D(self) -> Dict[int, np.ndarray]:
        n, m = self.cfg.n, self.cfg.m
        D: Dict[int, np.ndarray] = {}
        # degree 0: constant input map G0
        D[0] = self.rng.normal(0.0, self.scfg.G0_std, size=(1, n, m))
        # degree >=1
        for k in range(1, self.cfg.d_G + 1):
            Mk = n_monomials(n, k)
            std = self.scfg.input_base_std * (self.scfg.deg_decay ** (k - 1))
            block = self.rng.normal(0.0, std, size=(Mk, n, m))
            keep = self._mask_keep((Mk, 1, 1), self.scfg.keep_prob_deg2 if k == 2 else self.scfg.keep_prob_deg3p)
            D[k] = block * keep
        return D

    def sample_theta(self) -> Dict:
        cfg, scfg, rng = self.cfg, self.scfg, self.rng
        C = self._sample_C()
        D = self._sample_D()
        # x0 in inner box
        x0 = rng.uniform(-cfg.r * scfg.x0_box_frac, cfg.r * scfg.x0_box_frac, size=cfg.n)
        # x_end in inner box
        x_end_rand = rng.uniform(-cfg.r * scfg.x_end_box_frac, cfg.r * scfg.x_end_box_frac, size=cfg.n)
        # bounds and effort
        u_max = cfg.u_max
        # mu = rng.uniform(scfg.mu_min_frac * u_max, u_max)
        u_mu = u_max * cfg.mu
        rho = scfg.rho_frac * cfg.r
        # seed for witness control (separate from sampler seed)
        seed = int(rng.integers(0, np.iinfo(np.int32).max))
        Theta = {"C": C, "D": D, "x0": x0, "x_end_rand": x_end_rand, "u_max": u_max, "u_mu": u_mu, "rho": rho, "seed": seed}
        return Theta

    def sample_instance(self) -> Dict:
        """Sample Θ and immediately generate the solvable instance via PolyFromTheta."""
        Theta = self.sample_theta()
        gen = PolyFromTheta(self.cfg)
        return gen.generate(Theta)

# ------------------------------
# CLI
# ------------------------------
if __name__ == "__main__":
    import argparse, json, os
    p = argparse.ArgumentParser(description="Sample Θ vectors and/or full solvable instances.")
    p.add_argument('--num', type=int, default=5, help='number of samples')
    p.add_argument('--seed', type=int, default=0, help='sampler RNG seed')
    p.add_argument('--instances', action='store_true', help='emit full instances (not just Θ)')
    p.add_argument('--out', type=str, default='', help='output JSONL path (auto if empty)')
    # allow overriding core config quickly
    p.add_argument('--n', type=int, default=3)
    p.add_argument('--m', type=int, default=1)
    p.add_argument('--df', type=int, default=2)
    p.add_argument('--dG', type=int, default=2)
    p.add_argument('--dt', type=float, default=0.1)
    p.add_argument('--N', type=int, default=25)
    args = p.parse_args()

    cfg = PolyThetaConfig(n=args.n, m=args.m, d_f=args.df, d_G=args.dG, dt=args.dt, N=args.N)
    scfg = ThetaSamplerConfig()
    sampler = ThetaSampler(cfg, scfg, seed=args.seed)

    # items = []
    # for _ in range(args.num):
    #     if args.instances:
    #         items.append(sampler.sample_instance())
    #     else:
    #         items.append(sampler.sample_theta())
    #
    # if not args.out:
    #     kind = 'instances' if args.instances else 'theta'
    #     args.out = f"{kind}_n{cfg.n}_m{cfg.m}_df{cfg.d_f}_dG{cfg.d_G}_N{cfg.N}_num{args.num}_seed{args.seed}.jsonl"
    #
    # with open(args.out, 'w') as f:
    #     for it in items:
    #         # JSON‑ify numpy
    #         def tolist(o):
    #             if isinstance(o, np.ndarray):
    #                 return o.tolist()
    #             if isinstance(o, dict):
    #                 return {k: tolist(v) for k, v in o.items()}
    #             if isinstance(o, (list, tuple)):
    #                 return [tolist(v) for v in o]
    #             return o
    #         f.write(json.dumps(tolist(it)) + "\n")
    # print(f"Saved {len(items)} {'instances' if args.instances else 'Θ vectors'} to {os.path.abspath(args.out)}")
