"""
Poly generator from a *single continuous parameter vector* Θ
=============================================================

This module builds one **solvable** polynomial reachability instance from
Θ = {C, D, x0, u_max, mu, rho, seed} — no alpha/phi, no backbones.

Dynamics (discrete via Euler or RK4):
    x_{k+1} = x_k + dt * [ f(x_k) + G(x_k) u_k ]

Where:
  - f(x) and each column of G(x) are polynomials up to degrees d_f and d_G.
  - We *always* set the target by rollout using a bounded witness control with
    ||u||_∞ = mu ≤ u_max → solvable by construction.
  - A small, continuous **projection & scale** keeps the step map contractive
    on a state box so trajectories don't blow up.

Θ contents
----------
- C: dict[int -> np.ndarray]
    Polynomial coefficients for drift f.
    For degree k, C[k] has shape (M_k, n) where M_k = #monomials of degree k in n vars.
    k=0 → constant term f0; k=1 → linear term encoded in C[1] (C[1].T is A).
- D: dict[int -> np.ndarray]
    Polynomial coefficients for input map G.
    For degree k, D[k] has shape (M_k, n, m). k=0 → constant term G0.
- x0: (n,) initial state
- u_max: float > 0  (input box bound)
- mu: float in (0, u_max]  (witness effort; we scale witness to have ||u||_∞ = mu)
- rho: float > 0  (goal tolerance radius)
- seed: int (reproducibility for the witness controls)

Config (fixed for a study): n, m, d_f, d_G, N, dt, r, alpha_cap, eps_margin.
- r: state box half-width where you want trajectories to live.
- alpha_cap ∈ (0,1): linear contraction cap after stabilization (default 0.95).
- eps_margin: small slack so total Jacobian norm < 1 by a margin (default 0.02).

NOTE: If C or D omit a degree, it's treated as zeros for that degree.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np

# ------------------------------
# Utilities: monomial bookkeeping
# ------------------------------

def _stars_bars(n: int, k: int):
    if k == 0:
        return [tuple([0]*n)]
    out = []
    def rec(prefix, rem, d):
        if d == 1:
            out.append(tuple(prefix + [rem]))
            return
        for v in range(rem+1):
            rec(prefix+[v], rem-v, d-1)
    rec([], k, n)
    return out


def build_exps(n: int, dmax: int) -> Dict[int, np.ndarray]:
    return {k: np.array(_stars_bars(n, k), dtype=int) for k in range(dmax+1)}


def eval_monomials(x: np.ndarray, exps: np.ndarray) -> np.ndarray:
    # For each exponent vector alpha (row), compute x^alpha
    return np.prod(np.power(x[None, :], exps), axis=1)

def _spectral_radius(M: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(M))))

def norm2(M: np.ndarray) -> float:
    # spectral/operator 2-norm (largest singular value)
    return float(np.linalg.norm(M, 2))

def scale_A_to_cap(
    A: np.ndarray,
    dt: float,
    cap: float,
    metric: str = "rho",          # "rho" or "norm2"
    s_min: float = 0.0,
    s_max: float = 10.0,
    grid: int = 101,
    prefer_close_to_1: bool = False,
) -> float:
    """
    Find s in [s_min, s_max] such that metric(I + dt*s*A) <= cap if possible.

    If prefer_close_to_1:
        among feasible s, pick the one closest to 1 (least distortion).
    else:
        pick the feasible s with smallest metric (most contraction).
    If infeasible, return s that minimizes the metric over the interval.
    """
    n = A.shape[0]
    I = np.eye(n)

    if metric == "rho":
        f = lambda s: _spectral_radius(I + dt * (s * A))
    elif metric == "norm2":
        f = lambda s: norm2(I + dt * (s * A))
    else:
        raise ValueError("metric must be 'rho' or 'norm2'")

    ss = np.linspace(s_min, s_max, grid)
    vals = np.array([f(float(s)) for s in ss], dtype=float)

    feasible = np.where(vals <= cap)[0]
    if feasible.size > 0:
        if prefer_close_to_1:
            j = feasible[np.argmin(np.abs(ss[feasible] - 1.0))]
        else:
            j = feasible[np.argmin(vals[feasible])]
        return float(ss[j])

    # no feasible s found: return best effort (min metric)
    j = int(np.argmin(vals))
    return float(ss[j])

def scale_A_by_rho(A, dt, rho_target, s_max=10.0, iters=50):
    n = A.shape[0]
    I = np.eye(n)

    # If already <= target, keep s=1
    Ad = I + dt*(1.0*A)
    # if _spectral_radius(Ad) <= rho_target:
    if norm2(Ad) <= rho_target:
        return 1.0

    # Search s in [1, s_max] to reduce rho
    lo, hi = 1.0, s_max
    Ad_max = I + dt*(s_max*A)
    # if _spectral_radius(Ad_max) > rho_target:
    if norm2(Ad_max) > rho_target:
        # even very strong damping didn't reach target; give up
        return hi

    for _ in range(iters):
        mid = 0.5*(lo+hi)
        Ad_mid = I + dt*(mid*A)
        # if _spectral_radius(Ad_mid) > rho_target:
        if norm2(Ad_mid) > rho_target:

            lo = mid
        else:
            hi = mid
    return hi

def sample_near_boundary_goal(rng, x0, x_end_w, r_box, band=(0.8, 1.2), eps=1e-12):
    n = x0.size
    d_w = float(np.linalg.norm(x_end_w - x0) + eps)
    v = rng.normal(size=n)
    v = v / (np.linalg.norm(v) + eps)

    rad = rng.uniform(band[0] * d_w, band[1] * d_w)
    x_goal = x0 + rad * v

    # keep within state box
    x_goal = np.clip(x_goal, -r_box, r_box)
    return x_goal

def eval_monomials_and_grads(x: np.ndarray, exps: np.ndarray, eps: float = 1e-12):
    """
    x: (n,)
    exps: (M, n) integer exponents for monomials of total degree k
    returns:
      mon: (M,)
      dmon_dx: (M, n) where dmon_dx[j,i] = ∂ mon_j / ∂ x_i
    """
    x = np.asarray(x, dtype=float)
    exps = np.asarray(exps, dtype=int)
    M, n = exps.shape

    # mon_j = Π_i x_i^{e_{j,i}}
    mon = np.ones(M, dtype=float)
    for i in range(n):
        ei = exps[:, i]
        if np.any(ei):
            mon *= np.power(x[i], ei, dtype=float)

    dmon_dx = np.zeros((M, n), dtype=float)

    for i in range(n):
        ei = exps[:, i]
        nz = ei > 0
        if not np.any(nz):
            continue

        xi = x[i]
        if abs(xi) > eps:
            dmon_dx[nz, i] = mon[nz] * (ei[nz] / xi)
        else:
            # fallback: compute derivative without division at xi ~ 0
            # ∂/∂xi Π x_l^{e_l} = e_i * x_i^{e_i-1} * Π_{l≠i} x_l^{e_l}
            tmp = np.ones(np.sum(nz), dtype=float)
            idx = np.where(nz)[0]
            for l in range(n):
                el = exps[idx, l]
                if l == i:
                    tmp *= np.power(x[l], el - 1, dtype=float)  # el>=1 here
                else:
                    tmp *= np.power(x[l], el, dtype=float)
            dmon_dx[idx, i] = ei[idx] * tmp

    return mon, dmon_dx

def jacobian_f_nl(x, drift, exps_by_k):
    """
    returns J_f_nl(x) (n,n)
    """
    n = x.size
    J = np.zeros((n, n), dtype=float)

    for k, Ck in drift["C"].items():  # k>=2
        if Ck.size == 0:
            continue
        exps = exps_by_k[int(k)]              # (M_k, n)
        _, dmon_dx = eval_monomials_and_grads(x, exps)   # (M_k,n)
        # (n,M_k) @ (M_k,n)?? careful:
        # dmon_dx is (M_k,n). We want (n,M_k) to left-multiply Ck (M_k,n)
        # J += (dmon_dx.T @ Ck) gives (n,M_k)@(M_k,n)=(n,n)
        J += dmon_dx.T @ Ck

    return J

def jacobian_Gu(x, u, inp, exps_by_k):
    """
    returns J_{G(x)u}(x) (n,n)
    """
    x = np.asarray(x, float)
    u = np.asarray(u, float)
    n = x.size
    J = np.zeros((n, n), dtype=float)

    for k, Dk in inp["D"].items():  # k>=1
        if Dk.size == 0:
            continue
        exps = exps_by_k[int(k)]                 # (M_k,n)
        _, dmon_dx = eval_monomials_and_grads(x, exps)   # (M_k,n)

        # Dk: (M_k, n, m)
        # Contract with u: (M_k, n)
        Du = np.tensordot(Dk, u, axes=(2, 0))    # (M_k, n)

        # Now contribution to Jacobian: sum_j ∂mon_j/∂x_i * (Du[j,:])
        # dmon_dx.T is (n,M_k) times Du (M_k,n) => (n,n)
        J += dmon_dx.T @ Du

    return J

def estimate_L_nl(
    drift, inp, exps_by_k, r, u_max,
    num=256, q=0.99, seed=0
):
    rng = np.random.default_rng(seed)
    n = drift["A"].shape[0]
    m = inp["G0"].shape[1]

    vals = np.empty(num, dtype=float)
    for i in range(num):
        x = rng.uniform(-r, r, size=(n,))
        # sample a "worst-ish" u in the box
        u = u_max * np.sign(rng.standard_normal(size=(m,)))
        u[u == 0] = u_max

        Jf = jacobian_f_nl(x, drift, exps_by_k)
        Jgu = jacobian_Gu(x, u, inp, exps_by_k)
        vals[i] = np.linalg.norm(Jf + Jgu, 2)

    return float(np.quantile(vals, q)), float(np.max(vals)), float(np.mean(vals))
# ------------------------------
# Config
# ------------------------------
@dataclass
class PolyThetaConfig:
    n: int = 3
    m: int = 1
    d_f: int = 2               # degree for f (>=0)
    d_G: int = 2               # degree for G (>=0)
    dt: float = 0.1
    N: int = 25
    r: float = 2.0             # state box half-width
    u_max:float = 3.0          # max control input
    mu:float = 1.0             # control effort allowed
    # alpha_cap: float = 0.95  # cap for linear contraction (discrete-time)
    eps_margin: float = 0.3   # margin so total Jacobian < 1 - eps
    integrator: str = "euler"  # or "rk4"
    # integrator: str = "rk4"  # or "rk4"
    # probabilities for goal modes: [witness_end, near_boundary, random_goal]
    prob_mode: tuple[float, float, float] = (0.3, 0.5, 0.2)

# ------------------------------
# Core generator
# ------------------------------
class PolyFromTheta:
    def __init__(self, cfg: PolyThetaConfig):
        self.cfg = cfg
        self.exps = build_exps(cfg.n, max(cfg.d_f, cfg.d_G))

    # ---- helpers to fill missing degrees with zeros ----
    def _ensure_C(self, C: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        n = self.cfg.n
        CC: Dict[int, np.ndarray] = {}
        for k in range(self.cfg.d_f + 1):
            Mk = self.exps[k].shape[0]
            if k in C:
                CC[k] = np.array(C[k], dtype=float)
                assert CC[k].shape == (Mk, n), f"C[{k}] must be shape ({Mk}, {n})"
            else:
                CC[k] = np.zeros((Mk, n))
        return CC

    def _ensure_D(self, D: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        n, m = self.cfg.n, self.cfg.m
        DD: Dict[int, np.ndarray] = {}
        for k in range(self.cfg.d_G + 1):
            Mk = self.exps[k].shape[0]
            if k in D:
                DD[k] = np.array(D[k], dtype=float)
                assert DD[k].shape == (Mk, n, m), f"D[{k}] must be shape ({Mk}, {n}, {m})"
            else:
                DD[k] = np.zeros((Mk, n, m))
        return DD

    # ---- build drift/input dicts and stabilize ----
    def _stabilize_and_build(self, C_in: Dict[int, np.ndarray], D_in: Dict[int, np.ndarray], u_max: float, seed: int):
        cfg = self.cfg
        C = self._ensure_C(C_in)
        D = self._ensure_D(D_in)
        n, m = cfg.n, cfg.m
        # Split constant, linear, higher
        f0 = C[0][0]                          # (n,)
        C1 = C[1]                             # (n, n?) actually (M1=n, n)
        assert C1.shape[0] == n
        A = C1.T                               # (n, n), since y += mon @ C1 = C1^T x
        C_high = {k: C[k] for k in range(2, cfg.d_f + 1)}

        G0 = D[0][0]                          # (n, m)
        D_high = {k: D[k] for k in range(1, cfg.d_G + 1)}

        # (1) Linear contraction cap
        rng = np.random.default_rng(seed)
        alpha_cap = float(rng.uniform(0.1, 0.3))
        # print("this is target rho")
        # print(alpha_cap)
        # alpha_cap = 0.75
        # Ad = np.eye(n) + cfg.dt * A
        # rho = _spectral_radius(Ad)
        # sA = min(1.0, alpha_cap / rho)
        # print("this is sA")
        # print(sA)
        # if sA < 1.0:
        #     A *= sA
        #     # Scale higher drift degrees a bit more (helps Lipschitz on box)
        #     for k in C_high:
        #         C_high[k] = (sA ** k) * C_high[k]
        #     # Update Ad and rho after scaling
        #     Ad = np.eye(n) + cfg.dt * A
        #     rho = _spectral_radius(Ad)
        #     print("this is new rho")
        # print(rho)
        # sA = scale_A_by_rho(A, cfg.dt, alpha_cap)
        sA = scale_A_to_cap(A, dt=cfg.dt, cap=alpha_cap, metric="rho", s_min=0.0, s_max=10.0)
        A *= sA
        # Update Ad and rho after scaling
        Ad = np.eye(n) + cfg.dt * A
        # rho = _spectral_radius(Ad)
        rho = norm2(Ad)

        # (2) Bound nonlinear Jacobian on box and downscale if needed
        L_nl = 0.0
        # Drift part: sum over degrees k>=2 of k * r^{k-1} * sum_j ||C_k[j,:]||_2
        for k, Ck in C_high.items():
            if Ck.size:
                # L_nl += k * (cfg.r ** (k - 1)) * np.sum(np.linalg.norm(Ck, axis=1))
                L_nl += k * (cfg.r ** (k - 1)) * np.linalg.norm(Ck, "fro")
        # Input part (times u_max): degrees k>=1 of k * r^{k-1} * sum_j ||D_k[j,:,:]||_F
        for k, Dk in D_high.items():
            if Dk.size:
                mats = Dk.reshape(Dk.shape[0], -1)
                # L_nl += u_max * k * (cfg.r ** (k - 1)) * np.sum(np.linalg.norm(mats, axis=1))
                L_nl += u_max * k * (cfg.r ** (k - 1)) * np.linalg.norm(mats, "fro")

        # L_est, L_max, L_mean = estimate_L_nl(
        #     drift={"f0": f0, "A": A, "C": C_high},
        #     inp={"G0": G0, "D": D_high},
        #     exps_by_k=self.exps, r=cfg.r, u_max=u_max, num=256, q=0.99, seed=seed)

        # print("estimated")
        # print(L_est)
        # print(L_nl)

        # Ensure total step Jacobian < 1 - eps
        # Conservative bound: ||∂T/∂x|| ≤ ||Ad||_2 + dt * L_nl
        # ||Ad||_2 <= rho for symmetric Ad; use spectral radius as proxy.
        # if cfg.dt * L_nl + rho > (1.0 - cfg.eps_margin):

        if cfg.dt * L_nl + rho > (1.0 + cfg.eps_margin):
            # scale factor for nonlinear terms
            denom = L_nl + 1e-12
            s_nl = max(0.0, (1.0 + cfg.eps_margin - rho) / denom)
            s_nl = min(1.0, s_nl)
            for k in list(C_high.keys()):
                C_high[k] = s_nl * C_high[k]
            for k in list(D_high.keys()):
                D_high[k] = s_nl * D_high[k]

        # L_target = 5  # for dt=0.1 start here
        # print("L_nl this is nonlinear")
        # print(L_est)
        # denom = L_est + 1e-12
        # if L_est > L_target:
        #     print("it scales nonlinear part")
        #     s_nl = L_target / denom
        #     s_nl = float(np.clip(s_nl, 0.0, 1.0))
        #     # Always reduce nonlinear drift
        #     for k in C_high:
        #         C_high[k] *= s_nl
        #     # Optional: reduce state-dependent control nonlinearity too
        #     for k in D_high:
        #         D_high[k] *= s_nl

        drift = {"f0": f0, "A": A, "C": C_high}
        inp   = {"G0": G0, "D": D_high}
        return drift, inp

    # ---- evaluate f and G ----
    def eval_f(self, x: np.ndarray, drift: Dict) -> np.ndarray:
        y = drift['f0'] + drift['A'] @ x
        for k, Ck in drift['C'].items():
            if Ck.size == 0: continue
            mon = eval_monomials(x, self.exps[k])    # (M_k,)
            y = y + mon @ Ck                         # (n,)
        return y

    def eval_G(self, x: np.ndarray, inp: Dict) -> np.ndarray:
        G = inp['G0'].copy()
        for k, Dk in inp['D'].items():
            if Dk.size == 0: continue
            mon = eval_monomials(x, self.exps[k])    # (M_k,)
            G = G + np.tensordot(mon, Dk, axes=(0, 0))  # (n,m)
        return G

    # ---- integration ----
    def step(self, x: np.ndarray, u: np.ndarray, drift: Dict, inp: Dict) -> np.ndarray:
        dt = self.cfg.dt
        if self.cfg.integrator == 'euler':
            return x + dt * (self.eval_f(x, drift) + self.eval_G(x, inp) @ u)
        elif self.cfg.integrator == 'rk4':
            f = lambda z, v: self.eval_f(z, drift) + self.eval_G(z, inp) @ v
            k1 = f(x, u)
            k2 = f(x + 0.5*dt*k1, u)
            k3 = f(x + 0.5*dt*k2, u)
            k4 = f(x + dt*k3,  u)
            return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        else:
            raise ValueError('integrator must be euler or rk4')

    def rollout(self, drift: Dict, inp: Dict, x0: np.ndarray, U_seq: np.ndarray) -> np.ndarray:
        n, N = self.cfg.n, self.cfg.N
        X = np.zeros((N+1, n)); X[0] = x0; x = x0.copy()
        for k in range(N):
            x = self.step(x, U_seq[k], drift, inp)
            X[k+1] = x
        return X

    # ---- witness generation ----
    @staticmethod
    def make_witness(seed: int, N: int, m: int, u_mu: float, u_max: float) -> np.ndarray:
        rng = np.random.default_rng(seed)
        U = rng.uniform(-1.0, 1.0, size=(N, m))
        s = np.max(np.abs(U))
        if s < 1e-12:
            U[0, 0] = 1.0; s = 1.0
        U = (u_mu / s) * U
        # mu ≤ u_max by contract, so this is in the box; clip just in case
        U = np.clip(U, -u_max, u_max)
        return U

    # ---- main: Θ -> instance ----
    def generate(self, Theta: Dict) -> Dict:
        cfg = self.cfg
        C = Theta['C']
        D = Theta['D']
        x0 = np.array(Theta['x0'], dtype=float).reshape(cfg.n)
        u_max = float(Theta['u_max'])
        u_mu = float(Theta['u_mu'])
        rho = float(Theta['rho'])
        seed = int(Theta.get('seed', 0))
        assert 0 < u_mu <= u_max

        drift, inp = self._stabilize_and_build(C, D, u_max, seed)
        # u_mu = u_max
        U = self.make_witness(seed, cfg.N, cfg.m, u_mu, u_max)
        X = self.rollout(drift, inp, x0, U)
        # x_end is by sampling from witness control
        x_end_w = X[-1]
        # x_end = np.array(Theta['x_end'], dtype=float).reshape(cfg.n)
        rng = np.random.default_rng(seed)
        p = np.asarray(cfg.prob_mode, dtype=float)
        p = p / p.sum()
        mode = rng.choice(["witness", "near_boundary", "random"], p=p)
        if mode == "witness":
            x_end = x_end_w
            # print("this is the end point")
            # print(x_end_w)
        elif mode == "near_boundary":
            x_end = sample_near_boundary_goal(
                rng=rng, x0=x0, x_end_w=x_end_w, r_box=cfg.r, band=(0.8, 1.2)
            )
            # print("this is the end point")
            # print(x_end_w)

        else:  # "random"
            x_end = np.array(Theta['x_end_rand'], dtype=float).reshape(cfg.n)
            # print("this is the random end point")
            # print(x_end_w)

        # ---- make a step function f(x,u) for this instance ----
        def step_fun(x, u, drift=drift, inp=inp, self=self):
            return self.step(x, u, drift, inp)

        theta_inst = {
            'cfg': cfg,
            'meta': {
                'n': cfg.n, 'm': cfg.m, 'N': cfg.N, 'dt': cfg.dt, 'integrator': cfg.integrator,
                'r': cfg.r, 'u_max': u_max, 'rho': rho,
                # 'alpha_cap': alpha_cap,
                'eps_margin': cfg.eps_margin,
            },
            'coeffs': {'C': self._ensure_C(C), 'D': self._ensure_D(D)},
            'drift': drift,
            'input': inp,
            'init_goal': {'x0': x0, 'x_end': x_end, 'goal_radius': rho},
            'witness_controls': U,
            'step': step_fun,     # <--- here
        }
        return theta_inst

# ------------------------------
# Tiny smoke test / example
# ------------------------------
if __name__ == "__main__":
    # Example: n=2, m=1, quadratic f,G
    cfg = PolyThetaConfig(n=2, m=1, d_f=2, d_G=2, dt=0.1, N=30)
    gen = PolyFromTheta(cfg)

    # Build empty coeffs then fill a few entries
    C = {k: np.zeros((gen.exps[k].shape[0], cfg.n)) for k in range(cfg.d_f+1)}
    D = {k: np.zeros((gen.exps[k].shape[0], cfg.n, cfg.m)) for k in range(cfg.d_G+1)}
    # Constant drift and linear term (encode linear in C[1], so A = C[1].T)
    C[0][0] = np.array([0.0, 0.0])                 # f0
    C[1][:] = np.array([[ -0.3,  0.0],             # x1 term → contributes to both states
                        [  0.0, -0.4]])            # x2 term
    # Quadratic drift: small coupling on x1^2 and x1*x2
    C[2][:] = 0.02 * np.array([
        [ 1.0, -0.5],   # x1^2
        [ 0.0,  0.0],   # x1 x2
        [ 0.0,  0.0],   # x2^2
    ])[:C[2].shape[0]]  # trim if degree-2 has more monomials

    # Input map: constant G0 plus light state-dependence
    D[0][0,:,0] = np.array([1.0, 0.2])             # G0 (n×m)
    if cfg.d_G >= 1:
        D[1][:,:,0] = 0.05 * np.array([
            [ 0.1, 0.0],   # x1 term
            [ 0.0, 0.1],   # x2 term
        ])[:D[1].shape[0], :]

    Theta = {
        'C': C,
        'D': D,
        'x0': np.array([0.5, -0.3]),
        'u_max': 1.0,
        'mu': 0.8,
        'rho': 0.05,
        'seed': 42,
    }

    inst = gen.generate(Theta)
    print("Final state (x_end):", inst['init_goal']['x_end'])
    print("Witness max |u|:", np.max(np.abs(inst['witness_controls'])))
