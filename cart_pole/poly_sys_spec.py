import numpy as np
from poly_theta_generator import PolyThetaConfig, PolyFromTheta
from poly_theta_sampler import ThetaSampler, ThetaSamplerConfig
from itertools import combinations_with_replacement


# --------- monomial exponent utilities (total-degree basis) ---------
def _n_monomials(n, k):
    # C(n+k-1, k)
    from math import comb
    return comb(n + k - 1, k)

def _exponents_total_degree(n, k):
    """
    Return all multi-indices alpha in N^n with |alpha| = k (shape (M_k, n)).
    Order matches 'stars and bars' lexicographic by construction.
    """
    if k == 0:
        return np.zeros((1, n), dtype=int)
    exps = []
    # positions of bars among n-1 gaps over k stars
    for bars in combinations_with_replacement(range(n), k):
        alpha = np.zeros(n, dtype=int)
        for b in bars:
            alpha[b] += 1
        exps.append(alpha)
    return np.asarray(exps, dtype=int)

def _build_exps_from_coeffs(inst):
    """
    Build {k: exps_k} for all degrees present in coeffs['C'] and coeffs['D'].
    Verifies consistency with tensor shapes.
    """
    n = inst["meta"]["n"]
    Cdict = inst["coeffs"]["C"]
    Ddict = inst["coeffs"]["D"]
    ks = set(Cdict.keys()) | set(Ddict.keys())
    exps = {}
    for k in ks:
        Mk = _n_monomials(n, k)
        E = _exponents_total_degree(n, k)
        assert E.shape[0] == Mk, f"Mismatch Mk for k={k}"
        # sanity-check against stored tensor shapes (if present)
        if k in Cdict:
            assert Cdict[k].shape[0] == Mk, f"C[{k}] rows != M_k"
        if k in Ddict:
            assert Ddict[k].shape[0] == Mk, f"D[{k}] rows != M_k"
        exps[k] = E
    return exps

def _eval_monomials(x, exps_k):
    # returns vector [x^alpha] over rows alpha in exps_k
    if exps_k.size == 0:
        return np.zeros(0, dtype=float)
    return np.prod(np.power(x[None, :], exps_k), axis=1)

# --------- dynamics evaluation from inst ---------
def eval_f_poly(x, inst, exps=None):
    """
    f(x) = f0 + A x + sum_{k>=2} mon_k(x)^T * C[k]
    Uses 'drift' (stabilized) terms from inst for simulation.
    """
    drift = inst["drift"]
    y = drift["f0"] + drift["A"] @ x
    C_high = drift["C"]  # dict k>=2 -> (M_k, n)
    if exps is None:
        exps = _build_exps_from_coeffs(inst)
    for k, Ck in C_high.items():
        if Ck.size:
            mon = _eval_monomials(x, exps[k])
            y = y + mon @ Ck
    return y

def eval_G_poly(x, inst, exps=None):
    """
    G(x) = G0 + sum_{k>=1} sum_j mon_{k,j}(x) * D[k][j]
    Uses 'input' (stabilized) terms from inst for simulation.
    """
    inp = inst["input"]
    G = inp["G0"].copy()
    D_high = inp["D"]  # dict k>=1 -> (M_k, n, m)
    if exps is None:
        exps = _build_exps_from_coeffs(inst)
    for k, Dk in D_high.items():
        if Dk.size:
            mon = _eval_monomials(x, exps[k])        # (M_k,)
            G = G + np.tensordot(mon, Dk, axes=(0, 0))  # (n,m)
    return G

# --------- single-step integrators ---------
def step_euler(x, u, inst, exps=None):
    dt = inst["meta"]["dt"]
    f = eval_f_poly(x, inst, exps)
    G = eval_G_poly(x, inst, exps)
    return x + dt * (f + G @ u)

def step_rk4(x, u, inst, exps=None):
    dt = inst["meta"]["dt"]
    if exps is None:
        exps = _build_exps_from_coeffs(inst)
    def dyn(z):
        return eval_f_poly(z, inst, exps) + eval_G_poly(z, inst, exps) @ u
    k1 = dyn(x)
    k2 = dyn(x + 0.5 * dt * k1)
    k3 = dyn(x + 0.5 * dt * k2)
    k4 = dyn(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# --------- rollout ---------
def rollout_poly(inst, U_seq, integrator="euler"):
    """
    U_seq: (N, m); returns X: (N+1, n)
    Uses 'drift'/'input' from inst and derives monomials from inst['coeffs'].
    """
    N = inst["meta"]["N"]
    n = inst["meta"]["n"]
    X = np.zeros((N + 1, n), dtype=float)
    X[0] = inst["init_goal"]["x0"]
    exps = _build_exps_from_coeffs(inst)
    step = step_euler if integrator.lower() == "euler" else step_rk4
    for k in range(N):
        X[k + 1] = step(X[k], U_seq[k], inst, exps)
    return X

cfg = PolyThetaConfig(n=3, m=1, d_f=2, d_G=2, dt=0.1, N=25)
sampler = ThetaSampler(cfg, ThetaSamplerConfig(), seed=23)

inst  = sampler.sample_instance()  # directly solvable instance
U = inst["witness_controls"]
# print("Final state (x_end):", inst["init_goal"]["x_end"])
# print("Max |u| in witness:", np.max(np.abs(inst["witness_controls"])))
# print(inst['drift'])

x = inst['init_goal']['x0']
f_val = eval_f_poly(x, inst)                  # drift at x
G_val = eval_G_poly(x, inst)                  # input map at x
x1    = step_euler(x, U[0], inst)             # one Euler step
# Full rollout with the instance's witness controls
U = inst["witness_controls"]              # shape (N, m)
X = rollout_poly(inst, U, integrator="euler")  # or "euler"
x_end_est = X[-1]

gen = PolyFromTheta(cfg)
x0 =  inst['init_goal']['x0']
drift, inp = inst["drift"], inst["input"]
U = inst["witness_controls"]
X = gen.rollout(drift, inp, x0, U)
x_end = X[-1]
print(x_end)

f = inst['step']      # f(x, u) -> x_next
x = x0.copy()
for k in range(25):
    x = f(x, U[k])
print(x)

print("Computed x_end:", x_end_est)
print("Instance x_end:", inst["init_goal"]["x_end"])
print("Diff:", x_end_est - inst["init_goal"]["x_end"])

