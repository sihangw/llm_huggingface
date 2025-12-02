# from poly_theta_generator import PolyThetaConfig, PolyFromTheta
# import numpy as np
#
# # --- fixed study config ---
# cfg = PolyThetaConfig(n=3, m=1, d_f=2, d_G=2, dt=0.1, N=25)   # quadratic f and G
# gen = PolyFromTheta(cfg)
#
# # --- allocate coefficient tensors with correct shapes ---
# C = {k: np.zeros((gen.exps[k].shape[0], cfg.n)) for k in range(cfg.d_f + 1)}   # drift f
# D = {k: np.zeros((gen.exps[k].shape[0], cfg.n, cfg.m)) for k in range(cfg.d_G + 1)}  # input map G
#
# # ----------------------------
# # DRIFT: f(x) = f0 + A x + (quadratic terms)
# # ----------------------------
#
# # (a) Constant term f0 (n,)
# C[0][0] = np.array([0.0, 0.0, 0.0])  # no constant drift
#
# # (b) Linear term: choose a stable-ish A and set C[1] = A.T
# A = np.array([[-0.30,  0.05,  0.00],   # mild coupling but negative diagonal → stable-ish
#               [ 0.00, -0.25,  0.04],
#               [ 0.00,  0.00, -0.20]])
# C[1][:] = A.T
#
# # (c) Quadratic terms (degree-2). There are M2 = 6 monomials for n=3:
# #     order from gen.exps[2] is usually: (2,0,0),(1,1,0),(1,0,1),(0,2,0),(0,1,1),(0,0,2)
# ex2 = gen.exps[2]
#
# # helper to find the row index of a specific monomial exponent
# def idx_of(alpha):
#     return int(np.where((ex2 == np.array(alpha)).all(axis=1))[0][0])
#
# C[2][idx_of((2,0,0))] = np.array([ 0.02, -0.01, 0.00])  # x1^2 affects f1 (+) and f2 (-)
# C[2][idx_of((0,1,1))] = np.array([-0.005, 0.00, 0.015]) # x2*x3 gives small coupling into f3
#
# # ----------------------------
# # INPUT MAP: G(x) = G0 + (linear & quadratic terms)
# # ----------------------------
#
# # (a) Constant input matrix G0 (n×m). With m=1, it's a column vector.
# D[0][0,:,0] = np.array([1.0, 0.2, 0.0])  # input mainly pushes along x1, a bit on x2
#
# # (b) Linear state-dependence in G (degree-1): M1 = n rows, one per monomial x1,x2,x3
# # Row 0 (monomial x1) increases push on x1, row 1 on x2, row 2 on x3
# D[1][:,:,0] = np.array([[0.10, 0.00, 0.00],   # from x1
#                         [0.00, 0.08, 0.00],   # from x2
#                         [0.00, 0.00, 0.05]])
#
# # (c) Optional small quadratic input dependence (keep it tiny)
# D[2][idx_of((1,1,0)),:,0] = np.array([0.01, 0.00, 0.00])  # x1*x2 slightly boosts push on x1
#
# # ----------------------------
# # Wrap into Θ and generate one solvable instance
# # ----------------------------
# Theta = {
#     "C": C,
#     "D": D,
#     "x0": np.array([0.4, -0.2, 0.1]),  # initial state
#     "u_max": 1.0,                      # input bound
#     "mu": 0.6,                         # witness peak effort (||u||_inf = 0.6)
#     "rho": 0.05,                       # goal radius
#     "seed": 7,                         # RNG for witness control sequence
# }
#
# instance = gen.generate(Theta)
#
# # replay the witness to verify solvability
# X = gen.rollout(instance["drift"], instance["input"],
#                 instance["init_goal"]["x0"], instance["witness_controls"])
#
# print("Final state (x_end):", instance["init_goal"]["x_end"])
# print("Max |u| in witness:", np.max(np.abs(instance["witness_controls"])))
# print("Degree-2 monomials order:", ex2)

import numpy as np
from poly_theta_generator import PolyThetaConfig
from poly_theta_sampler import ThetaSampler, ThetaSamplerConfig

cfg = PolyThetaConfig(n=3, m=1, d_f=2, d_G=2, dt=0.1, N=25)
sampler = ThetaSampler(cfg, ThetaSamplerConfig(), seed=23)

Theta = sampler.sample_theta()
inst  = sampler.sample_instance()  # directly solvable instance
print("Final state (x_end):", inst["init_goal"]["x_end"])
print("Max |u| in witness:", np.max(np.abs(inst["witness_controls"])))
print(inst['drift'])

x = inst['init_goal']['x0']

