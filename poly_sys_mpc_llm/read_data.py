import numpy as np
import json

def load_jsonl(path: str):
    data = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data

# dataset = load_jsonl("dataset_run1.jsonl")
# print(len(dataset))
# print(dataset[0].keys())
# print("feat_vec length:", len(dataset[0]["feat_vec"]))
# print("best_Y:", dataset[0]["best_Y"])
# print("prompt_sys:", dataset[0]["prompt_sys"])
# print("prompt_score:", dataset[0]["prompt_score"])

#
# X = np.load("dataset_run1_arrays/run1_X.npy")
# Y = np.load("dataset_run1_arrays/run1_Y.npy")
# print("100")
# print(Y)
# print(X)
# W = np.load("dataset_run1_arrays/run1_W.npy")


dataset = load_jsonl("data_dt01_N50/dataset_all.jsonl")
print(len(dataset))
print(dataset[0].keys())
print("feat_vec length:", len(dataset[0]["feat_vec"]))
print("best_Y:", dataset[0]["best_Y"])
# print("prompt_sys:", dataset[0]["prompt_sys"])
# print("prompt_score:", dataset[0]["prompt_score"])


# Y = np.load("data_100_dt01_N100/run1_Y.npy")
# print("100")
# print(Y)

Y_llm = np.load("sft_eval_scores/Y_llm.npy")
Y_oracle = np.load("data_500_dt01_N100/run1_Y.npy")
W_llm = np.load("sft_eval_scores/W_llm.npy")

print(Y_llm)
# print(Y_llm)
# print(Y_oracle)


# X = dataset[1]["params"]
# print(X)
#
# import numpy as np
# import os, json, time
# from typing import Any, Dict, List
# from poly_theta_generator import PolyThetaConfig
# from ilqr_mpc import PolyDynamicsWithJac, QuadCostConfig
# from poly_theta_sampler import ThetaSampler, ThetaSamplerConfig
# from instance_sampler import ExperimentSampler, ExperimentSamplerConfig
# from weights_search_ilqr import run_weight_search_for_instance_fast_ilqr
# from CompositeScore import CompositeScoreConfig, CompositeTrajectoryScorer
# from instance_prompt import build_instance_prompt, build_beta_prompt_simple
# from ilqr_mpc import ILQRMPCConfig, FastILQRMPC, run_closed_loop_fast_ilqr_mpc, FastQuadraticCost
#
#
# cfg = PolyThetaConfig(n=6, m=1, d_f=2, d_G=2, dt=0.1, N=50, r=8.908970372559686)
# sampler = ThetaSampler(cfg, ThetaSamplerConfig(), seed=4231)
# inst  = sampler.sample_instance()  # directly solvable instance
#
# feat_sys, prompt_sys = build_instance_prompt(inst, 5, 0.2)
# print(prompt_sys)
#
# # Score definitions
# x_goal = np.asarray(inst["init_goal"]["x_end"], dtype=float)
# rho    = float(inst["init_goal"]["goal_radius"])
# u_max  = float(inst["meta"]["u_max"])
# # u_max = 100
# def phi_target(x):
#     # negative inside target set
#     return np.linalg.norm(x - x_goal) - rho
# b_m = 1.0
# b_t = 1.0
# b_u = 1.0
# b_du = 1.0
# cfg_score = CompositeScoreConfig(
#     dt=inst["meta"]["dt"],
#     D=inst["meta"]["rho"],
#     beta_margin=b_m,
#     beta_time=b_t,
#     beta_u=b_u,
#     beta_du=b_du,
#     U_ref=u_max,
#     dU_ref=0.25*u_max,
# )
#
# scorer = CompositeTrajectoryScorer(phi_target, cfg_score)
#
# # Solve for each instance by MPC
# dyn = PolyDynamicsWithJac(
#     cfg=cfg,
#     drift=inst["drift"],
#     inp=inst["input"],
#     # eval_monomials=eval_monomials,  # not strictly needed; we used monomials_and_grads
# )
#
# # cost
# qc = QuadCostConfig(
#     x_goal=np.asarray(inst["init_goal"]["x_end"], dtype=float),
#     u_max=float(inst["meta"]["u_max"]),
#     r_safe=float(inst["meta"]["r"]),
# )
# cost = FastQuadraticCost(qc)
#
# # MPC config
# cfg_ilqr = ILQRMPCConfig(
#     H=5,
#     max_iters=10,
#     u_min=-float(inst["meta"]["u_max"]),
#     u_max= float(inst["meta"]["u_max"]),
#     shift_fill="zero",
# )
#
# mpc = FastILQRMPC(dyn=dyn, cost=cost, cfg=cfg_ilqr)
#
# X_cl, U_cl = run_closed_loop_fast_ilqr_mpc(
#     mpc=mpc,
#     step_fn_true=inst["step"],  # true closed-loop step
#     x0=np.asarray(inst["init_goal"]["x0"], dtype=float),
#     T_outer=100,
# )
#
# print("Final:", X_cl[-1], "Goal:", inst["init_goal"]["x_end"])
# print(np.asarray(inst["init_goal"]["x0"], dtype=float))
# print(inst["meta"]["u_max"])



















import reward_MLP_offline as rff
X = np.load("data_500_dt01_N100/run1_X.npy")
Y = np.load("sft_eval_scores/Y_llm.npy")
# print(Y)
sel = rff.select_rff_and_init(X, Y)
phi = sel["phi"]
w0  = sel["w0"]



X_test = np.load("data_500_dt01_N100/run1_X.npy")
Y_test = np.load("sft_eval_scores/Y_llm.npy")
Y_opt = np.load("data_500_dt01_N100/run1_Y.npy")
X_test = np.asarray(X_test, dtype=float)
Y_test = np.asarray(Y_test, dtype=float).reshape(-1)
Y_opt = np.asarray(Y_opt, dtype=float).reshape(-1)
# predictions for all test samples
y_pred = np.array([float(phi.transform(X_test[i]) @ w0) for i in range(X_test.shape[0])], dtype=float)
for i in range(10):
    print(i, "y_true=", float(Y_test[i]), "y_pred=", float(y_pred[i]), "y_opt=", float(Y_opt[i]))












# # 1) make sure x is a 1D feature vector
# x = np.asarray(X_test, dtype=float).reshape(-1)
# y_true = float(np.asarray(Y_test, dtype=float))
#
# # 2) predict with the selected RFF map + learned weights w0
# phi_x = phi.transform(x)          # (Dphi,)
# y_pred = float(phi_x @ w0)        # scalar
#
# for i in range(10):
#     print(i, "y_true=", float(Y_test[i]), "y_pred=", float(y_pred[i]))


# for i in range(X_test.shape[0]):
#     print(i, "y_true=", float(Y_test[i]), "y_pred=", float(y_pred[i]), "err=", float(err[i]))

# print("y_true:", y_true)
# print("y_pred:", y_pred)
# print("abs error:", abs(y_pred - y_true))
# print("sq error:", (y_pred - y_true)**2)



# X_test = np.asarray(X_test, dtype=float)
# Y_test = np.asarray(Y_test, dtype=float).reshape(-1)
#
# # predictions for all test samples
# y_pred = np.array([float(phi.transform(X_test[i]) @ w0) for i in range(X_test.shape[0])], dtype=float)
#
# # metrics
# err = y_pred - Y_test
# rmse = float(np.sqrt(np.mean(err**2)))
# mae  = float(np.mean(np.abs(err)))
#
# print("N_test:", X_test.shape[0])
# print("RMSE:", rmse)
# print("MAE:", mae)
#
# # optional: look at first few
# for i in range(X_test.shape[0]):
#     print(i, "y_true=", float(Y_test[i]), "y_pred=", float(y_pred[i]), "err=", float(err[i]))