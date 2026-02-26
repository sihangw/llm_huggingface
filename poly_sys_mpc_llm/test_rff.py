import os
import json
import numpy as np
from typing import Any, Dict, List
from poly_theta_generator import PolyThetaConfig
from ilqr_mpc import PolyDynamicsWithJac, QuadCostConfig
from poly_theta_sampler import ThetaSampler, ThetaSamplerConfig
from instance_sampler import ExperimentSampler, ExperimentSamplerConfig
from weights_search_ilqr import run_weight_search_for_instance_fast_ilqr
from CompositeScore import CompositeScoreConfig, CompositeTrajectoryScorer
from instance_prompt import build_instance_prompt, build_beta_prompt_simple
from ilqr_mpc import ILQRMPCConfig, FastILQRMPC, run_closed_loop_fast_ilqr_mpc, FastQuadraticCost

from poly_mpc import collect_and_save_dataset, save_dataset_np

sampler = ExperimentSampler(ExperimentSamplerConfig(
    dt=0.1, N=30, K=64, max_iters=10,
    n_range=(2, 6), m_range=(1, 3), d_f_range=(1, 3), d_G_range=(1, 3),
    H_range=(3, 8),
    T_outer_slack_frac_range=(0.3, 1.0),   # <N or >N allowed
    R_safe=(2.0, 10.0),          # sample safe radius relative to r
    master_seed=1234,
))

dataset = collect_and_save_dataset(
    sampler=sampler,
    num_trials=5,
    out_path="dataset_test1.jsonl",
    build_instance_prompt=build_instance_prompt,
    build_beta_prompt_simple=build_beta_prompt_simple,
    CompositeScoreConfig=CompositeScoreConfig,
    CompositeTrajectoryScorer=CompositeTrajectoryScorer,
    ILQRMPCConfig=ILQRMPCConfig,
    PolyDynamicsWithJac=PolyDynamicsWithJac,
    FastILQRMPC=FastILQRMPC,
    QuadCostConfig=QuadCostConfig,
    FastQuadraticCost=FastQuadraticCost,
    run_weight_search_for_instance_fast_ilqr=run_weight_search_for_instance_fast_ilqr,
    run_closed_loop_fast_ilqr_mpc=run_closed_loop_fast_ilqr_mpc,
    keep_top=5,
    early_stop_Y=0.99,
    verbose=True,
    save_every=1,
    overwrite=True,
)

paths = save_dataset_np(dataset, out_dir="dataset_test1_arrays", prefix="run1")
print(paths)