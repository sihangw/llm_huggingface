import numpy as np
import os, json, time
from typing import Any, Dict, List
from poly_theta_generator import PolyThetaConfig
from ilqr_mpc import PolyDynamicsWithJac, QuadCostConfig
from poly_theta_sampler import ThetaSampler, ThetaSamplerConfig
from instance_sampler import ExperimentSampler, ExperimentSamplerConfig
from weights_search_ilqr import run_weight_search_for_instance_fast_ilqr
from CompositeScore import CompositeScoreConfig, CompositeTrajectoryScorer
from instance_prompt import build_instance_prompt, build_beta_prompt_simple
from ilqr_mpc import ILQRMPCConfig, FastILQRMPC, run_closed_loop_fast_ilqr_mpc, FastQuadraticCost


def collect_and_save_dataset(
    sampler,  # ExperimentSamplerV3 (or your sampler with .sample_trial())
    num_trials: int,
    out_path: str,
    build_instance_prompt,
    build_beta_prompt_simple,
    CompositeScoreConfig,
    CompositeTrajectoryScorer,
    ILQRMPCConfig,
    PolyDynamicsWithJac,
    FastILQRMPC,
    QuadCostConfig,
    FastQuadraticCost,
    run_weight_search_for_instance_fast_ilqr,
    run_closed_loop_fast_ilqr_mpc,
    keep_top: int = 5,
    early_stop_Y: float = 0.99,
    verbose: bool = True,
    save_every: int = 1,
    overwrite: bool = True,
    time_every: int = 10,  # print timing every N trials (0 disables)
) -> List[Dict[str, Any]]:
    """
    Collect num_trials datapoints:
      - sample instance + params
      - build numeric feature vector feat_vec = [feat_sys.values(), feat_beta.values()]
      - run weight search (fast iLQR MPC)
      - store datum dict
      - save to JSONL incrementally (recommended for robustness)

    Saves:
      - out_path (JSONL): one record per line

    Returns:
      dataset: list of records (also saved to disk)
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if (not overwrite) and os.path.exists(out_path):
        raise FileExistsError(f"{out_path} exists and overwrite=False")

    dataset: List[Dict[str, Any]] = []

    # Write JSONL incrementally to avoid losing work if something crashes.
    mode = "w" if overwrite else "a"
    t_start = time.perf_counter()
    t_last = t_start
    with open(out_path, mode) as f:
        for i in range(num_trials):
            t0 = time.perf_counter()
            trial = sampler.sample_trial()
            inst = trial["inst"]
            p = trial["params"]

            # ---- prompts / numeric features ----
            feat_sys, prompt_sys = build_instance_prompt(inst, H=p["H"], T_mpc=p["T_outer"], R_safe=p.get("R_safe", None))
            feat_beta, prompt_beta = build_beta_prompt_simple(
                p["beta_margin"], p["beta_time"], p["beta_u"], p["beta_du"]
            )
            feat_vec = np.array(list(feat_sys.values()) + list(feat_beta.values()), dtype=float)

            # ---- scorer (uses betas) ----
            x_goal = np.asarray(inst["init_goal"]["x_end"], dtype=float)
            rho = float(inst["init_goal"]["goal_radius"])
            u_max = float(inst["meta"]["u_max"])

            def phi_target(x):
                return np.linalg.norm(x - x_goal) - rho

            cfg_score = CompositeScoreConfig(
                dt=float(inst["meta"]["dt"]),
                D=float(inst["meta"]["rho"]),
                beta_margin=float(p["beta_margin"]),
                beta_time=float(p["beta_time"]),
                beta_u=float(p["beta_u"]),
                beta_du=float(p["beta_du"]),
                U_ref=u_max,
                dU_ref=0.25 * u_max,
            )
            scorer = CompositeTrajectoryScorer(phi_target, cfg_score)
            # ---- ILQR MPC config ----
            cfg_ilqr = ILQRMPCConfig(
                H=int(p["H"]),
                max_iters=int(p["max_iters"]),
                u_min=-u_max,
                u_max=u_max,
                shift_fill="zero",

            )

            # IMPORTANT: builder must accept (inst, cost_obj, cfg_ilqr)
            def mpc_builder(inst_local, cost_obj, cfg_ilqr_local):
                dyn = PolyDynamicsWithJac(
                    cfg=inst_local["cfg"],
                    drift=inst_local["drift"],
                    inp=inst_local["input"],
                )
                return FastILQRMPC(dyn=dyn, cost=cost_obj, cfg=cfg_ilqr_local)

            # ---- run weight search ----
            best = run_weight_search_for_instance_fast_ilqr(
                inst=inst,
                scorer=scorer,
                run_closed_loop_fast_ilqr_mpc=run_closed_loop_fast_ilqr_mpc,
                mpc_builder=mpc_builder,
                ilqr_cfg_base=cfg_ilqr,
                QuadCostConfig=QuadCostConfig,
                FastQuadraticCost=FastQuadraticCost,
                K=int(p["K"]),
                T_outer=int(p["T_outer"]),
                rng=trial["rng_search"],
                keep_top=int(p.get("keep_top", keep_top)) if isinstance(p, dict) else keep_top,
                early_stop_Y=p.get("early_stop_Y", early_stop_Y) if isinstance(p, dict) else early_stop_Y,
                verbose=verbose,
            )

            # ---- record ----
            datum: Dict[str, Any] = {
                "trial_id": i,
                "params": p,
                "prompt_sys": prompt_sys,
                "prompt_score": prompt_beta,
                "feat_sys": feat_sys,
                "feat_score": feat_beta,
                "feat_vec": feat_vec.tolist(),
                "best_Y": float(best["Y"]),
                "success": best["success"],
                "best_w": np.asarray(best["w"], dtype=float).tolist(),
            }

            dataset.append(datum)

            # # save incrementally (JSONL)
            # if save_every > 0 and ((i + 1) % save_every == 0):
            #     f.write(json.dumps(datum) + "\n")
            #     f.flush()
            #
            # if verbose and ((i + 1) % 10 == 0):
            #     print(f"[{i + 1}/{num_trials}] collected. last best_Y={datum['best_Y']:.4f}")

            if save_every > 0 and ((i + 1) % save_every == 0):
                f.write(json.dumps(datum) + "\n")
                f.flush()

            # ---- timing prints ----
            t1 = time.perf_counter()
            dt_trial = t1 - t0

            if time_every and ((i + 1) % time_every == 0):
                elapsed = t1 - t_start
                avg = elapsed / (i + 1)
                since_last = t1 - t_last
                avg_block = since_last / time_every
                remain = avg * (num_trials - (i + 1))
                print(
                    f"[timing] {i + 1}/{num_trials} | "
                    f"trial={dt_trial:.3f}s | "
                    f"avg={avg:.3f}s | "
                    f"avg(last {time_every})={avg_block:.3f}s | "
                    f"elapsed={elapsed:.1f}s | "
                    f"ETA≈{remain:.1f}s"
                )
                t_last = t1

        total = time.perf_counter() - t_start
        print(f"[done] collected {num_trials} trials in {total:.2f}s (avg {total / num_trials:.3f}s/trial)")

    return dataset


def save_dataset_np(
    dataset: List[Dict[str, Any]],
    out_dir: str,
    prefix: str = "data",
) -> Dict[str, str]:
    """
    Optional helper: save dense arrays for training:
      - X.npy (features)
      - Y.npy (labels)
      - W.npy (best weights)
    """
    os.makedirs(out_dir, exist_ok=True)

    X = np.array([d["feat_vec"] for d in dataset], dtype=float)
    Y = np.array([d["best_Y"] for d in dataset], dtype=float)
    W = np.array([d["best_w"] for d in dataset], dtype=float)
    S = np.array([int(d["success"]) for d in dataset], dtype=np.int64)  # <-- add this

    paths = {
        "X": os.path.join(out_dir, f"{prefix}_X.npy"),
        "Y": os.path.join(out_dir, f"{prefix}_Y.npy"),
        "W": os.path.join(out_dir, f"{prefix}_W.npy"),
        "S": os.path.join(out_dir, f"{prefix}_S.npy"),
    }
    np.save(paths["X"], X)
    np.save(paths["Y"], Y)
    np.save(paths["W"], W)
    np.save(paths["S"], S)
    return paths

if __name__ == "__main__":
    out_dir = "data_10000_dt05_N20_sampling"
    os.makedirs(out_dir, exist_ok=True)

    # JSONL path in data/
    jsonl_path = os.path.join(out_dir, "dataset_all.jsonl")
    sampler = ExperimentSampler(ExperimentSamplerConfig(
        dt=0.5, N=20, K=10, max_iters=5,
        n_range=(2, 6), m_range=(2, 3), d_f_range=(1, 3), d_G_range=(1, 3),
        H_range=(3, 5),
        T_outer_slack_frac_range=(0.2, 1.0),  # <N or >N allowed
        R_safe=(2.0, 100.0),  # sample safe radius relative to r
        u_max_range=(0.1, 50.0),
        prob_mode=(0.2, 0.6, 0.2),
        master_seed=32234,
    ))

    dataset = collect_and_save_dataset(
        sampler=sampler,
        num_trials=10000,
        out_path=jsonl_path,
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
        verbose=False,
        save_every=1,
        overwrite=True,
        time_every=10,
    )

    paths = save_dataset_np(dataset, out_dir=out_dir, prefix="run1")
    print(paths)