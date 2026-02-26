import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"   # must be set before torch initializes CUDA

import re
import dill
import json
import math
import torch
torch.cuda.set_device(0)

import argparse
import numpy as np
from datasets import Dataset

from peft import LoraConfig, PeftModel
from datasets import load_dataset
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer

# from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from typing import List, Dict, Any, Tuple, Union

from trl.rewards import think_format_reward

from ilqr_mpc import PolyDynamicsWithJac, QuadCostConfig
from instance_sampler import ExperimentSampler, ExperimentSamplerConfig
from weights_search_ilqr import run_weight_search_for_instance_fast_ilqr, make_fast_cost_from_w
from CompositeScore import CompositeScoreConfig, CompositeTrajectoryScorer
from instance_prompt import build_instance_prompt, build_beta_prompt_simple
from ilqr_mpc import ILQRMPCConfig, FastILQRMPC, run_closed_loop_fast_ilqr_mpc, FastQuadraticCost




# -----------------------------
# Helper functions (same as before)
# -----------------------------
SYSTEM_MSG = """
Respond in the following format:
{\"w\": [w1, w2, w3, w4, w5, w6, w7]}
Where w1-w7 are non-negative numbers.
The <answer> section MUST contain valid JSON only.
"""

# def build_user_msg(prompt_sys: str, prompt_score: str, w_dim: int) -> str:
#     # Keep this consistent with your downstream parser and weight sampler semantics.
#     semantics = (
#         "Weight vector format (length {w_dim}): "
#         "w = [Ws_track, Ws_safe, Ws_u, Ws_smooth, Wt_track, Wt_safe, Wt_u] "
#         "-Ws_* are stage (running) weights. "
#         "-Wt_* are terminal weights. "
#         "-track penalizes ||x - x_goal||^2 "
#         "-safe penalizes max(0, ||x||_inf - R_safe)^2 "
#         "-u penalizes ||u||^2 normalized by u_max "
#         "-smooth penalizes ||u - u_prev||^2 (stage only) "
#         "Weight prior/range hint (for stability; do not output negatives): "
#         "Ws_track, Ws_safe in about [10^-1.5, 10^2] (may scale with 1/rho). "
#         "Ws_u, Ws_smooth in about [10^-3, 10^0.5]. "
#         "Wt_track, Wt_safe typically 3x–30x larger than stage weights. "
#         "Wt_u typically similar scale to Ws_u. "
#     )
#     return (
#         # "INSTANCE SUMMARY:\n" #already in dataset
#         f"{prompt_sys} "
#         # "SCORE PREFERENCES:\n"
#         f"{prompt_score} "
#         f"{semantics} "
#         "TASK: Choose w to maximize the score. "
#         "OUTPUT FORMAT (STRICT): Return ONLY JSON: {{\"w\": [..{w_dim} numbers..]}}"
#         "Example:"
#         "{\"w\": [1.0, 5.0, 0.1, 0.1, 10.0, 20.0, 0.1]}"
#     )

def build_user_msg(prompt_sys: str, prompt_score: str, w_dim: int) -> str:
    semantics = (
        f"w = [Ws_track, Ws_safe, Ws_u, Ws_smooth, Wt_track, Wt_safe, Wt_u]"
        f"All {w_dim} values must be >= 0."
    )
    return (
        "INSTANCE SUMMARY:"
        f"{prompt_sys}"
        "SCORE PREFERENCES:"
        f"{prompt_score}"
        "TASK: Choose w to maximize the final score."
        f"{semantics}"
        "OUTPUT FORMAT (STRICT):"
        f"Return ONLY JSON: {{\"w\": [..{w_dim} numbers..]}}"
        # "Hints:"
        # "- Terminal track/safe weights are usually larger than stage weights."
        # "- If safety matters more, increase safe weights."
        # "- If effort/smoothness matters more, increase Ws_u / Ws_smooth."
        # "OUTPUT FORMAT (STRICT):"
        # f"Return ONLY JSON: {{\"w\": [..{w_dim} numbers..]}}"
        # "Example:\n"
        # "{\"w\": [1.0, 5.0, 0.1, 0.1, 10.0, 20.0, 0.1]}\n"
    )

def _extract_json_obj(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError("No JSON object found")
    return json.loads(m.group(0))

def parse_w(content, w_dim=7):
    # Accept either a string JSON or an already-parsed dict
    if isinstance(content, dict):
        obj = content
    else:
        text = (content or "").strip()
        obj = json.loads(text)

    w = np.asarray(obj["w"], dtype=np.float32)
    if w.shape != (w_dim,):
        raise ValueError(f"wrong dim {w.shape}")
    if not np.all(np.isfinite(w)):
        raise ValueError("non-finite")
    return w

def load_prompts(jsonl_path: str) -> List[Dict[str, str]]:
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            rows.append({
                "prompt_sys": ex["prompt_sys"],
                "prompt_score": ex["prompt_score"],
            })
    return rows

def _get_w_from_row(row: Dict[str, Any]) -> List[float]:
    # support a few common keys
    if "best_w" in row:
        w = row["best_w"]
    elif "w" in row:
        w = row["w"]
    else:
        raise KeyError("Row missing weight vector key: expected 'best_w' or 'w'")
    w = np.asarray(w, dtype=float).reshape(-1)
    return w.tolist()

def _sanitize_w(
    w: List[float],
    w_dim: int,
    w_clip: Tuple[float, float],
) -> List[float]:
    w = np.asarray(w, dtype=float).reshape(-1)
    if w.size != w_dim:
        raise ValueError(f"Expected w_dim={w_dim}, got {w.size}")
    if not np.all(np.isfinite(w)):
        raise ValueError("Non-finite weights")
    lo, hi = w_clip
    w = np.clip(w, lo, hi)
    return w.tolist()

def make_messages_from_row(row: Dict[str, Any], w_dim: int, w_clip: Tuple[float, float]) -> Dict[str, Any]:
    prompt_sys = row.get("prompt_sys", "")
    prompt_score = row.get("prompt_score", "")
    params = row.get("params", "")
    ids = row.get("trial_id", "")

    if not prompt_sys:
        raise KeyError("Row missing 'prompt_sys'")
    if not prompt_score:
        raise KeyError("Row missing 'prompt_score'")

    w = _get_w_from_row(row)
    w = _sanitize_w(w, w_dim=w_dim, w_clip=w_clip)

    assistant = json.dumps({"w": w}, separators=(",", ":"))  # compact JSON

    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": build_user_msg(prompt_sys, prompt_score, w_dim=w_dim)},
        # {"role": "assistant", "content": assistant},
    ]

    return {"prompt": messages, "solution": assistant, "trial_ids": ids, "params":params}

# LOAD instances directly for reward
def load_instances(path: str) -> dict:
    with open(path, "rb") as f:
        instances = dill.load(f)
    print(f"Loaded {len(instances)} instances")
    return instances

# -----------------------------
# GRPO-specific reward function
# -----------------------------
def _pow10(x: float) -> float:
    return float(10.0 ** x)

def get_weight_bounds_from_sampler(
    rho_goal: float,
    w_dim: int = 7,
    rho_track_ref: float = 0.2,
    rho_track_clip: tuple[float, float] = (0.5, 5.0),
) -> tuple[np.ndarray, np.ndarray]:
    if w_dim != 7:
        raise ValueError("Assumes w_dim=7")

    Ws_track_lo, Ws_track_hi = _pow10(-1.5), _pow10(2.0)
    Ws_safe_lo,  Ws_safe_hi  = _pow10(-1.5), _pow10(2.0)
    Ws_u_lo,     Ws_u_hi     = _pow10(-3.0), _pow10(0.5)
    Ws_sm_lo,    Ws_sm_hi    = _pow10(-3.0), _pow10(0.5)

    if rho_goal is None or rho_goal <= 0:
        rho_scale = 1.0
    else:
        rho_scale = float(np.clip(rho_track_ref / float(rho_goal),
                                  rho_track_clip[0], rho_track_clip[1]))

    Ws_track_lo *= rho_scale
    Ws_track_hi *= rho_scale

    mult_ts_lo, mult_ts_hi = _pow10(0.5), _pow10(1.5)
    mult_u_lo,  mult_u_hi  = _pow10(-0.5), _pow10(0.5)

    Wt_track_lo, Wt_track_hi = Ws_track_lo * mult_ts_lo, Ws_track_hi * mult_ts_hi
    Wt_safe_lo,  Wt_safe_hi  = Ws_safe_lo  * mult_ts_lo, Ws_safe_hi  * mult_ts_hi
    Wt_u_lo,     Wt_u_hi     = Ws_u_lo     * mult_u_lo,  Ws_u_hi     * mult_u_hi

    lo = np.array([Ws_track_lo, Ws_safe_lo, Ws_u_lo, Ws_sm_lo, Wt_track_lo, Wt_safe_lo, Wt_u_lo], dtype=float)
    hi = np.array([Ws_track_hi, Ws_safe_hi, Ws_u_hi, Ws_sm_hi, Wt_track_hi, Wt_safe_hi, Wt_u_hi], dtype=float)
    return lo, hi

def reward_num_unique_letters(completions, **kwargs):
    """Reward function that rewards completions with more unique letters."""
    completion_contents = [completion[0]["content"] for completion in completions]
    return [float(len(set(content))) for content in completion_contents]

# instances = load_instances(path="data_500_dt05_N20_grpo/instances.dill")

def rank_with_ties_average(rewards, tie_tol=0.0):
    r = np.asarray(rewards, dtype=float)
    K = len(r)
    if K == 1:
        return np.array([0.5], dtype=np.float32)

    order = np.argsort(r)  # ascending: worst..best
    ranks = np.empty(K, dtype=float)
    ranks[order] = np.arange(K, dtype=float)

    # Average ranks for ties in sorted order
    r_sorted = r[order]
    i = 0
    while i < K:
        j = i + 1
        while j < K and abs(r_sorted[j] - r_sorted[i]) <= tie_tol:
            j += 1
        if j - i > 1:
            idxs = order[i:j]
            ranks[idxs] = ranks[idxs].mean()
        i = j

    return (ranks / (K - 1.0)).astype(np.float32)

def rank_ties_random(rewards, rng=None, tie_tol=0.0):
    if rng is None:
        rng = np.random.default_rng()

    r = np.asarray(rewards, dtype=float)
    K = len(r)
    if K == 1:
        return np.array([0.5], dtype=np.float32)

    order = np.argsort(r)
    r_sorted = r[order]

    # shuffle within tie blocks
    order2 = order.copy()
    i = 0
    while i < K:
        j = i + 1
        while j < K and abs(r_sorted[j] - r_sorted[i]) <= tie_tol:
            j += 1
        if j - i > 1:
            block = order2[i:j].copy()
            rng.shuffle(block)
            order2[i:j] = block
        i = j

    ranks = np.empty(K, dtype=float)
    ranks[order2] = np.arange(K, dtype=float)
    return (ranks / (K - 1.0)).astype(np.float32)


# def weights_reward(completions, **kwargs):
#     """
#     GRPO reward with:
#       - strong format/bounds reward (fast format learning)
#       - normalized MPC score bonus (prevents mode collapse)
#     """
#     texts = [c[0]["content"] for c in completions]
#     n = len(texts)
#
#     def to_list_strict(key, default=None):
#         """Broadcast scalars or length-1 lists; require length n otherwise."""
#         val = kwargs.get(key, default)
#         if isinstance(val, (list, tuple, np.ndarray)):
#             lst = list(val)
#             if len(lst) == n:
#                 return lst
#             if len(lst) == 1:
#                 return lst * n
#             raise ValueError(f"{key} length {len(lst)} != n={n}")
#         return [val] * n
#
#     trial_id_list = to_list_strict("trial_ids")
#     params_list   = to_list_strict("params")
#
#     # ----- reward buffers -----
#     rewards = np.full(n, -2.0, dtype=float)     # default: invalid JSON/exception
#     Y_buf   = np.full(n, np.nan, dtype=float)   # store MPC score for valid ones
#
#     # ----- knobs -----
#     R_VALID_FORMAT = 1.0    # teaches JSON quickly
#     R_BOUNDS_FAIL  = -1.0   # valid JSON but bad weights
#     R_EXCEPTION    = -2.0   # parse error / crash / wrong schema
#     alpha          = 1.0    # scale of normalized MPC term (try 0.5~2.0)
#
#     lo = np.ones(7, dtype=float) * 1e-3
#     hi = np.ones(7, dtype=float) * 1e3
#
#     # ========== PASS 1: gate + compute Y ==========
#     for idx, text in enumerate(texts):
#         try:
#             trial_idx = trial_id_list[idx]
#             rho_goal  = float(instances[trial_idx]["meta"]["rho"])
#
#             # 1) parse JSON -> w (must throw if invalid)
#             w = parse_w(text, w_dim=7)  # your parse_w
#             w = np.asarray(w, dtype=float).reshape(-1)
#
#             # 2) hard checks
#             if w.shape[0] != 7 or not np.all(np.isfinite(w)) or np.any(w < 0):
#                 rewards[idx] = R_EXCEPTION
#                 continue
#
#             # 3) bounds gate
#             if np.any(w < lo) or np.any(w > hi):
#                 rewards[idx] = R_BOUNDS_FAIL
#                 continue
#
#             # 4) passed format+bounds
#             rewards[idx] = R_VALID_FORMAT
#
#             # ----- scorer (depends on betas + goal) -----
#             u_max = instances[trial_idx]["meta"]["u_max"]
#             x0 = instances[trial_idx]["init_goal"]["x0"]
#             xg = instances[trial_idx]["init_goal"]["x_end"]
#
#             def phi_target(x):
#                 return np.linalg.norm(x0 - xg) - rho_goal
#
#             p = params_list[idx]
#             cfg_score = CompositeScoreConfig(
#                 dt=float(instances[trial_idx]["meta"]["dt"]),
#                 D=float(instances[trial_idx]["meta"]["rho"]),
#                 beta_margin=float(p["beta_margin"]),
#                 beta_time=float(p["beta_time"]),
#                 beta_u=float(p["beta_u"]),
#                 beta_du=float(p["beta_du"]),
#                 U_ref=u_max,
#                 dU_ref=0.25 * u_max,
#             )
#             scorer = CompositeTrajectoryScorer(phi_target, cfg_score)
#
#             cfg_ilqr = ILQRMPCConfig(
#                 H=int(p["H"]),
#                 max_iters=int(p["max_iters"]),
#                 u_min=-u_max,
#                 u_max=u_max,
#                 shift_fill="zero",
#             )
#
#             def mpc_builder(inst_local, cost_obj, cfg_ilqr_local):
#                 dyn = PolyDynamicsWithJac(cfg=inst_local["cfg"], drift=inst_local["drift"], inp=inst_local["input"])
#                 return FastILQRMPC(dyn=dyn, cost=cost_obj, cfg=cfg_ilqr_local)
#
#             inst = instances[trial_idx]
#             cost_obj = make_fast_cost_from_w(inst, w, QuadCostConfig, FastQuadraticCost)
#             mpc_llm = mpc_builder(inst, cost_obj, cfg_ilqr)
#
#             X_llm, U_llm = run_closed_loop_fast_ilqr_mpc(
#                 mpc=mpc_llm,
#                 step_fn_true=inst["step"],
#                 x0=x0,
#                 T_outer=int(p["T_outer"]),
#             )
#             score_dict = scorer.score(X_llm, U_llm)
#             Y_llm = float(score_dict["score"])   # expected 0..1
#
#             # store for normalization
#             Y_buf[idx] = 1000*Y_llm
#
#         except Exception:
#             rewards[idx] = R_EXCEPTION
#
#     # # ========== PASS 2: normalize MPC among valid samples ==========
#     # valid = np.isfinite(Y_buf)  # only those that passed gates and got a score
#     # if np.any(valid):
#     #     Yv = Y_buf[valid]
#     #
#     #     # Option A: z-score normalization (recommended)
#     #     mu = float(np.mean(Yv))
#     #     sd = float(np.std(Yv)) + 1e-6
#     #     Y_norm = (Yv - mu) / sd               # mean 0, std 1
#     #
#     #     # If std is extremely tiny, z-score becomes noisy; fall back to minmax:
#     #     if sd < 1e-3:
#     #         ymin, ymax = float(np.min(Yv)), float(np.max(Yv))
#     #         den = (ymax - ymin) + 1e-6
#     #         Y_norm = 2.0 * (Yv - ymin) / den - 1.0  # [-1, 1]
#     #
#     #     rewards[valid] += alpha * Y_norm
#
#     # ========== PASS 2: normalize MPC among valid samples ==========
#     valid = np.isfinite(Y_buf)
#     if np.any(valid):
#         Yv = Y_buf[valid]
#         mu = float(np.mean(Yv))
#         sd = float(np.std(Yv))
#
#         if sd >= 1e-3:
#             Y_norm = (Yv - mu) / (sd + 1e-6)
#         else:
#             ymin, ymax = float(np.min(Yv)), float(np.max(Yv))
#             if (ymax - ymin) < 1e-6:
#                 Y_norm = np.zeros_like(Yv)  # <-- key fix: don't make them all -1
#             else:
#                 den = (ymax - ymin)
#                 Y_norm = 2.0 * (Yv - ymin) / den - 1.0
#
#         rewards[valid] += alpha * Y_norm
#
#     print("this is the whole reward:", rewards)
#
#     return rewards.tolist()

# instances = load_instances(path="data_10000_dt05_N20_grpo/instances.dill")
instances = load_instances(path="data_10000_dt05_N20_sampling/instances.dill")
def weights_reward(completions, **kwargs):
    """
    Strict format reward for GRPO:
      - returns -1.0 if not valid JSON {"w":[...]} or wrong dim/non-finite/negatives/out of bounds
      - else returns 0.0 (format reward only)

    Expects:
      completions: list[list[dict]]  (TRL format)
      kwargs:      extra dataset columns passed automatically

    """
    # TRL gives completions as: completions[i][0]["content"]
    texts = [c[0]["content"] for c in completions]
    n = len(texts)

    # # Helper to ensure list length matches texts
    def to_list(key, default=None):
        val = kwargs.get(key, default)
        if isinstance(val, (list, tuple, np.ndarray)):
            lst = list(val)
            if len(lst) != n:
                return lst * n  # ← just multiply to get n copies!
            return lst
        else:
            return [val] * n

    trial_id_list = to_list("trial_ids")
    params_list = to_list("params")

    rewards = []
    for idx, text in enumerate(texts):
        r = 0
        try:
            trial_idx = trial_id_list[idx]
            rho_goal = float(instances[trial_idx]["meta"]["rho"])
            # 1) parse JSON -> w
            w = parse_w(text, w_dim=7)  # <- YOUR parse_w
            r += 0.2

            # lo, hi = get_weight_bounds_from_sampler(float(rho_goal), w_dim=7)
            lo = np.ones(7, dtype=float) * 1e-3
            hi = np.ones(7, dtype=float) * 1e3
            if np.any(w < lo) or np.any(w > hi):
                rewards.append(r)
                # raw_scores[idx] = -10.0   # valid JSON but bad bounds
                continue
            r += 0.2

            # valid
            # ----- scorer (depends on betas + goal) -----
            u_max = instances[trial_idx]["meta"]["u_max"]
            x0 = instances[trial_idx]["init_goal"]["x0"]
            xg = instances[trial_idx]["init_goal"]["x_end"]

            def phi_target(x):
                return np.linalg.norm(x0 - xg) - rho_goal  # negative inside target

            cfg_score = CompositeScoreConfig(
                dt = float(instances[trial_idx]["meta"]["dt"]),
                D = float(instances[trial_idx]["meta"]["rho"]),
                beta_margin=float(params_list[idx]["beta_margin"]),
                beta_time=float(params_list[idx]["beta_time"]),
                beta_u=float(params_list[idx]["beta_u"]),
                beta_du=float(params_list[idx]["beta_du"]),
                U_ref=u_max,
                dU_ref=0.25 * u_max,
            )
            scorer = CompositeTrajectoryScorer(phi_target, cfg_score)

            # ----- MPC config for evaluation -----
            cfg_ilqr = ILQRMPCConfig(
                H=int(params_list[idx]["H"]),
                max_iters=int(params_list[idx]["max_iters"]),
                u_min=-u_max,
                u_max=u_max,
                shift_fill="zero",
            )

            def mpc_builder(inst_local, cost_obj, cfg_ilqr_local):
                dyn = PolyDynamicsWithJac(cfg=inst_local["cfg"], drift=inst_local["drift"], inp=inst_local["input"])
                return FastILQRMPC(dyn=dyn, cost=cost_obj, cfg=cfg_ilqr_local)

            # Ensure ndarray float
            w_llm = np.asarray(w, dtype=float).reshape(-1)
            # print("this is approved w", w_llm)
            # Score it
            inst = instances[trial_idx]
            cost_obj = make_fast_cost_from_w(inst, w_llm, QuadCostConfig, FastQuadraticCost)
            mpc_llm = mpc_builder(inst, cost_obj, cfg_ilqr)
            X_llm, U_llm = run_closed_loop_fast_ilqr_mpc(
                mpc=mpc_llm,
                step_fn_true=inst["step"],
                x0=x0,
                T_outer=int(params_list[idx]["T_outer"]),
            )
            score_dict = scorer.score(X_llm, U_llm)
            Y_llm = float(score_dict["score"])
            # print("potential score", Y_llm)
            r += 1000*Y_llm

            # # This is to encourage scale of w_llm
            # logw = np.log10(np.maximum(w_llm, 1e-12))
            # target = np.array([-0.0, -0.0, -1.5, -1.5, 1.0, 1.0, -1.0])  # rough centers
            # print("usually this", float(np.sum((logw - target)**2)))
            # r -= 0.02 * float(np.sum((logw - target)**2))
            rewards.append(r)

        except Exception:
            r -= 0.1
            # r -= 10.0
            rewards.append(r)

    print("this is the whole reward:", rewards)
    return rewards



def _rank_to_unit_interval(scores: np.ndarray) -> np.ndarray:
    """
    Convert scores (higher is better) to rank-based values in [-1, +1].
    Ties get the same rank value (average rank).
    """
    K = len(scores)
    if K == 1:
        return np.array([0.0], dtype=np.float32)

    # argsort twice gives ranks; handle ties by averaging
    order = np.argsort(scores)  # ascending
    ranks = np.empty(K, dtype=np.float32)
    ranks[order] = np.arange(K, dtype=np.float32)

    # tie handling: average ranks for equal scores
    # (simple approach)
    uniq = {}
    for i, s in enumerate(scores):
        uniq.setdefault(s, []).append(i)
    for s, idxs in uniq.items():
        if len(idxs) > 1:
            avg = float(np.mean(ranks[idxs]))
            ranks[idxs] = avg

    # map [0..K-1] -> [-1..+1]
    return 2.0 * (ranks / (K - 1.0)) - 1.0


def weights_reward_ranked(completions, **kwargs):
    texts = [c[0]["content"] for c in completions]
    n = len(texts)

    # TRL usually passes num_generations into kwargs for GRPO, but not always.
    # Best: use kwargs if present, else fall back to training_args.num_generations.
    K = kwargs.get("num_generations", None)
    if K is None:
        # If not provided, assume fixed K and infer from trial_ids repetition if possible.
        # Most setups: n is divisible by K; if not, you must pass K in kwargs.
        K = 8  # <-- set to your GRPOConfig.num_generations
    K = int(K)
    assert n % K == 0, f"Expected n%K==0, got n={n}, K={K}"

    def to_list(key, default=None):
        val = kwargs.get(key, default)
        if isinstance(val, (list, tuple, np.ndarray)):
            lst = list(val)
            return lst if len(lst) == n else lst * n
        return [val] * n

    trial_id_list = to_list("trial_ids")
    params_list   = to_list("params")

    # 1) compute raw scores per completion
    raw_scores = np.full(n, -1e9, dtype=np.float32)  # huge negative = worst
    # optional: keep a small format bonus separately
    format_ok  = np.zeros(n, dtype=np.float32)

    for idx, text in enumerate(texts):
        try:
            raw = (text or "").strip()
            # STRICT JSON gate (recommended). If you want “embedded JSON allowed”, set require_json_only=False.
            w = parse_w(raw, w_dim=7)
            format_ok[idx] = 0.2  # small bonus for being parseable

            trial_idx = trial_id_list[idx]
            rho_goal = float(instances[trial_idx]["meta"]["rho"])

            # lo, hi = get_weight_bounds_from_sampler(float(rho_goal), w_dim=7)
            lo = np.ones(7, dtype=float) * 1e-3
            hi = np.ones(7, dtype=float) * 1e3
            if np.any(w < lo) or np.any(w > hi):
                raw_scores[idx] = -10.0   # valid JSON but bad bounds
                continue

            # ---- MPC score in [0,1] ----
            u_max = instances[trial_idx]["meta"]["u_max"]
            x0 = instances[trial_idx]["init_goal"]["x0"]
            xg = instances[trial_idx]["init_goal"]["x_end"]

            def phi_target(x):
                return np.linalg.norm(x0 - xg) - rho_goal

            cfg_score = CompositeScoreConfig(
                dt=float(instances[trial_idx]["meta"]["dt"]),
                D=float(instances[trial_idx]["meta"]["rho"]),
                beta_margin=float(params_list[idx]["beta_margin"]),
                beta_time=float(params_list[idx]["beta_time"]),
                beta_u=float(params_list[idx]["beta_u"]),
                beta_du=float(params_list[idx]["beta_du"]),
                U_ref=u_max,
                dU_ref=0.25 * u_max,
            )
            scorer = CompositeTrajectoryScorer(phi_target, cfg_score)

            cfg_ilqr = ILQRMPCConfig(
                H=int(params_list[idx]["H"]),
                max_iters=int(params_list[idx]["max_iters"]),
                u_min=-u_max,
                u_max=u_max,
                shift_fill="zero",
            )

            def mpc_builder(inst_local, cost_obj, cfg_ilqr_local):
                dyn = PolyDynamicsWithJac(cfg=inst_local["cfg"], drift=inst_local["drift"], inp=inst_local["input"])
                return FastILQRMPC(dyn=dyn, cost=cost_obj, cfg=cfg_ilqr_local)

            inst = instances[trial_idx]
            w_llm = np.asarray(w, dtype=float).reshape(-1)

            cost_obj = make_fast_cost_from_w(inst, w_llm, QuadCostConfig, FastQuadraticCost)
            mpc_llm = mpc_builder(inst, cost_obj, cfg_ilqr)
            X_llm, U_llm = run_closed_loop_fast_ilqr_mpc(
                mpc=mpc_llm,
                step_fn_true=inst["step"],
                x0=x0,
                T_outer=int(params_list[idx]["T_outer"]),
            )

            Y = float(scorer.score(X_llm, U_llm)["score"])  # in [0,1]
            raw_scores[idx] = Y

        except Exception:
            # leave raw_scores as -1e9 (worst), format_ok as 0
            pass

    # 2) convert raw scores to rank rewards per group of K
    rewards = np.zeros(n, dtype=np.float32)
    print(rewards)
    for g in range(n // K):
        a = g * K
        b = a + K
        group_scores = raw_scores[a:b]

        # If ALL are invalid (-1e9), give all the same reward (0) to avoid NaNs.
        if np.all(group_scores < -1e8):
            rewards[a:b] = -2.0  # or 0.0; -2 discourages total garbage
            continue

        rank_vals = _rank_to_unit_interval(group_scores)  # [-1, +1]
        rewards[a:b] = rank_vals

    # 3) final reward = rank reward + small format bonus
    final = rewards + format_ok
    print(final.tolist())
    return final.tolist()

def format_reward(completions, **kwargs):
    """
    Simple format reward - checks JSON structure only.
    Max reward: 1.0
    """
    texts = [c[0]["content"] for c in completions]
    rewards = []

    for text in texts:
        r = 0.0

        try:
            # Clean text
            cleaned = re.sub(r'```json\s*', '', text)
            cleaned = re.sub(r'```\s*', '', cleaned).strip()

            # Try parse
            try:
                obj = json.loads(cleaned)
            except:
                # Try finding JSON in text
                match = re.search(r'\{[^{}]*"w"\s*:\s*\[[^\]]*\][^{}]*\}', cleaned)
                if not match:
                    rewards.append(0.0)
                    continue
                obj = json.loads(match.group(0))

            r += 0.5  # Valid JSON!

            # Check has "w" key
            if "w" not in obj:
                rewards.append(r)
                continue
            r += 0.2

            # Check dimension
            w = np.asarray(obj["w"], dtype=np.float32)
            if w.shape != (7,):
                rewards.append(r)
                continue
            r += 0.2

            # Check finite and non-negative
            if not np.all(np.isfinite(w)) or np.any(w < 0):
                rewards.append(r)
                continue
            r += 0.1

            rewards.append(r)  # Perfect format = 1.0

        except:
            rewards.append(0.0)

    return rewards
# -----------------------------
# Main GRPO training
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    # ap.add_argument("--train_jsonl", type=str, default="data_500_dt05_N20_grpo/dataset_all.jsonl")
    # ap.add_argument("--train_instances", type=str, default="data_500_dt05_N20_grpo/instances.dill")
    # ap.add_argument("--train_jsonl", type=str, default="data_10000_dt05_N20_grpo/dataset_all.jsonl")
    # ap.add_argument("--train_instances", type=str, default="data_10000_dt05_N20_grpo/instances.dill")
    ap.add_argument("--train_jsonl", type=str, default="data_10000_dt05_N20_sampling/dataset_all.jsonl")
    ap.add_argument("--train_instances", type=str, default="data_10000_dt05_N20_sampling/instances.dill")


    # ap.add_argument("--base_model", type=str, default="unsloth/Llama-3.2-1B-Instruct")
    ap.add_argument("--base_model", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--peft_model", type=str, default="./sft-pre-grpo-10000-sampling")
    ap.add_argument("--out_dir", type=str, default="./grpo-out-10000-sampling")
    # ap.add_argument("--peft_model", type=str, default="./sft-pre-grpo-10000-ranked")
    # ap.add_argument("--out_dir", type=str, default="./grpo-out-10000-ranked")

    ap.add_argument("--w_dim", type=int, default=7)
    ap.add_argument("--w_clip_lo", type=float, default=0.0)
    ap.add_argument("--w_clip_hi", type=float, default=1e3)
    ap.add_argument("--test_split", type=float, default=0.2)

    # GRPO-specific
    ap.add_argument("--num_generations", type=int, default=8, help="Number of completions per prompt")
    ap.add_argument("--max_completion_length", type=int, default=64)

    # Training
    ap.add_argument("--num_train_epochs", type=int, default=5)
    ap.add_argument("--per_device_train_batch_size", type=int, default=8)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=4)
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=12)

    args = ap.parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- Load and format dataset ----
    ds = load_dataset("json", data_files=args.train_jsonl, split="train")
    ds = ds.train_test_split(test_size=args.test_split, seed=args.seed)
    train_ds = ds["train"]
    eval_ds  = ds["test"]
    w_clip = (args.w_clip_lo, args.w_clip_hi)
    # Build messages from your existing fields
    train_ds = train_ds.map(
        lambda row: make_messages_from_row(row, w_dim=args.w_dim, w_clip=w_clip),
        remove_columns=train_ds.column_names,
    )
    eval_ds = eval_ds.map(
        lambda row: make_messages_from_row(row, w_dim=args.w_dim, w_clip=w_clip),
        remove_columns=eval_ds.column_names,
    )

    # ---- GRPO Config ----
    # Configure training arguments using GRPOConfig
    training_args = GRPOConfig(
        # Training schedule / optimization
        learning_rate=args.lr,  # Learning rate for the optimizer
        # num_train_epochs=args.num_train_epochs,
        max_steps=500,  # Number of dataset passes. For full trainings, use `num_train_epochs` instead

        # Parameters that control GRPO training (you can adapt them)
        per_device_train_batch_size=args.per_device_train_batch_size,
        max_completion_length=args.max_completion_length,  # default: 256               # Max completion length produced during training
        num_generations=args.num_generations,
        # default: 8                         # Number of generations produced during trainig for comparison

        # Optimizations
        optim="paged_adamw_8bit",  # Optimizer
        use_liger_kernel=True,  # Enable Liger kernel optimizations for faster training

        # Parameters related to reporting and saving
        output_dir=args.out_dir,  # Where to save model checkpoints and logs
        logging_steps=args.logging_steps,  # Log training metrics every N steps
        report_to="trackio",  # Experiment tracking tool
        # trackio_space_id=args.out_dir,  # HF Space where the experiment tracking will be saved
        log_completions=False,  # Return model completions during training
        # temperature=0.8,
        # top_p=0.95,

        # Hub integration
        push_to_hub=False,  # Automatically push the trained model to the Hugging Face Hub
        # The model will be saved under your Hub account in the repository named `output_dir`
        # vLLM params
        # use_vllm=False,                                        # Activate vLLM training for faster training
        # vllm_mode='colocate',
        # vllm_gpu_memory_utilization=0.1,
        # vllm_enable_sleep_mode=True
    )
    # ---- GRPO Trainer ----
    adapter_model = args.peft_model
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype="auto", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    fine_tuned_model = PeftModel.from_pretrained(base_model, adapter_model, is_trainable=True)
    fine_tuned_model.train()
    fine_tuned_model.print_trainable_parameters()

    trainer = GRPOTrainer(
        # model=args.base_model,
        model = fine_tuned_model,
        reward_funcs=[format_reward, weights_reward],
        # reward_funcs=[format_reward, weights_reward_ranked],
        # reward_funcs=[strict_json_only_reward],
        # reward_funcs=[strict_json_only_reward, weights_reward_ranked],
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,   # <-- add this
        # peft_config=peft_config,
    )

    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)

    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    trainer_stats = trainer.train()

    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)

    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

    # save the fine-tuned model both locally and to the Hugging Face Hub using the credentials from your account.
    trainer.save_model(args.out_dir)
    trainer.push_to_hub(dataset_name='grpo-500')

if __name__ == "__main__":
    main()