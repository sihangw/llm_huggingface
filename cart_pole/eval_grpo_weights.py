import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # must be set before torch initializes CUDA
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
torch.cuda.set_device(0)
import argparse, json, re, time
from datasets import load_dataset
from typing import List, Dict, Any, Tuple, Optional
# from vllm import LLM, SamplingParams

from ilqr_mpc import PolyDynamicsWithJac, QuadCostConfig
from instance_sampler import ExperimentSampler, ExperimentSamplerConfig
from weights_search_ilqr import run_weight_search_for_instance_fast_ilqr, make_fast_cost_from_w
from CompositeScore import CompositeScoreConfig, CompositeTrajectoryScorer
from instance_prompt import build_instance_prompt, build_beta_prompt_simple
from ilqr_mpc import ILQRMPCConfig, FastILQRMPC, run_closed_loop_fast_ilqr_mpc, FastQuadraticCost


# SYSTEM_MSG = (
#     "You are an expert in nonlinear control and MPC."
#     "You MUST return ONLY valid JSON and nothing else."
#     "No explanation, no markdown, no code blocks, no extra text."
#     "Your ENTIRE response must be exactly this format:"
#     '{"w": [w1, w2, w3, w4, w5, w6, w7]}'
#     "Where w1-w7 are non-negative numbers."
#     "DO NOT add any text before or after the JSON."
# )

SYSTEM_MSG = """
Respond in the following format:
<answer>{\"w\": [w1, w2, w3, w4, w5, w6, w7]}</answer>
Where w1-w7 are non-negative numbers.
The <answer> section MUST contain valid JSON only.
"""

def build_user_msg(prompt_sys: str, prompt_score: str, w_dim: int) -> str:
    semantics = (
        f"w = [Ws_track, Ws_safe, Ws_u, Ws_smooth, Wt_track, Wt_safe, Wt_u]"
        f"All {w_dim} values must be >= 0."
    )
    return (
        # "INSTANCE SUMMARY:"
        f"{prompt_sys}"
        # "SCORE PREFERENCES:"
        f"{prompt_score}"
        "TASK: Choose w to maximize the score. "
        f"{semantics}"
        "OUTPUT FORMAT (STRICT):"
        f"Return ONLY JSON: {{\"w\": [..{w_dim} numbers..]}}"
        # "Example:\n"
        # "{\"w\": [1.0, 5.0, 0.1, 0.1, 10.0, 20.0, 0.1]}\n"
    )

def _pow10(x: float) -> float:
    return float(10.0 ** x)

def get_weight_bounds_from_sampler(
    rho_goal: float,
    w_dim: int = 7,
    rho_track_ref: float = 0.2,
    rho_track_clip: Tuple[float, float] = (0.5, 5.0),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bounds implied by your sampling:
      Ws_track  ~ 10^[-1.5, 2] * clip(0.2/rho, 0.5, 5)
      Ws_safe   ~ 10^[-1.5, 2]
      Ws_u      ~ 10^[-3, 0.5]
      Ws_smooth ~ 10^[-3, 0.5]
      Wt_track  = Ws_track * 10^[0.5, 1.5]
      Wt_safe   = Ws_safe  * 10^[0.5, 1.5]
      Wt_u      = Ws_u     * 10^[-0.5, 0.5]

    Returns (lo, hi) arrays of length 7 in the same order:
      [Ws_track, Ws_safe, Ws_u, Ws_smooth, Wt_track, Wt_safe, Wt_u]
    """
    if w_dim != 7:
        raise ValueError("This bound helper is defined for w_dim=7.")

    # Base stage ranges (from your code)
    Ws_track_lo, Ws_track_hi = _pow10(-1.5), _pow10(2.0)    # [0.0316, 100]
    Ws_safe_lo,  Ws_safe_hi  = _pow10(-1.5), _pow10(2.0)
    Ws_u_lo,     Ws_u_hi     = _pow10(-3.0), _pow10(0.5)    # [0.001, 3.162]
    Ws_sm_lo,    Ws_sm_hi    = _pow10(-3.0), _pow10(0.5)

    # Your rho scaling affects Ws_track only (bounded)
    if rho_goal is None or rho_goal <= 0:
        rho_scale = 1.0
    else:
        rho_scale = float(np.clip(rho_track_ref / float(rho_goal), rho_track_clip[0], rho_track_clip[1]))

    Ws_track_lo *= rho_scale
    Ws_track_hi *= rho_scale

    # Terminal multipliers
    mult_ts_lo, mult_ts_hi = _pow10(0.5), _pow10(1.5)       # [3.162, 31.62]
    mult_u_lo,  mult_u_hi  = _pow10(-0.5), _pow10(0.5)      # [0.316, 3.162]

    Wt_track_lo, Wt_track_hi = Ws_track_lo * mult_ts_lo, Ws_track_hi * mult_ts_hi
    Wt_safe_lo,  Wt_safe_hi  = Ws_safe_lo  * mult_ts_lo, Ws_safe_hi  * mult_ts_hi
    Wt_u_lo,     Wt_u_hi     = Ws_u_lo     * mult_u_lo,  Ws_u_hi     * mult_u_hi

    lo = np.array([Ws_track_lo, Ws_safe_lo, Ws_u_lo, Ws_sm_lo, Wt_track_lo, Wt_safe_lo, Wt_u_lo], dtype=float)
    hi = np.array([Ws_track_hi, Ws_safe_hi, Ws_u_hi, Ws_sm_hi, Wt_track_hi, Wt_safe_hi, Wt_u_hi], dtype=float)
    return lo, hi

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

# ---------- robust JSON parsing ----------
def _extract_json_obj(text: str) -> Dict[str, Any]:
    text = (text or "").strip()

    # direct parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # find first {...}
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError(f"No JSON object found. Raw:\n{text[:1200]}")
    return json.loads(m.group(0))

def parse_and_validate_w(
    text: str,
    rho_goal: float,
    w_dim: int = 7,
    strict: bool = False,
) -> np.ndarray:
    """
    If strict=True: reject if any weight is out of bounds.
    If strict=False: clip to bounds.
    """
    obj = _extract_json_obj(text)
    if "w" not in obj:
        raise ValueError(f"Missing key 'w'. Keys={list(obj.keys())}")

    w = np.asarray(obj["w"], dtype=float).reshape(-1)
    if w.size != w_dim:
        raise ValueError(f"Expected {w_dim} weights, got {w.size}")
    if not np.all(np.isfinite(w)):
        raise ValueError("Non-finite weights")

    # lo, hi = get_weight_bounds_from_sampler(rho_goal=rho_goal, w_dim=w_dim)

    lo = np.ones(7, dtype=float) * 1e-3
    hi = np.ones(7, dtype=float) * 1e3

    if strict:
        bad = np.where((w < lo) | (w > hi))[0]
        if bad.size > 0:
            i = int(bad[0])
            raise ValueError(
                f"Weight w[{i}]={w[i]:.6g} out of bounds [{lo[i]:.6g}, {hi[i]:.6g}]"
            )
        return w

    # non-strict: clip
    return np.minimum(np.maximum(w, lo), hi)

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

@torch.inference_mode()
def generate_once(
    model,
    tokenizer,
    prompt_sys: str,
    prompt_score: str,
    w_dim: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    # seed: Optional[int] = None,
) -> str:
    # if seed is not None:
    #     torch.manual_seed(seed)
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": build_user_msg(prompt_sys, prompt_score, w_dim=w_dim)},
    ]
    text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        # do_sample=do_sample,
        # temperature=temperature if do_sample else None,
        # top_p=top_p if do_sample else None,
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
    # Decode and extract model response
    generated_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    return generated_text


def run_file(
    model,
    tokenizer,
    inp_path: str,
    out_path: str,
    w_dim: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    w_clip: Tuple[float, float],
):
    with open(inp_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        n = 0
        for line in fin:
            line = line.strip()
            if not line:
                continue

            ex = json.loads(line)

            # expected keys from your dataset
            prompt_sys = ex.get("prompt_sys")
            prompt_score = ex.get("prompt_score")
            # rho is the 8th element of feat_vec
            rho_sys = ex.get("feat_vec")[7]

            # allow older style: pack both into one "prompt" if needed
            if prompt_sys is None and "prompt" in ex:
                prompt_sys = ex["prompt"]
            if prompt_sys is None or prompt_score is None:
                # skip quietly (or raise)
                continue

            raw = generate_once(
                model=model,
                tokenizer=tokenizer,
                prompt_sys=prompt_sys,
                prompt_score=prompt_score,
                w_dim=w_dim,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
            )

            out_row = dict(ex)  # keep original fields
            out_row["raw_model_output"] = raw
            print(raw)

            try:
                # w = parse_and_sanitize_w(raw, w_dim=w_dim, w_clip=w_clip)
                w = parse_and_validate_w(raw, rho_goal = rho_sys, w_dim=w_dim)
                out_row["w_pred"] = w.tolist()
                out_row["parse_ok"] = True
            except Exception as e:
                out_row["w_pred"] = None
                out_row["parse_ok"] = False
                out_row["parse_error"] = str(e)[:300]

            fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            n += 1

    print(f"Generated {n} predictions → {out_path}")

def evaluate_llm_vs_oracle(
    sampler,
    num_trials: int,
    out_dir: str,
    run_weight_search_for_instance_fast_ilqr,
    run_closed_loop_fast_ilqr_mpc,
    QuadCostConfig,
    FastQuadraticCost,
    CompositeScoreConfig,
    CompositeTrajectoryScorer,
    ILQRMPCConfig,
    PolyDynamicsWithJac,
    FastILQRMPC,
    build_instance_prompt,
    build_beta_prompt_simple,
    # ---- LLM hook ----
    model,
    tokenizer,
    # eos_ids,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    # llm_predict_w: Callable[[str, str, int, float], np.ndarray],
    w_dim: int = 7,
    # ---- fixed search params ----
    oracle_search: bool = False,
    keep_top: int = 5,
    early_stop_Y: float = 0.99,
    verbose: bool = False,
    time_every: int = 10,
    save_every: int = 1,
) -> Dict[str, str]:
    """
    For each trial:
      - sample (inst, params)
      - build prompts + numeric features
      - get oracle weights via your weight-search => best_Y, best_w
      - get llm weights via llm_predict_w(prompt_sys, prompt_score, w_dim, rho_goal)
      - score llm weights by running ONE closed-loop MPC rollout (no weight search)
      - store everything to JSONL, plus dense arrays to NPY
    """
    os.makedirs(out_dir, exist_ok=True)
    jsonl_path = os.path.join(out_dir, "eval.jsonl")

    # Dense arrays for training/analysis
    X_list, Y_llm_list = [], []
    W_llm_list, succ_llm_list = [], []
    Y_oracle_list, W_oracle_list, succ_oracle_list = [], [], []

    t_start = time.perf_counter()
    t_last = t_start

    mode = "w"
    with open(jsonl_path, mode, encoding="utf-8") as f:
        for i in range(num_trials):
            t0 = time.perf_counter()

            trial = sampler.sample_trial()
            inst = trial["inst"]
            p = trial["params"]

            # ----- prompts + features -----
            feat_sys, prompt_sys = build_instance_prompt(
                inst,
                H=int(p["H"]),
                T_mpc=int(p["T_outer"]),
                R_safe=p.get("R_safe", None),
            )
            feat_beta, prompt_beta = build_beta_prompt_simple(
                float(p["beta_margin"]), float(p["beta_time"]), float(p["beta_u"]), float(p["beta_du"])
            )
            feat_vec = np.array(list(feat_sys.values()) + list(feat_beta.values()), dtype=float)
            X_list.append(feat_vec)

            # ----- scorer (depends on betas + goal) -----
            x_0 = np.asarray(inst["init_goal"]["x0"], dtype=float)
            x_goal = np.asarray(inst["init_goal"]["x_end"], dtype=float)
            rho = float(inst["init_goal"]["goal_radius"])
            u_max = float(inst["meta"]["u_max"])

            def phi_target(x):
                return np.linalg.norm(x - x_goal) - rho  # negative inside target

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

            # ----- MPC config for evaluation -----
            cfg_ilqr = ILQRMPCConfig(
                H=int(p["H"]),
                max_iters=int(p["max_iters"]),
                u_min=-u_max,
                u_max=u_max,
                shift_fill="zero",
            )

            def mpc_builder(inst_local, cost_obj, cfg_ilqr_local):
                dyn = PolyDynamicsWithJac(cfg=inst_local["cfg"], drift=inst_local["drift"], inp=inst_local["input"])
                return FastILQRMPC(dyn=dyn, cost=cost_obj, cfg=cfg_ilqr_local)

            # # =========================
            # # (A) ORACLE: weight search
            # # =========================
            if oracle_search:
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
                    keep_top=int(p.get("keep_top", keep_top)),
                    early_stop_Y=float(p.get("early_stop_Y", early_stop_Y)),
                    verbose=verbose,
                )

                Y_oracle = float(best["Y"])
                w_oracle = np.asarray(best["w"], dtype=float).reshape(-1)
                succ_oracle = bool(best.get("success", Y_oracle > 0.5))  # fallback if your best dict lacks success

            # =========================
            # (B) LLM: predict weights
            # =========================
            raw = generate_once(
                model=model,
                tokenizer=tokenizer,
                # eos_ids=eos_ids,
                prompt_sys=prompt_sys,
                prompt_score=prompt_beta,
                w_dim=w_dim,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
            )
            print(raw)
            rho_goal = float(inst["meta"]["rho"])  # or goal tolerance used in your bound logic

            w_llm = None  # numeric np.ndarray or None
            w_llm_raw = None  # whatever we extracted before casting (list / mixed)
            raw_text = raw  # keep raw model output no matter what

            try:
                # If you already have parse_and_validate_w -> it should RETURN numeric ndarray.
                # But if it might still choke due to strings, catch below.
                w = parse_and_validate_w(raw_text, rho_goal=rho_goal, w_dim=w_dim)

                # Ensure ndarray float
                w_llm = np.asarray(w, dtype=float).reshape(-1)
                if w_llm.shape[0] != w_dim or (not np.all(np.isfinite(w_llm))):
                    raise ValueError("Parsed w not finite or wrong shape.")

                # Score it
                cost_obj = make_fast_cost_from_w(inst, w_llm, QuadCostConfig, FastQuadraticCost)
                mpc_llm = mpc_builder(inst, cost_obj, cfg_ilqr)
                X_llm, U_llm = run_closed_loop_fast_ilqr_mpc(
                    mpc=mpc_llm,
                    step_fn_true=inst["step"],
                    x0=x_0,
                    T_outer=int(p["T_outer"]),
                )
                score_dict = scorer.score(X_llm, U_llm)
                Y_llm = float(score_dict["score"])
                succ_llm = bool(score_dict["success"])
                llm_valid = True

            except Exception as e:
                # If parse failed, try to *salvage* a raw "w" (even if strings),
                # so you can store it for debugging.
                llm_valid = False
                Y_llm = 0.0
                succ_llm = False

                # Attempt salvage: extract JSON and store obj["w"] as-is
                try:
                    obj = _extract_json_obj(raw_text)  # your helper
                    w_llm_raw = obj.get("w", None)  # might be list w/ strings
                except Exception:
                    w_llm_raw = None

                print("[LLM parse/score failed]", repr(e))

            datum = {
                "trial_id": i,
                "params": p,
                "prompt_sys": prompt_sys,
                "prompt_score": prompt_beta,
                "feat_vec": feat_vec.tolist(),
                "llm": {
                    "valid": llm_valid,
                    "Y": Y_llm,
                    "success": succ_llm,
                    "w": (w_llm.tolist() if w_llm is not None else None),  # numeric if valid
                    "w_raw": w_llm_raw,  # possibly strings
                    "raw_text": raw_text,  # always keep
                },
            }

            # Save JSONL
            if save_every > 0 and ((i + 1) % save_every == 0):
                f.write(json.dumps(datum) + "\n")
                f.flush()

            # Save dense arrays
            Y_llm_list.append(Y_llm)
            W_llm_list.append(w_llm)
            succ_llm_list.append(succ_llm)
            if oracle_search:
                Y_oracle_list.append(Y_oracle)
                W_oracle_list.append(w_oracle)
                succ_oracle_list.append(succ_oracle)

            # Timing
            t1 = time.perf_counter()
            dt_trial = t1 - t0
            if time_every and ((i + 1) % time_every == 0):
                elapsed = t1 - t_start
                avg = elapsed / (i + 1)
                since_last = t1 - t_last
                avg_block = since_last / time_every
                remain = avg * (num_trials - (i + 1))
                print(
                    f"[timing] {i+1}/{num_trials} | trial={dt_trial:.3f}s | "
                    f"avg={avg:.3f}s | avg(last {time_every})={avg_block:.3f}s | "
                    f"elapsed={elapsed:.1f}s | ETA≈{remain:.1f}s"
                )
                t_last = t1

    # ---- write NPYs ----
    # X = np.asarray(X_list, dtype=float)
    # Y_llm_arr = np.asarray(Y_llm_list, dtype=float)
    # S_llm = np.asarray(succ_llm_list, dtype=bool)

    # Build fixed-shape W_llm with NaNs, plus a validity mask
    W_llm = np.full((len(W_llm_list), w_dim), np.nan, dtype=float)
    W_llm_valid = np.zeros((len(W_llm_list),), dtype=bool)

    for i, w in enumerate(W_llm_list):
        # Accept: np.ndarray/list/tuple of correct length and numeric
        try:
            if w is None:
                continue
            w_arr = np.asarray(w, dtype=float).reshape(-1)
            if w_arr.size != w_dim:
                continue
            if not np.all(np.isfinite(w_arr)):
                continue
            W_llm[i, :] = w_arr
            W_llm_valid[i] = True
        except Exception:
            # trash -> leave as NaNs
            continue


    # Y_oracle_arr = np.asarray(Y_oracle_list, dtype=float)
    # W_oracle = np.asarray(W_oracle_list, dtype=float)
    # S_oracle = np.asarray(succ_oracle_list, dtype=bool)

    # npy_paths = {
    #     "X": os.path.join(out_dir, "X.npy"),
    #     "Y_llm": os.path.join(out_dir, "Y_llm.npy"),
    #     "W_llm": os.path.join(out_dir, "W_llm.npy"),
    #     "W_llm_valid": os.path.join(out_dir, "W_llm_valid.npy"),
    #     "S_llm": os.path.join(out_dir, "S_llm.npy"),
    # }

    if oracle_search:
        npy_paths = {
            "X": os.path.join(out_dir, "X.npy"),
            "Y_oracle": os.path.join(out_dir, "Y_oracle.npy"),
            "Y_llm": os.path.join(out_dir, "Y_llm.npy"),
            "W_oracle": os.path.join(out_dir, "W_oracle.npy"),
            "W_llm": os.path.join(out_dir, "W_llm.npy"),
            "W_llm_valid": os.path.join(out_dir, "W_llm_valid.npy"),
            "S_oracle": os.path.join(out_dir, "S_oracle.npy"),
            "S_llm": os.path.join(out_dir, "S_llm.npy"),
        }
        Y_oracle_arr = np.asarray(Y_oracle_list, dtype=float)
        W_oracle = np.asarray(W_oracle_list, dtype=float)
        S_oracle = np.asarray(succ_oracle_list, dtype=bool)
        np.save(npy_paths["Y_oracle"], Y_oracle_arr)
        np.save(npy_paths["W_oracle"], W_oracle)
        np.save(npy_paths["S_oracle"], S_oracle)

    else:
        npy_paths = {
            "X": os.path.join(out_dir, "X.npy"),
            "Y_llm": os.path.join(out_dir, "Y_llm.npy"),
            "W_llm": os.path.join(out_dir, "W_llm.npy"),
            "W_llm_valid": os.path.join(out_dir, "W_llm_valid.npy"),
            "S_llm": os.path.join(out_dir, "S_llm.npy"),
        }

    X = np.asarray(X_list, dtype=float)
    Y_llm_arr = np.asarray(Y_llm_list, dtype=float)
    # W_llm = np.asarray(W_llm_list, dtype=float)
    S_llm = np.asarray(succ_llm_list, dtype=bool)
    # W_llm_valid = np.asarray(W_llm_valid, dtype=bool)
    np.save(npy_paths["X"], X)
    np.save(npy_paths["Y_llm"], Y_llm_arr)
    np.save(npy_paths["W_llm"], W_llm)
    np.save(npy_paths["S_llm"], S_llm)
    np.save(npy_paths["W_llm_valid"], S_llm)

    print(f"[done] wrote {jsonl_path} and npys to {out_dir}")
    return {"jsonl": jsonl_path, **npy_paths}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    # ap.add_argument("--peft_model", type=str, default="./grpo-out-500")
    ap.add_argument("--peft_model", type=str, default="./grpo-out-10000-sampling")
    # ap.add_argument("--peft_model", type=str, default="./grpo-out-10000-ranked")
    # ap.add_argument("--peft_model", type=str, default="./sft-pre-grpo-500")

    # ap.add_argument("--out_dir", type=str, default="./grpo-out")
    ap.add_argument("--w_clip_lo", type=float, default=0.0)
    ap.add_argument("--w_clip_hi", type=float, default=1e6)
    ap.add_argument("--test_split", type=float, default=0.2)

    # GRPO generation
    ap.add_argument("--w_dim", type=int, default=7)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.1)  # default deterministic
    ap.add_argument("--top_p", type=float, default=0.0)
    ap.add_argument("--do_sample", action="store_true", help="Enable sampling (default off)")

    # Training
    ap.add_argument("--num_train_epochs", type=int, default=3)
    ap.add_argument("--per_device_train_batch_size", type=int, default=2)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=0)

    # dataset mode
    ap.add_argument("--input_jsonl", type=str, default="data_10000_dt05_N20_sampling/dataset_all.jsonl")
    ap.add_argument("--output_dir", type=str, default="grpo_eval_scores_sampling")

    args = ap.parse_args()

    adapter_model = args.peft_model
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype="auto", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    fine_tuned_model = PeftModel.from_pretrained(base_model, adapter_model)

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

    # ---- run eval: 20 new trials ----
    paths = evaluate_llm_vs_oracle(
        sampler=sampler,
        num_trials=10000,
        out_dir=args.output_dir,
        # your pipeline funcs/classes:
        run_weight_search_for_instance_fast_ilqr=run_weight_search_for_instance_fast_ilqr,
        run_closed_loop_fast_ilqr_mpc=run_closed_loop_fast_ilqr_mpc,
        QuadCostConfig=QuadCostConfig,
        FastQuadraticCost=FastQuadraticCost,
        CompositeScoreConfig=CompositeScoreConfig,
        CompositeTrajectoryScorer=CompositeTrajectoryScorer,
        ILQRMPCConfig=ILQRMPCConfig,
        PolyDynamicsWithJac=PolyDynamicsWithJac,
        FastILQRMPC=FastILQRMPC,
        build_instance_prompt=build_instance_prompt,
        build_beta_prompt_simple=build_beta_prompt_simple,
        # LLM hook:
        model=fine_tuned_model,
        tokenizer=tokenizer,
        # eos_ids=eos_ids,
        max_new_tokens=128,
        temperature=0.1,
        top_p=0.0,
        do_sample=False,
        w_dim=7,
        oracle_search=False,
        keep_top=5,
        early_stop_Y=0.99,
        verbose=True,
        time_every=5,
        save_every=1,
    )

    print(paths)  # shows jsonl + npy paths

if __name__ == "__main__":
    main()