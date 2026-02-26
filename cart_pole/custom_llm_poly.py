# llama_weights_unsloth.py
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import torch

# Unsloth loader
from unsloth import FastLanguageModel


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

# ----------------------------
# 1) Prompting (system + user)
# ----------------------------

SYSTEM_MSG = (
    "You are an expert in nonlinear control and model predictive control (MPC).\n"
    "You will propose MPC cost weights.\n\n"
    "CRITICAL OUTPUT RULE:\n"
    "- Return ONLY valid JSON and nothing else.\n"
    "- Output must be exactly one JSON object with key \"w\".\n"
    "- \"w\" must be a list of finite numbers.\n"
    "- All numbers must be >= 0.\n"
    "- Length must match the requested dimension.\n"
    "- No extra keys, no markdown, no prose.\n"
)

def build_user_msg(prompt_sys: str, prompt_score: str, w_dim: int = 7) -> str:
    # Adjust semantics if your cost changes ordering
    semantics = (
        f"Weight vector format (length {w_dim}):\n"
        "w = [Ws_track, Ws_safe, Ws_u, Ws_smooth, Wt_track, Wt_safe, Wt_u]\n"
        "- Ws_* are stage (running) weights.\n"
        "- Wt_* are terminal weights.\n"
        "- track penalizes ||x - x_goal||^2\n"
        "- safe penalizes max(0, ||x||_inf - R_safe)^2\n"
        "- u penalizes ||u||^2 normalized by u_max\n"
        "- smooth penalizes ||u - u_prev||^2 (stage only)\n"
    )

    return (
        "INSTANCE SUMMARY:\n"
        f"{prompt_sys}\n\n"
        "SCORE PREFERENCES:\n"
        f"{prompt_score}\n\n"
        f"{semantics}\n"
        "TASK:\n"
        "Choose w to maximize the composite score.\n"
        "Hints:\n"
        "- Terminal track/safe weights are usually larger than stage weights.\n"
        "- If safety matters more, increase safe weights.\n"
        "- If effort/smoothness matters more, increase Ws_u / Ws_smooth.\n\n"
        "OUTPUT FORMAT (STRICT):\n"
        f"Return ONLY JSON: {{\"w\": [..{w_dim} numbers..]}}\n"
        "Example:\n"
        "{\"w\": [1.0, 5.0, 0.1, 0.1, 10.0, 20.0, 0.1]}\n"
    )

# def build_user_msg_with_limits(prompt_sys: str, prompt_score: str, rho_goal: float, w_dim: int = 7) -> str:
#     lo, hi = get_weight_bounds_from_sampler(rho_goal=rho_goal, w_dim=w_dim)
#     names = ["Ws_track","Ws_safe","Ws_u","Ws_smooth","Wt_track","Wt_safe","Wt_u"]
#
#     bounds_lines = "\n".join(
#         [f"- {names[i]} must be in [{lo[i]:.6g}, {hi[i]:.6g}]" for i in range(w_dim)]
#     )
#
#     semantics = (
#         f"Weight vector format (length {w_dim}):\n"
#         "w = [Ws_track, Ws_safe, Ws_u, Ws_smooth, Wt_track, Wt_safe, Wt_u]\n"
#         "- Ws_* are stage weights; Wt_* are terminal weights.\n"
#         "- track penalizes ||x - x_goal||^2\n"
#         "- safe penalizes max(0, ||x||_inf - R_safe)^2\n"
#         "- u penalizes ||u||^2 normalized by u_max\n"
#         "- smooth penalizes ||u - u_prev||^2 (stage only)\n"
#     )
#
#     return (
#         "INSTANCE SUMMARY:\n"
#         f"{prompt_sys}\n\n"
#         "SCORE PREFERENCES:\n"
#         f"{prompt_score}\n\n"
#         f"{semantics}\n"
#         "HARD CONSTRAINTS (IMPORTANT):\n"
#         f"{bounds_lines}\n\n"
#         "TASK:\n"
#         "Choose w to maximize the composite score.\n\n"
#         "OUTPUT FORMAT (STRICT):\n"
#         f"Return ONLY JSON: {{\"w\": [..{w_dim} numbers..]}}\n"
#         "No extra keys, no markdown, no prose.\n"
#     )

# ----------------------------
# 2) Robust JSON parsing
# ----------------------------

def _extract_json_obj(text: str) -> Dict[str, Any]:
    text = (text or "").strip()

    # Direct parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Find first {...} block
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError(f"No JSON object found. Raw:\n{text[:1200]}")
    return json.loads(m.group(0))

def parse_and_sanitize_w(
    text: str,
    w_dim: int = 7,
    w_clip: Tuple[float, float] = (0.0, 1e6),
) -> np.ndarray:
    obj = _extract_json_obj(text)
    if "w" not in obj:
        raise ValueError(f"Missing key 'w'. Keys={list(obj.keys())}")

    w = np.asarray(obj["w"], dtype=float).reshape(-1)
    if w.size != w_dim:
        raise ValueError(f"Expected {w_dim} weights, got {w.size}")
    if not np.all(np.isfinite(w)):
        raise ValueError("Non-finite weights")

    lo, hi = w_clip
    w = np.clip(w, lo, hi)
    return w

def parse_and_validate_w(
    text: str,
    rho_goal: float,
    w_dim: int = 7,
    strict: bool = True,
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

    lo, hi = get_weight_bounds_from_sampler(rho_goal=rho_goal, w_dim=w_dim)

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


# ----------------------------
# 3) Unsloth generation
# ----------------------------

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
    # seed: Optional[int] = None,
    rho_goal: float,
) -> str:
    # if seed is not None:
    #     torch.manual_seed(seed)

    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": build_user_msg(prompt_sys, prompt_score, w_dim=w_dim)},
    ]

    # Use chat template (important for Instruct Llama)
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    do_sample = temperature > 0.0

    out = model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        eos_token_id=tokenizer.eos_token_id,
        # seed = seed,
    )

    text = tokenizer.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True).strip()
    return text

def main():
    # ----- one example prompt pair -----
    sys_prompt = (
        "INSTANCE SUMMARY (interpretable)\n"
        "- Dimensions: n=4 (state dim), m=2 (control dim)\n"
        "- Discretization: dt=0.1, H=4, T_mpc=100\n"
        "- Constraints/region: u_max=3.0 (|u|<=u_max), R_safe=2.0, rho_goal=0.02\n"
        "- Goal difficulty: d0=||x0-x_goal||2=3.5, d0_norm=0.425\n"
        "- Linear stability: rho_lin=max|eig(I+dt*A)|=0.92\n"
        "- Control authority at x=0: g_norm=||G0||2=1.30\n"
        "- Nonlinearity level: eta_nl=2.80\n"
        "- Reachability ratio: reach_ratio=1.70\n"
    )

    score_prompt = (
        "SCORE SUMMARY\n"
        "- Composite score Y = weighted average of four sub-scores in [0,1]\n"
        "- beta_margin=50.0 (prefer reaching target set)\n"
        "- beta_time=1.0 (prefer reaching earlier)\n"
        "- beta_u=16.0 (prefer low control effort)\n"
        "- beta_du=18.0 (prefer smooth controls)\n"
    )

    # ----- model config -----
    model_name = "unsloth/Llama-3.2-3B-Instruct"
    max_seq_len = 4096
    load_in_4bit = True

    # generation config
    w_dim = 7
    num = 3
    max_new_tokens = 128
    temperature = 0.1
    top_p = 0.9

    # ----- load model -----
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_len,
        dtype=None,
        load_in_4bit=load_in_4bit,
    )
    FastLanguageModel.for_inference(model)

    rho_goal = 0.2

    # ----- run a few samples -----
    for i in range(num):
        raw = generate_once(
            model=model,
            tokenizer=tokenizer,
            prompt_sys=sys_prompt,
            prompt_score=score_prompt,
            w_dim=w_dim,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            rho_goal = rho_goal,
        )
        print("\n--- raw output ---")
        print(raw)

        try:
            w = parse_and_validate_w(raw, rho_goal=rho_goal, w_dim=w_dim, strict=True)
            # w = parse_and_sanitize_w(raw, w_dim=w_dim)
            print("--- parsed w ---")
            print(w)
        except Exception as e:
            print("[parse failed]", e)


if __name__ == "__main__":
    main()


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--model", default="unsloth/Llama-3.2-3B-Instruct")
#     ap.add_argument("--max_seq_len", type=int, default=4096)
#     ap.add_argument("--load_in_4bit", action="store_true", help="Use 4-bit quantization (faster/less VRAM).")
#
#     ap.add_argument("--prompt_sys_path", required=True, type=str)
#     ap.add_argument("--prompt_score_path", required=True, type=str)
#
#     ap.add_argument("--w_dim", type=int, default=7)
#     ap.add_argument("--num", type=int, default=5)
#     ap.add_argument("--max_new_tokens", type=int, default=128)
#     ap.add_argument("--temperature", type=float, default=0.2)
#     ap.add_argument("--top_p", type=float, default=0.9)
#
#     ap.add_argument("--out_dir", type=str, default="samples_weights")
#     ap.add_argument("--seed", type=int, default=0, help="Base seed; each variant uses seed+idx.")
#     args = ap.parse_args()
#
#     prompt_sys = Path(args.prompt_sys_path).read_text(encoding="utf-8").strip()
#     prompt_score = Path(args.prompt_score_path).read_text(encoding="utf-8").strip()
#
#     # Load model using Unsloth
#     model, tokenizer = FastLanguageModel.from_pretrained(
#         model_name=args.model,
#         max_seq_length=args.max_seq_len,
#         dtype=None,                 # let Unsloth choose
#         load_in_4bit=args.load_in_4bit,
#     )
#     FastLanguageModel.for_inference(model)
#
#     out_dir = Path(args.out_dir)
#     out_dir.mkdir(parents=True, exist_ok=True)
#
#     for i in range(args.num):
#         # Generate
#         raw = generate_once(
#             model=model,
#             tokenizer=tokenizer,
#             prompt_sys=prompt_sys,
#             prompt_score=prompt_score,
#             w_dim=args.w_dim,
#             max_new_tokens=args.max_new_tokens,
#             temperature=args.temperature,
#             top_p=args.top_p,
#             seed=(args.seed + i) if args.seed is not None else None,
#         )
#
#         # Save raw
#         (out_dir / f"raw_v{i+1}.txt").write_text(raw, encoding="utf-8")
#
#         # Parse + save JSON
#         try:
#             w = parse_and_sanitize_w(raw, w_dim=args.w_dim)
#             obj = {"w": w.tolist()}
#             (out_dir / f"weights_v{i+1}.json").write_text(json.dumps(obj), encoding="utf-8")
#             print(f"[OK] v{i+1} w={np.array2string(w, precision=3, suppress_small=True)}")
#         except Exception as e:
#             (out_dir / f"error_v{i+1}.txt").write_text(f"{type(e).__name__}: {e}\n\nRAW:\n{raw}", encoding="utf-8")
#             print(f"[FAIL] v{i+1}: {e}")
#
#     print(f"Done. Files saved to {out_dir}")
#
#
# if __name__ == "__main__":
#     main()

