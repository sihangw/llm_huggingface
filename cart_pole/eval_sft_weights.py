# eval_weights_unsloth.py
from __future__ import annotations

import argparse, json, re
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import torch
from transformers import TextStreamer
from unsloth import FastLanguageModel


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

def _get_eos_ids(tokenizer):
    ids = {tokenizer.eos_token_id}
    for tok in ["<|eot_id|>", "<|eom_id|>", "<|end|>", "<|im_end|>"]:
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is not None and tid != tokenizer.unk_token_id and tid != -1:
            ids.add(tid)
    return list(ids)

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

def parse_and_sanitize_w(
    text: str,
    w_dim: int,
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

def build_inputs(tokenizer, prompt_sys: str, prompt_score: str, w_dim: int, device):
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": build_user_msg(prompt_sys, prompt_score, w_dim=w_dim)},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)
    return inputs

@torch.inference_mode()
def generate_once(
    model,
    tokenizer,
    eos_ids,
    prompt_sys: str,
    prompt_score: str,
    w_dim: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    seed: Optional[int] = None,
) -> str:
    if seed is not None:
        torch.manual_seed(seed)

    inputs = build_inputs(tokenizer, prompt_sys, prompt_score, w_dim=w_dim, device=model.device)
    input_len = inputs.shape[1]

    out_ids = model.generate(
        input_ids=inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        use_cache=True,
        eos_token_id=eos_ids,
        pad_token_id=tokenizer.pad_token_id,
    )

    gen_ids = out_ids[0, input_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

def run_file(
    model,
    tokenizer,
    eos_ids,
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
            print(rho_sys)

            # allow older style: pack both into one "prompt" if needed
            if prompt_sys is None and "prompt" in ex:
                prompt_sys = ex["prompt"]
            if prompt_sys is None or prompt_score is None:
                # skip quietly (or raise)
                continue

            raw = generate_once(
                model=model,
                tokenizer=tokenizer,
                eos_ids=eos_ids,
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, default="./sft-weights-out")
    ap.add_argument("--max_seq_len", type=int, default=2048)

    # generation
    ap.add_argument("--w_dim", type=int, default=7)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.1)  # default deterministic
    ap.add_argument("--top_p", type=float, default=0.0)
    ap.add_argument("--do_sample", action="store_true", help="Enable sampling (default off)")
    ap.add_argument("--seed", type=int, default=None)

    # sanitize range
    ap.add_argument("--w_clip_lo", type=float, default=0.0)
    ap.add_argument("--w_clip_hi", type=float, default=1e6)

    # dataset mode
    ap.add_argument("--input_jsonl", type=str, default="data_100_dt01_N100/dataset_all.jsonl")
    ap.add_argument("--output_jsonl", type=str, default="data_100_dt01_N100/test_weights_pred.jsonl")

    # interactive
    ap.add_argument("--interactive", action="store_true")
    args = ap.parse_args()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=args.max_seq_len,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    eos_ids = _get_eos_ids(tokenizer)
    w_clip = (args.w_clip_lo, args.w_clip_hi)

    # ---------- Dataset mode ----------
    if (not args.interactive) and args.input_jsonl:
        run_file(
            model=model,
            tokenizer=tokenizer,
            eos_ids=eos_ids,
            inp_path=args.input_jsonl,
            out_path=args.output_jsonl,
            w_dim=args.w_dim,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.do_sample,
            w_clip=w_clip,
        )
        return

    # ---------- Interactive mode ----------
    print("Interactive mode.")
    print("Paste prompt_sys then prompt_score. Commands: /exit")
    while True:
        try:
            prompt_sys = input("\nPROMPT_SYS> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not prompt_sys or prompt_sys.lower() == "/exit":
            print("Bye.")
            break
        prompt_score = input("PROMPT_SCORE> ").strip()
        if not prompt_score:
            print("Need prompt_score.")
            continue

        raw = generate_once(
            model=model,
            tokenizer=tokenizer,
            eos_ids=eos_ids,
            prompt_sys=prompt_sys,
            prompt_score=prompt_score,
            w_dim=args.w_dim,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.do_sample,
            seed=args.seed,
        )
        print("\nRAW:")
        print(raw)

        try:
            w = parse_and_sanitize_w(raw, w_dim=args.w_dim, w_clip=w_clip)
            print("PARSED w:", w)
        except Exception as e:
            print("[parse failed]", e)

if __name__ == "__main__":
    main()
