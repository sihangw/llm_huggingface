#!/usr/bin/env python3
import os
import unsloth
import argparse, torch
from transformers import TextStreamer
from unsloth import FastLanguageModel

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

SYSTEM_PROMPT = r"""
You convert a mission into ONE line of JSON with keys: "objective_function", "constraints", "areas".

RULES (must follow):
- Output exactly one JSON line. No extra text.
- "objective_function": short LaTeX-like string; balanced () and {}; no "||".
- "constraints": LaTeX-like string or "".
- "areas": array of integers from [1,2,3,4], ordered by priority.
  Legend: 1=stability_robustness, 2=optimality_efficiency, 3=reachability_controllability, 4=safety_risk
- Do NOT invent numeric values. If none are given in the mission, keep symbols (e.g., α, ε).

Guidance (soft, optional):
- Choose an objective that matches the mission (e.g., time, energy, smoothness, coverage, robustness/risk).
- If the mission mentions uncertainty or safety, reflect it using an appropriate formalism (chance, robust, DRO, or risk-sensitive), without inventing numbers.

Only valid JSON on one line.
"""

# SYSTEM_PROMPT = (
#     'Output JSON only (one line): {"area":"<composition-encoded labels>",'
#     '"objective_function":"<ONE LINE>", "explanation":"<2–3 sentences, optional>"}. '
#     "Labels: {stability_robustness, optimality_efficiency, reachability_controllability, safety_risk}. "
#     "Composition: A > B...[hierarchy]; (A || B) ...[parallel]; A -> B ...[feedback]; or a single label. "
#     "Choose 1–4 labels in priority order; objective is one LaTeX-like line, no constraints/dynamics."
#     "Explanations may be 2–3 sentences and should justify the label composition and the objective's emphasis."
#
# )

# SYSTEM_PROMPT = r"""
# You are “Objective Designer,” a control/trajectory-planning assistant.
#
# Return STRICT JSON on ONE line. No prose outside JSON. No $$.
#
# Output schema (and only this):
# {"objective_function":"<ONE line LaTeX-like objective>", "constraints":"<hard limits only, optional>"}
#
# Rules:
# - Use tell-tale tokens implied by the mission:
#   • time budget/deadline → include "min_{u,T} T" or "min T".
#   • worst-case gusts/EMI/disturbances → include "max_{\\|\\delta_t\\|\\le\\varepsilon}".
#   • CVaR/tail risk → include "CVaR_{\\alpha}[ ... ]".
#   • distributional ambiguity → include "sup_{P\\in\\mathcal{P}_\\rho} \\mathbb{E}_P[ ... ]" or "Wasserstein(\\rho)".
#   • coverage/search → include "\\sum_k \\Lambda_k |c_k(u)-\\phi_k|^2".
#   • smooth/low wear → include "\\sum_t \\|\\Delta u_t\\|_2^2".
#   • energy thrift → include "\\sum_t \\|u_t\\|_R^2".
# - Keep the objective ONE line. Put actuator/rate/torque/time limits ONLY in "constraints".
# - If chance safety is implied, put "P(g(z_t)\\le0)\\ge 1-\\epsilon" in "constraints".
# - Use mission units when provided (e.g., 35°/s).
# - Do NOT output placeholders, nulls, or empty wrappers (e.g., no "max_{...} 0").
# - Prefer the family signaled by cues (time → min T; gusts/EMI → worst-case/CVaR/DRO; coverage → spectral term; battery/wear → energy/smoothness).
#
# If the mission is too vague, still return a valid JSON with a reasonable baseline objective (e.g., tracking + small smoothness), and leave "constraints" empty.
# Return ONLY the JSON object.
# """


# SYSTEM_PROMPT = r"""
# You are “Objective Designer,” a control/trajectory-planning assistant.
#
# Return STRICT JSON on ONE line. No prose. No $$.
#
# Allowed outputs (choose exactly one):
# 1) {"objective_function":"<ONE line LaTeX-like objective>"}
# 2) {"objective_function":"<ONE line LaTeX-like objective>", "constraints":"<hard limits only>"}
#
# Rules:
# - Choose tell-tale tokens implied by the mission:
#   • deadline/time budget → include "min_{u,T} T" or "min T".
#   • worst-case gusts/EMI/disturbances → include "max_{\\|\\delta_t\\|\\le\\varepsilon}".
#   • tail risk → include "CVaR_{\\alpha}[ ... ]".
#   • distributional ambiguity → include "sup_{P\\in\\mathcal{P}_\\rho}\\,\\mathbb{E}_P[ ... ]" or "Wasserstein(\\rho)".
#   • coverage/search → include "\\sum_k \\Lambda_k |c_k(u)-\\phi_k|^2".
#   • smooth/low wear → include "\\sum_t \\|\\Delta u_t\\|_2^2".
#   • energy thrift → include "\\sum_t \\|u_t\\|_R^2".
# - Keep the objective ONE line. Do NOT add constraints unless the mission **explicitly states** hard numeric limits or a probability threshold.
#
# Constraints policy (very strict):
# - Only include "constraints" if the mission text **explicitly** provides numeric limits (e.g., rates, speeds, torques) or a probability form (e.g., "P(g(z_t)≤0)≥1−ε") or a concrete time cap (e.g., "≤ 150 s").
# - If included, **echo values and units verbatim** from the mission. Do NOT invent defaults, units, or variables. Do NOT infer platform-specific symbols (e.g., \\dot\\psi, \\dot q_i) unless they appear in the mission itself.
# - If the mission does not provide explicit numeric limits or ε, **omit the "constraints" field entirely** (use output form #1).
#
# Formatting bans:
# - No placeholders, nulls, or empty wrappers (e.g., no "max_{...} 0").
# - No duplicated bars "||", no made-up units, no platform guesswork.
# - If the mission is vague, still return a valid objective (e.g., tracking + small smoothness) with **no constraints field**.
#
# Return ONLY the JSON object on one line.
# """

# SYSTEM_PROMPT = r"""
# You are “Objective Designer,” a control/trajectory-planning assistant.
#
# Return STRICT JSON on ONE line. No prose outside JSON. No $$.
#
# Output schema (and only this):
# {"objective_function":"<ONE line LaTeX-like objective>", "constraints":"<hard limits only, optional>"}
#
# Rules:
# - Use tell-tale tokens implied by the mission:
#   • time budget/deadline → include "min_{u,T} T" or "min T".
#   • worst-case gusts/EMI/disturbances → include "max_{\\|\\delta_t\\|\\le\\varepsilon}".
#   • CVaR/tail risk → include "CVaR_{\\alpha}[ ... ]".
#   • distributional ambiguity → include "sup_{P\\in\\mathcal{P}_\\rho} \\mathbb{E}_P[ ... ]" or "Wasserstein(\\rho)".
#   • coverage/search → include "\\sum_k \\Lambda_k |c_k(u)-\\phi_k|^2".
#   • smooth/low wear → include "\\sum_t \\|\\Delta u_t\\|_2^2".
#   • energy thrift → include "\\sum_t \\|u_t\\|_R^2".
# - Keep the objective ONE line. Put actuator/rate/torque/time limits ONLY in "constraints".
# - If chance safety is implied, put "P(g(z_t)\\le0)\\ge 1-\\epsilon" in "constraints".
# - Use mission units when provided (e.g., 35°/s).
# - Do NOT output placeholders, nulls, or empty wrappers (e.g., no "max_{...} 0").
# - Prefer the family signaled by cues (time → min T; gusts/EMI → worst-case/CVaR/DRO; coverage → spectral term; battery/wear → energy/smoothness).
#
# If the mission is too vague, still return a valid JSON with a reasonable baseline objective (e.g., tracking + small smoothness), and leave "constraints" empty.
# Return ONLY the JSON object.
# """

# SYSTEM_PROMPT = r"""
# You are “Objective Designer,” a control-theory assistant for UAV MPC/trajectory planning.
#
# Return STRICT JSON on ONE line. No prose outside JSON. Do not include LaTeX delimiters like $$.
#
# ## Task
# Given a mission text, produce a mathematically coherent objective that may COMBINE multiple objective families with an explicit composition rule and a clear risk model. Put hard safety/rate limits in constraints (unless using soft-barriers by design).
#
# ## Allowed choices
# risk_model ∈ {expectation, worst_case, cvar, variance_penalized, dro}
# objective_families ⊆ {
#   time_optimal, energy_min, tracking, smoothness, H_infinity, minmax,
#   ergodic_coverage, tube_MPC, chance_constrained, softbarrier,
#   lexicographic, epsilon_constraint, staged, goal_programming,
#   mixed_integer_layer, minimax_regret
# }
# composition.type ∈ {weighted_sum, lexicographic, epsilon_constraint, minmax_then_regularize, staged}
#
# ## Output schema
# {
#   "objective_function": "<ONE line LaTeX-like objective reflecting chosen families & composition>",
#   "constraints": "<hard limits; units allowed>",
#   "risk_model": "<one of the risk_model values>",
#   "objective_families": ["<one or more from objective_families>"],
#   "composition": {
#     "type": "<composition.type>",
#     "order": ["<if lexicographic: families in priority order>"],
#     "weights": {"<family>": <number>, "...": <number>},   // only if weighted_sum
#     "epsilon": <number>,                                   // only if epsilon_constraint
#     "phases": ["<phase1>", "<phase2>"]                     // only if staged
#   },
#   "parameters": {
#     "disturbance_radius_eps": "<if worst_case/minmax or H_infinity>",
#     "cvar_alpha": "<if cvar>",
#     "dro_ball": "<e.g., Wasserstein radius if dro>",
#     "coverage_basis": "<if ergodic_coverage: e.g., Fourier>",
#     "notes": "<any short tuning notes>"
#   },
#   "structure": {
#     "tiers": [[ "<primary label(s)>" ], [ "<secondary>" ], [ "<tertiary>" ]],
#     "feedback_edges": [[ "<from>", "<to>" ], ...]
#   },
#   "explanation": "<≤3 short sentences justifying family+risk choices; mention tell-tale tokens in the objective>"
# }
#
# ## Hard rules
# 1) Put safety/actuator/rate limits in "constraints" (unless using softbarrier family).
# 2) If objective_families has >1 item, you MUST supply a valid "composition".
# 3) The objective must show unmistakable signals for chosen family(ies):
#    - time_optimal → includes “min T” or T in objective; optional ρ·(T−Tmax)_+^2.
#    - H_infinity or minmax/worst_case → includes “max_{δ}”, “sup”, or disturbance set radius.
#    - cvar → includes “CVaR_α[...]”.
#    - chance_constrained → constraints include “P(g(z)≤0)≥1−ε”.
#    - ergodic_coverage → includes Σ_k Λ_k |c_k(u) − φ_k|^2 (or spectral equivalent).
#    - softbarrier → includes barrier/hinge/[_]+ or log/reciprocal barrier terms.
#    - lexicographic → encode as a vector objective or add “[…]_lexi”.
#    - epsilon_constraint → objective for A, with separate constraint B ≤ ε.
#    - staged → show phase-specific symbols (e.g., u^(1), u^(2)) or piecewise form.
# 4) Keep the objective to ONE line; move all limits to "constraints".
# 5) Use mission units when provided (e.g., 35°/s).
# 6) Be diverse: avoid the same tracking+effort+slew template. Prefer families signaled by cues (time budgets → time_optimal/lexicographic; strong disturbances/EMI → worst_case/H_infinity/cvar/dro; coverage/search → ergodic_coverage; battery/throttle → energy_min + L1/L∞/smoothness).
# 7) If discrete decisions appear, include "mixed_integer_layer" (JSON only—no solver code).
# 8) Field gating (no stray fields):
#    - Include "phases" **only** when composition.type = "staged".
#    - Include "weights" **only** when composition.type = "weighted_sum".
#    - Include "order" **only** when composition.type = "lexicographic" (priority order).
#    - Include "epsilon" **only** when composition.type = "epsilon_constraint".
#    - composition.type must be one of {weighted_sum, lexicographic, epsilon_constraint, minmax_then_regularize, staged}. Do not invent other values.
#    - Do **not** nest "composition" or "objective_families" inside "parameters". Keys must appear exactly as in the schema.
#
# ## Rejection rule
# If the mission is too vague to choose families, output a single-family baseline (tracking or energy_min) AND state one missing cue in "parameters.notes".
#
# Return ONLY the JSON object.
# """

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--model_path", type=str, default="./sft-output")
#     ap.add_argument("--max_seq_len", type=int, default=2048)
#     ap.add_argument("--temperature", type=float, default=0.2)
#     ap.add_argument("--min_p", type=float, default=0.95)
#     ap.add_argument("--max_new_tokens", type=int, default=256)
#     args = ap.parse_args()
#
#     # Load model
#     model, tokenizer = FastLanguageModel.from_pretrained(
#         model_name=args.model_path,
#         max_seq_length=args.max_seq_len,
#         dtype=torch.bfloat16,
#         load_in_4bit=True,
#     )
#     FastLanguageModel.for_inference(model)
#     # Ensure a valid pad token setup (common for Llama/Unsloth)
#     if tokenizer.pad_token_id is None:
#         tokenizer.pad_token = tokenizer.eos_token
#     model.config.pad_token_id = tokenizer.pad_token_id
#     # Conversation history: keep just system + last exchange (simple!)
#     messages = [{"role": "system", "content": SYSTEM_PROMPT}]
#     print("Interactive mode. Type mission text. Commands: /exit")
#     while True:
#         try:
#             user_text = input("Mission> ").strip()
#         except (EOFError, KeyboardInterrupt):
#             print("\nBye.")
#             break
#         if not user_text or user_text.lower() == "/exit":
#             print("Bye.")
#             break
#         # Wrap exactly like training examples
#         messages.append({"role": "user", "content": f'Mission: "{user_text}"'})
#         # Build input and stream output
#         inputs = tokenizer.apply_chat_template(
#             messages,
#             tokenize=True,
#             add_generation_prompt=True,   # important for generation
#             return_tensors="pt",
#         ).to(model.device)
#         print("Assistant:")
#         streamer = TextStreamer(tokenizer, skip_prompt=True)
#         with torch.no_grad():
#             _ = model.generate(
#                 input_ids=inputs,
#                 streamer=streamer,
#                 max_new_tokens=args.max_new_tokens,
#                 do_sample=False,
#                 # temperature=args.temperature,
#                 # min_p=args.min_p,
#                 use_cache=True,
#                 eos_token_id=tokenizer.eos_token_id,
#                 pad_token_id=tokenizer.pad_token_id,
#             )
#         print("-" * 60)
#         # Keep only system + last assistant to keep context tiny
#         if len(messages) >= 3:
#             messages = [messages[0]]  # keep system only
#
#     return
#
# if __name__ == "__main__":
#     main()



def _get_eos_ids(tokenizer):
    ids = {tokenizer.eos_token_id}
    for tok in ["<|eot_id|>", "<|eom_id|>", "<|end|>", "<|im_end|>"]:
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is not None and tid != tokenizer.unk_token_id and tid != -1:
            ids.add(tid)
    return list(ids)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, default="./sft-output")
    ap.add_argument("--max_seq_len", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.7)   # was 0.2
    ap.add_argument("--top_p", type=float, default=0.9)         # replace min_p
    ap.add_argument("--max_new_tokens", type=int, default=256)
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

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    print("Interactive mode. Type mission text. Commands: /exit")
    while True:
        try:
            user_text = input("Mission> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not user_text or user_text.lower() == "/exit":
            print("Bye.")
            break

        # match training: NO quotes around mission
        messages.append({"role": "user", "content": f"Mission: {user_text}"})

        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)

        print("Assistant:")
        streamer = TextStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,   # hide <|eot_id|> etc.
        )
        with torch.no_grad():
            _ = model.generate(
                input_ids=inputs,
                streamer=streamer,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,                 # was False → can instantly emit EOT
                temperature=args.temperature,
                top_p=args.top_p,
                use_cache=True,
                eos_token_id=eos_ids,           # allow stopping on chat EOT tokens
                pad_token_id=tokenizer.pad_token_id,
            )
        print("-" * 60)

        # keep only system after each round (tiny context)
        messages = [messages[0]]

if __name__ == "__main__":
    main()
