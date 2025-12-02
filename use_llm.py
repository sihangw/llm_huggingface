# """
# Example driver that uses custom_llm to generate CartPole cost/constraints/policy files.
# If --spec/--original are not provided, embedded defaults are used so you can run instantly.
#
# Usage:
#   # With Unsloth
#   python use_custom_llm.py --provider unsloth --model_path unsloth/Llama-3.2-1B-Instruct --stream
#
#   # With vanilla HF
#   python use_custom_llm.py --provider hf --model_path mistralai/Mistral-7B-Instruct-v0.2
# """
# import argparse, pathlib, re, textwrap
# from typing import Optional
# from custom_llm import LLMConfig, ChatSession
#
# CONTROL_SYSTEM_PROMPT = (
#     "You write small, deterministic Python functions for control. "
#     "Use numpy only. Follow the provided function signatures exactly."
# )
#
# CODEGEN_INSTRUCTIONS = (
#     "RETURN exactly three files with headers:\n"
#     "### FILE: cost_fn.py\n...code...\n"
#     "### FILE: constraints.py\n...code...\n"
#     "### FILE: policy_program.py\n...code...\n"
# )
#
# EMBEDDED_SPEC = textwrap.dedent('''
# task:
#   name: CartPole-Continuous
#   horizon: 1000
#   dt: 0.02
# interfaces:
#   obs: [x, x_dot, theta, theta_dot]
#   act: [force]
# mission: |
#   Keep the pole upright (theta≈0), keep the cart near the center (x≈0), and minimize control effort.
#   Prioritize safety: never exceed the angle and position bounds. Penalize any soft exceedance.
# safety:
#   - type: state_bound
#     key: theta
#     max_deg: 12.0
#   - type: state_bound
#     key: x
#     max: 2.4
#   - type: input_bound
#     key: force
#     min: -1.0
#     max:  1.0
# signatures:
#   cost_fn: "def cost_fn(obs, act, info, w) -> float"
#   constraints_fn: "def g_constraints(obs, act, info) -> list[float]"
#   policy_fn: "def policy(params, obs) -> dict"
# preferences:
#   style: "numpy-only; deterministic; O(1); parametric weights; no globals beyond DEFAULT_*"
# ''')
#
# EMBEDDED_ORIGINAL = textwrap.dedent('''
# import numpy as np
# def original_cost(obs, act, info):
#     x, x_dot = float(obs["x"]), float(obs["x_dot"])
#     th, th_dot = float(obs["theta"]), float(obs["theta_dot"])
#     u = float(act.get("force", 0.0))
#     tilt = th*th
#     pos  = x*x
#     vel  = 0.05*(x_dot*x_dot + th_dot*th_dot)
#     effort = 1e-3*u*u
#     th_lim = float(info.get("angle_limit_deg", 12.0))
#     x_lim  = float(info.get("pos_limit", 2.4))
#     s_ang = max(0.0, abs(np.degrees(th)) - th_lim)**2
#     s_pos = max(0.0, abs(x) - x_lim)**2
#     return float(4.0*tilt + 1.0*pos + 0.2*vel + effort + 5.0*(s_ang+s_pos))
# ''')
#
# def build_prompt(spec_yaml: str, original_fn_code: Optional[str]) -> str:
#     parts = [
#         "SYSTEM:\n" + CONTROL_SYSTEM_PROMPT,
#         "\nSPEC (YAML):\n" + spec_yaml.strip(),
#     ]
#     if original_fn_code is not None:
#         parts.append("\nORIGINAL_FUNCTION (Python):\n### FILE: original_fn.py\n" + original_fn_code.strip())
#     parts.append(
#         "\nREQUIREMENTS:\n"
#         "1) Implement cost_fn(obs, act, info, w)->float (nonnegative; deterministic).\n"
#         "2) Implement g_constraints(obs, act, info)->list[float] with g_i<=0 satisfied.\n"
#         "3) Implement policy(params, obs)->dict; clip to action bounds; parametric.\n"
#         "4) Use names in spec exactly; numpy only; no globals except DEFAULT_*.\n"
#         "5) Keep code O(1), side-effect free.\n\n" + CODEGEN_INSTRUCTIONS
#     )
#     return "\n".join(parts)
#
# def split_and_write(completion_text: str, outdir: pathlib.Path):
#     outdir.mkdir(parents=True, exist_ok=True)
#     blocks = re.split(r"^### FILE: ", completion_text, flags=re.M)
#     paths = {}
#     for b in blocks:
#         if not b.strip():
#             continue
#         name, *rest = b.splitlines()
#         code = "\n".join(rest).strip() + "\n"
#         p = outdir / name.strip()
#         p.write_text(code, encoding="utf-8")
#         paths[name.strip()] = p
#     return paths
#
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--provider", default="unsloth", choices=["unsloth","hf"])
#     ap.add_argument("--model_path", default="unsloth/Llama-3.2-1B-Instruct")
#     ap.add_argument("--max_seq_len", type=int, default=4096)
#     ap.add_argument("--max_new_tokens", type=int, default=1024)
#     ap.add_argument("--temperature", type=float, default=0.2)
#     ap.add_argument("--top_p", type=float, default=0.95)
#     ap.add_argument("--spec", type=str, default=None, help="Path to YAML spec. If omitted, uses embedded CartPole spec.")
#     ap.add_argument("--original", type=str, default=None, help="Path to original function. If omitted, uses embedded baseline.")
#     ap.add_argument("--outdir", type=str, default="generated")
#     ap.add_argument("--stream", action="store_true")
#     args = ap.parse_args()
#
#     cfg = LLMConfig(
#         provider=args.provider,
#         model_path=args.model_path,
#         max_seq_len=args.max_seq_len,
#         max_new_tokens=args.max_new_tokens,
#         temperature=args.temperature,
#         top_p=args.top_p,
#     )
#     chat = ChatSession(cfg)
#     chat.set_system(CONTROL_SYSTEM_PROMPT)
#
#     spec_yaml = pathlib.Path(args.spec).read_text(encoding="utf-8") if args.spec and pathlib.Path(args.spec).exists() else EMBEDDED_SPEC
#     original_code = pathlib.Path(args.original).read_text(encoding="utf-8") if args.original and pathlib.Path(args.original).exists() else EMBEDDED_ORIGINAL
#
#     prompt = build_prompt(spec_yaml, original_code)
#     chat.add_user(prompt)
#     res = chat.generate(stream=args.stream)
#
#     paths = split_and_write(res.text, pathlib.Path(args.outdir))
#     for name, p in paths.items():
#         print(f"[wrote] {p}")
#
# if __name__ == "__main__":
#     main()




import argparse
from custom_llm import LLMConfig, ChatSession

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="unsloth", choices=["unsloth","hf"])
    ap.add_argument("--model_path", default="unsloth/Llama-3.2-1B-Instruct")
    ap.add_argument("--max_seq_len", type=int, default=4096)
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--question", type=str, default="give me a review of lexus lx570?")
    args = ap.parse_args()

    cfg = LLMConfig(
        provider=args.provider,
        model_path=args.model_path,
        max_seq_len=args.max_seq_len,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    chat = ChatSession(cfg)
    chat.set_system("You are a helpful assistant. Answer concisely.")

    chat.add_user(args.question)
    res = chat.generate(stream=args.stream)
    if not args.stream:
        print(res.text)

if __name__ == "__main__":
    main()









