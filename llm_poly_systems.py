import re, ast
import argparse
from pathlib import Path
from custom_llm import LLMConfig, ChatSession

PROMPT_TEMPLATE = """
You will design a self-contained Python cost function for sampling-based MPC.

### Mission
{mission}

### Context (spec.py)
```python
{spec_code}

Return ONLY valid Python function.
"""

# PROMPT_TEMPLATE = """
# You will design a self-contained Python cost function for sampling-based MPC
# on a polynomial, control-affine system with a known safe region.
#
# ### Mission
# {mission}
#
# ### Context (spec.py)
# The following Python code defines the system configuration and constants, such as:
# - state dimension n
# - input dimension m
# - time horizon N
# - contraction-safe state box B_R (e.g. R_SAFE or R_BOX)
# - target state X_GOAL
# - input magnitude limit U_MAX
# - any helper utilities you may call
#
# ```python
# {spec_code}

# SYS_DEFAULT = (
#     "You are in control/robotics domain."
#     "Return only valid, self-contained Python code that defines cost_fn(x, u, u_prev=None, terminal=False). "
#     "No markdown. No comments. No explanations. Code only."
# )

# SYS_DEFAULT = (
#     "You are an expert in nonlinear control, model predictive control (MPC), "
#     "and reinforcement learning. "
#     "Your job is to write valid, self-contained Python code that defines "
#     "a scalar cost function for a discrete-time control problem."
#     "Requirements:"
#     "- Return ONLY valid Python code. No markdown."
#     "- Define a single function with signature:"
#     "    def cost_fn(x, u, u_prev=None, terminal=False):"
#     "- The function must:"
#     "    * Take x as a 1D numpy array of shape (n,), representing the current state."
#     "    * Take u as a 1D numpy array of shape (m,), representing the current control."
#     "    * Take u_prev as the previous control (1D array) or None."
#     "    * Take terminal as a boolean flag indicating whether this is the final step."
#     "    * Return a single Python float cost."
#     "- You may import numpy as np at the top of the code."
#     "- Do not read from files or the network. Do not print. Just compute and return the cost."
# )


SYS_DEFAULT = (
    "You are an expert in nonlinear control and model predictive control (MPC). "
    "Your job is to write valid, self-contained Python code for a scalar cost function.\n"
    "Requirements:\n"
    "- Return ONLY valid Python code, no markdown, no comments, no explanations.\n"
    "- Define exactly one function:\n"
    "    def cost_fn(x, u, u_prev=None, terminal=False):\n"
    "- x and u are 1D numpy arrays; u_prev is the previous control or None; "
    "terminal is a boolean for the final step.\n"
    "- You may import numpy as np (if needed) and return a single Python float.\n"
    "- Do not read from files or the network. Do not print."
)

SYS_PROMPT = (
    "You are an exceptionally intelligent PYTHON coding assistant that consistently"
    "delivers accurate and reliable responses in nonlinear control and optimization.\n"
    "Your response MUST be ONLY PYTHON code in the following exact format:\n"
    "```python\n"
    "import numpy as np\n"
    "def cost_fn(x, u, u_prev=None, terminal=False):\n"
    "    u_prev = u if u_prev is None else u_prev\n"
    "    cost = 0.0\n"
    "    # FILL\n"
    "    return cost\n"
    "```\n"
    "Rules:\n"
    "- Define exactly ONE function named cost_fn with the signature above.\n"
    "- x and u are 1D numpy arrays; u_prev may be None; terminal is bool.\n"
    "- Return a finite Python float for any inputs.\n"
    "- Do NOT read files, do NOT use network, do NOT print.\n"
    "### Instruction\n"
)
RESPONESE_PROMPT = "\n### Response\n"

PROMPT = """
We control an n-dimensional polynomial, control-affine discrete-time system
    x_{k+1} = T(x_k, u_k),
and we want a cost function for sampling-based MPC or trajectory optimization
to solve a finite-horizon reachability task.

Global variables available in the same module:
- X_GOAL : 1D numpy array (target state)
- U_MAX  : positive float (characteristic scale / upper bound for control magnitude)
- R_SAFE : positive float (defines a preferred safe region in state space, e.g. ||x||_inf <= R_SAFE)

Requirements:
- x and u are 1D numpy arrays; return a finite float.
- If u_prev is None, treat u_prev = u (no smoothness penalty on first step).
- Running cost should include: goal shaping, control effort normalized by U_MAX^2,
  smoothness penalty on (u-u_prev), and a safety violation penalty based on max(0, ||x||_inf - R_SAFE).
- Prefer smooth penalties (quadratic + hinge/softplus). Avoid hard discontinuities.
- Terminal cost should strongly emphasize goal and safety (e.g., significantly larger than running terms).
"""

USER_PROMPT = """
We control an n-dimensional polynomial, control-affine discrete-time system
    x_{k+1} = T(x_k, u_k),
and we want a cost function for sampling-based MPC or trajectory optimization
to solve a finite-horizon reachability task.

Global variables available in the same module:
- X_GOAL : 1D numpy array (target state)
- U_MAX  : positive float (characteristic scale / upper bound for control magnitude)
- R_SAFE : positive float (defines a preferred safe region in state space, e.g. ||x||_inf <= R_SAFE)

Design a function

    def cost_fn(x, u, u_prev=None, terminal=False):

that encodes the following high-level objectives:

1. The state should reach and stay near X_GOAL by the end of the horizon.
2. Control inputs should not be excessively large compared to U_MAX.
3. Control inputs should not change too abruptly from one step to the next.
4. The state should remain inside or close to the safe region associated with R_SAFE.

You may choose:
- how to measure tracking error,
- how to measure control magnitude and smoothness,
- how to quantify safe-region violation,
- how to weight these different terms,
- and how to treat terminal vs non-terminal steps,
as long as the resulting cost function is reasonable for this task.

Guidelines:
- For terminal == False, the cost should balance progress toward X_GOAL, moderation of control effort, smoothness, and staying in or near the safe region.
- For terminal == True, the cost should place greater emphasis on being very close to X_GOAL and inside the safe region, with other terms as you judge appropriate.
- The function must return a finite Python float for any x and u.

Return ONLY the Python code defining cost_fn(x, u, u_prev=None, terminal=False).
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="unsloth", choices=["unsloth","hf"])
    ap.add_argument("--model_path", default="unsloth/Llama-3.2-3B-Instruct")
    ap.add_argument("--max_seq_len", type=int, default=4096)
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--stream", action="store_true")
    # ap.add_argument("--spec", type=str, default="./poly_sys_spec.py", help="Path to spec.py with any code/config to show the model")


    ap.add_argument(
        "--mission",
        type=str,
        default=(
            "We control an n-dimensional polynomial, control-affine system. "
            "Design a cost_fn that drives x to X_GOAL with small control effort "
            "and smooth inputs (use u_prev if provided). "
            "Enforce/penalize input magnitudes relative to U_MAX. "
            "Return only valid Python defining cost_fn(x, u, u_prev=None, terminal=False)."
        ),
    )

    ap.add_argument("--system", type=str, default=SYS_DEFAULT)
    ap.add_argument("--num", type=int, default=1, help="Number of variants to generate")
    # ap.add_argument("--out", type=str, default="./cost_fn.py", help="Optional path to save LLM output (e.g., cost_fn.py)")
    ap.add_argument("--out_dir", type=str, default="./samples", help="Directory to save variants")

    args = ap.parse_args()

    # spec_path = Path(args.spec)
    # spec_code = spec_path.read_text(encoding="utf-8")

    cfg = LLMConfig(
        provider=args.provider,
        model_path=args.model_path,
        max_seq_len=args.max_seq_len,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    chat = ChatSession(cfg)
    chat.set_system(args.system)

    # base_prompt = (
    #
    #     PROMPT_TEMPLATE.format(
    #     mission=args.mission.strip(),
    #     spec_code=spec_code.strip(),
    # ))

    base_prompt = PROMPT+RESPONESE_PROMPT
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate M variants
    for i in range(1, args.num + 1):
        # Keep system, clear chat so each variant is independent
        chat.clear_history(keep_system=True)

        # Optional nudge for diversity (tag each variant)
        prompt = base_prompt + f"\n# Variant: {i}\n"

        chat.add_user(prompt)
        res = chat.generate(stream=False)  # streaming off for batch saves
        text = res.text or ""

        # text = extract_cost_fn(text)
        # text = ensure_double_integrator(text)
        # text = ensure_single_numpy_import(text)

        # Save as cost_fn_v{i}.py
        out_path = out_dir / f"cost_fn_v{i}.py"
        out_path.write_text(text, encoding="utf-8")
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()