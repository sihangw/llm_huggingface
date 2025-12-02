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

SYS_DEFAULT = (
    "You are in control/robotics domain."
    "Return only valid, self-contained Python code that defines cost_fn(x, u, u_prev=None, terminal=False). "
    "No markdown. No comments. No explanations. Code only."
)

DOUBLE_INTEGRATOR_DEF = """
import numpy as np

def double_integrator_discretized(dt: float = 0.01):
    A = np.array([[1.0, dt],
                  [0.0, 1.0]], dtype=float)
    B = np.array([[0.5 * dt ** 2],
                  [dt]], dtype=float)
    c = np.zeros(2, dtype=float)
    return A, B, c
""".strip()

def ensure_double_integrator(code: str) -> str:
    """Prepend the helper ONLY if its function definition is missing."""
    has_def = re.search(r'(?m)^\s*def\s+double_integrator_discretized\s*\(', code) is not None
    if not has_def:
        code = f"{DOUBLE_INTEGRATOR_DEF}\n\n{code}".lstrip()
    return code


def extract_cost_fn(text: str) -> str:
    # 1) strip code fences
    text = re.sub(r"^```[a-zA-Z]*|```$", "", text.strip(), flags=re.MULTILINE)
    lines = text.splitlines()

    # 2) find 'def cost_fn('
    start = None
    for i, ln in enumerate(lines):
        if re.search(r'^\s*def\s+cost_fn\s*\(', ln):
            start = i
            break
    if start is None:
        raise ValueError("cost_fn not found")

    # 3) determine the body indent from the first non-empty line after def
    body_indent = None
    for j in range(start + 1, len(lines)):
        s = lines[j].strip()
        if not s:   # skip blank lines
            continue
        body_indent = len(lines[j]) - len(lines[j].lstrip())
        break
    if body_indent is None:
        # empty body (unlikely but handle)
        body_indent = 4

    # 4) collect only the def line + indented block; stop at the first dedent
    kept = [lines[start]]
    for k in range(start + 1, len(lines)):
        ln = lines[k]
        if ln.strip() == "":
            kept.append(ln)              # allow blank lines inside the function
            continue
        indent = len(ln) - len(ln.lstrip())
        if indent < body_indent:
            break                        # function ended; ignore following prose
        kept.append(ln)

    code = "\n".join(kept).strip()

    # 5) remove full-line comments and inline trailing comments
    cleaned = []
    for ln in code.splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        ln = re.sub(r"#.*", "", ln).rstrip()
        if ln:
            cleaned.append(ln)
    code = "\n".join(cleaned).strip()

    # 6) ensure numpy import if used
    if "np." in code and "import numpy" not in code:
        code = "import numpy as np\n\n" + code

    return code

def ensure_single_numpy_import(code: str) -> str:
    """Remove all numpy imports and add exactly one 'import numpy as np' at the top if needed."""
    # Drop any numpy import lines
    body_lines = []
    for ln in code.splitlines():
        s = ln.strip()
        if re.match(r"^(from\s+numpy\s+import\b|import\s+numpy(\s+as\s+\w+)?\b)", s):
            continue
        body_lines.append(ln)
    body = "\n".join(body_lines).lstrip()

    # Add a single canonical import if np is used
    if "np." in body:
        return "import numpy as np\n\n" + body
    return body

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="unsloth", choices=["unsloth","hf"])
    ap.add_argument("--model_path", default="unsloth/Llama-3.2-3B-Instruct")
    ap.add_argument("--max_seq_len", type=int, default=4096)
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--spec", type=str, default="./double_integrator_spec.py", help="Path to spec.py with any code/config to show the model")


    ap.add_argument(
        "--mission",
        type=str,
        default=(
            "We are controlling a double integrator system: position and velocity. "
            "Finish the cost function design such that pushes both position and velocity to 0 "
            "with minimal control effort. The control at any time should be in the range [-2, 2]"
        ),
    )

    ap.add_argument("--system", type=str, default=SYS_DEFAULT)
    ap.add_argument("--num", type=int, default=20, help="Number of variants to generate")
    # ap.add_argument("--out", type=str, default="./cost_fn.py", help="Optional path to save LLM output (e.g., cost_fn.py)")
    ap.add_argument("--out_dir", type=str, default="./samples", help="Directory to save variants")

    args = ap.parse_args()

    spec_path = Path(args.spec)
    spec_code = spec_path.read_text(encoding="utf-8")

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

    base_prompt = PROMPT_TEMPLATE.format(
        mission=args.mission.strip(),
        spec_code=spec_code.strip(),
    )

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

        text = extract_cost_fn(text)
        text = ensure_double_integrator(text)
        text = ensure_single_numpy_import(text)

        # Save as cost_fn_v{i}.py
        out_path = out_dir / f"cost_fn_v{i}.py"
        out_path.write_text(text, encoding="utf-8")
        print(f"Saved: {out_path}")

    # cfg = LLMConfig(
    #     provider=args.provider,
    #     model_path=args.model_path,
    #     max_seq_len=args.max_seq_len,
    #     max_new_tokens=args.max_new_tokens,
    #     temperature=args.temperature,
    #     top_p=args.top_p,
    # )
    # chat = ChatSession(cfg)
    # chat.set_system(args.system)
    #
    # prompt = PROMPT_TEMPLATE.format(
    #     mission=args.mission.strip(),
    #     spec_code=spec_code.strip(),
    # )
    #
    # chat.add_user(prompt)
    # res = chat.generate(stream=args.stream)
    #
    # if args.stream:
    #     # streaming already printed by your client; also capture final text if available
    #     text = getattr(res, "text", "")
    # else:
    #     text = res.text
    #
    # # --- 🧹 Clean Markdown fences like ```python ... ```
    # if text.startswith("```"):
    #     parts = text.strip().split("```")
    #     text = "".join(parts[1:]) if len(parts) > 1 else text
    #     # Drop optional 'python' language tag
    #     if text.lstrip().lower().startswith("python"):
    #         text = text.split("\n", 1)[1]
    #     text = text.strip()
    # # --- end cleaning section ---
    #
    # if args.out:
    #     Path(args.out).write_text(text, encoding="utf-8")
    # else:
    #     print(text)

if __name__ == "__main__":
    main()