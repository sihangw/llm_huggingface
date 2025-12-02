import os, json, re, torch
from dataclasses import dataclass
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, GenerationConfig, StoppingCriteria, StoppingCriteriaList
)

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
is_ddp = local_rank != -1  # True when launched via torchrun/accelerate

# ---------- Config ----------
HF_TOKEN = os.environ.get("HF_TOKEN")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct" if torch.cuda.is_available() \
             else "KingNish/Qwen2.5-0.5b-Test-ft"
USE_HISTORY = False
MAX_NEW_TOKENS = 256
SEED = 42

CONTROL_SPEC_SYSTEM = """You are a control-spec mapper.

TASK
- Read the mission (plain English).
- Choose a subset of areas from this fixed set:
  {stability_robustness, optimality_efficiency, reachability_controllability, safety_risk}.
- Return 1–4 areas that apply. Include only those supported by the mission text.
- Order = priority (most important first).
- Output JSON ONLY in this exact format:

{
  "area": "<area1>[ | <area2>[ | <area3>[ | <area4>]]]",
  "objective_function": "<ONE LINE mathematical objective (LaTeX-friendly)>",
  "explanation": "<sentences why these areas (optional)>"
}

RULES
- If only one area dominates → return just that one.
- If multiple cues are present → include 2–4, ordered by importance.
- Do NOT invent new labels; use only the 4 above.
- objective_function: ONE line of math, no constraints/dynamics; Σ, norms, terminal terms are OK.
- Return JSON only; no extra prose.

Example:
Mission: "Keep the quadrotor stable in wind gusts."
Output:
{
  "area": "stability_robustness",
  "objective_function": "\\min_{\\{u_t\\}} \\sum_t (\\|x_t\\|_2^2 + \\lambda \\|u_t\\|_2^2)",
  "explanation": "Emphasizes disturbance rejection and stability."
}
Mission: "Hold altitude despite gusts and minimize energy use."
Output:
{
  "area": "stability_robustness | optimality_efficiency",
  "objective_function": "\\min \\sum_t (\\alpha\\|u_t\\|_2^2 + \\beta\\|\\Delta h_t\\|_2^2)",
  "explanation": "Gust rejection implies robustness; energy use implies efficiency."
}

Mission: "Reach a waypoint quickly without violating speed limits."
Output:
{
  "area": "reachability_controllability | safety_risk",
  "objective_function": "\\min \\, T_f",
  "explanation": "Waypoint = reachability; limits = safety."
}
"""

# ---------- Utils ----------
def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_messages(user_text, history, system_text=CONTROL_SPEC_SYSTEM):
    msgs = [{"role": "system", "content": system_text}]
    if USE_HISTORY: msgs += history
    msgs.append({"role": "user", "content": f'Mission: "{user_text}"'})
    return msgs

# Stop once we see a top-level closing brace.
class StopOnFirstJson(StoppingCriteria):
    def __init__(self):
        super().__init__()
        self.stack = 0
        self.started = False
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Simple, byte-level brace balance on generated suffix
        # Works best if eos is also set; this is just a guardrail.
        # NOTE: this is approximate and language-model dependent.
        decoded = kwargs["tokenizer"].decode(input_ids[0].tolist(), skip_special_tokens=True)
        # Only check the *last* assistant turn content (best effort)
        # Find the last '{' and track braces from there
        last = decoded.rfind("{")
        if last == -1:
            return False
        text = decoded[last:]
        bal = 0
        for ch in text:
            if ch == "{": bal += 1
            elif ch == "}":
                bal -= 1
                if bal == 0:
                    return True
        return False

@dataclass
class InferenceObjects:
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    gen_cfg: GenerationConfig
    stoppers: StoppingCriteriaList

# ---------- Load ----------
print("=" * 60)
print(f"Loading model: {MODEL_NAME} (device: {DEVICE})")
print("=" * 60)

set_seed(SEED)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, token=HF_TOKEN, trust_remote_code=False, use_safetensors=True
)

# Set pad to eos for decoder-only models (common practice)
if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
    tokenizer.pad_token = tokenizer.eos_token

torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else None

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    token=HF_TOKEN,
    low_cpu_mem_usage=True,
    torch_dtype=torch_dtype,
    # device_map="auto" if torch.cuda.is_available() else None,
    device_map=None if is_ddp else "auto",
    trust_remote_code=False,
    use_safetensors=True,
)

model.eval()
model.config.use_cache = True

# Single source of truth for generation params
gen_cfg = GenerationConfig(
    max_new_tokens=MAX_NEW_TOKENS,
    do_sample=False,
    repetition_penalty=1.05,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)

stoppers = StoppingCriteriaList([StopOnFirstJson()])

objs = InferenceObjects(model=model, tokenizer=tokenizer, gen_cfg=gen_cfg, stoppers=stoppers)

# ---------- Inference ----------
@torch.inference_mode()
def generate_once(objs: InferenceObjects, user_text: str, history):
    msgs = build_messages(user_text, history)
    # use chat template -> PT tensors directly
    inputs = objs.tokenizer.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt", truncation=True
    ).to(objs.model.device)

    out = objs.model.generate(
        input_ids=inputs,
        generation_config=objs.gen_cfg,
        stopping_criteria=objs.stoppers,
        tokenizer=objs.tokenizer,  # passed through to stopper
    )

    gen = out[0, inputs.shape[1]:]
    text = objs.tokenizer.decode(gen, skip_special_tokens=True).strip()

    # Optional: try to extract first JSON block to be safe
    m = re.search(r"\{.*?\}", text, flags=re.S)
    return m.group(0) if m else text

# ---------- Interactive Loop ----------
print("\nInteractive chat started. Type 'exit' or 'quit' to stop.\n")
history = []

try:
    while True:
        user_in = input("> ")
        if user_in.strip().lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        reply = generate_once(objs, user_in, history)
        print(f"\nAssistant: {reply}\n")

        if USE_HISTORY:
            history.append({"role": "user", "content": f'Mission: \"{user_in}\"'})
            history.append({"role": "assistant", "content": reply})
except KeyboardInterrupt:
    print("\nInterrupted. Goodbye!")
