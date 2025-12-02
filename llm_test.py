import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# ---------- Config ----------
# Add Hugging Face token for authentication (Ensure this is stored securely)
HF_TOKEN = os.environ.get("HF_TOKEN")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct" if torch.cuda.is_available() \
             else "KingNish/Qwen2.5-0.5b-Test-ft"
USE_HISTORY = False
MAX_NEW_TOKENS = 256
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
"""
# Example:
# Mission: "Keep the quadrotor stable in wind gusts."
# Output:
# {
#   "area": "stability_robustness",
#   "objective_function": "\\min_{\\{u_t\\}} \\sum_t (\\|x_t\\|_2^2 + \\lambda \\|u_t\\|_2^2)",
#   "explanation": "Emphasizes disturbance rejection and stability."
# }
# Mission: "Hold altitude despite gusts and minimize energy use."
# Output:
# {
#   "area": "stability_robustness | optimality_efficiency",
#   "objective_function": "\\min \\sum_t (\\alpha\\|u_t\\|_2^2 + \\beta\\|\\Delta h_t\\|_2^2)",
#   "explanation": "Gust rejection implies robustness; energy use implies efficiency."
# }
#
# Mission: "Reach a waypoint quickly without violating speed limits."
# Output:
# {
#   "area": "reachability_controllability | safety_risk",
#   "objective_function": "\\min \\, T_f",
#   "explanation": "Waypoint = reachability; limits = safety."
# }

# ---------- Helpers ----------
# def render_prompt(tokenizer, user_text, history, system_text=CONTROL_SPEC_SYSTEM):
#     msgs = [{"role": "system", "content": system_text}]
#     if USE_HISTORY:
#         msgs += history
#     msgs.append({"role": "user", "content": f'Mission: "{user_text}"'})
#     # if hasattr(tokenizer, "apply_chat_template"):
#     #     inputs = tokenizer.apply_chat_template(
#     #         msgs,
#     #         add_generation_prompt=True,
#     #         # tokenize=True,
#     #         return_tensors="pt",
#     #         padding=True,
#     #         truncation=True,
#     #     )
#     #     # return inputs
#     #     return tokenizer.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", truncation=True)
#
# @torch.inference_mode()
# def generate_once(model, tokenizer, inputs, max_new_tokens=MAX_NEW_TOKENS):
#     """
#     inputs: dict from render_prompt (expects 'input_ids' and 'attention_mask')
#     """
#     # Resolve pad/eos ids robustly
#     pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
#     eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else pad_id
#
#     # Build a clean GenerationConfig to avoid sampling warnings
#     gen_cfg = GenerationConfig(
#         do_sample=False,          # greedy
#         temperature=None,         # explicitly unset sampling params
#         top_p=None,
#         top_k=None,
#         max_new_tokens=max_new_tokens,
#         repetition_penalty=1.05,
#         eos_token_id=eos_id,
#         pad_token_id=pad_id,
#     )
#
#     input_ids = inputs["input_ids"].to(model.device)
#     attention_mask = inputs["attention_mask"].to(model.device)
#
#     output = model.generate(
#         input_ids=input_ids,
#         attention_mask=attention_mask,   # <<< fixes the attention mask warning
#         generation_config=gen_cfg,
#     )
#
#     # slice off the prompt tokens (batch size assumed 1)
#     gen = output[0, input_ids.shape[1]:]
#     return tokenizer.decode(gen, skip_special_tokens=True).strip()
# @torch.inference_mode()
# def generate_once(model, tokenizer, input_ids, max_new_tokens=MAX_NEW_TOKENS):
#     output = model.generate(
#         input_ids=input_ids.to(model.device),
#         max_new_tokens=max_new_tokens,
#         do_sample=False,
#         # temperature=0.0,
#         # top_p=0.9,
#         repetition_penalty=1.05,
#         eos_token_id=tokenizer.eos_token_id,
#         pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
#     )
#     gen = output[0, input_ids.shape[1]:]
#     return tokenizer.decode(gen, skip_special_tokens=True).strip()

def render_prompt(tokenizer, user_text, history, system_text=CONTROL_SPEC_SYSTEM):
    msgs = [{"role": "system", "content": system_text}]
    if USE_HISTORY:
        msgs += history
    msgs.append({"role": "user", "content": f'Mission: "{user_text}"'})

    # Ensure pad token is defined
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1) Get the chat as a string, not tensors
    chat_str = tokenizer.apply_chat_template(
        msgs,
        add_generation_prompt=True,
        tokenize=False,
    )

    # 2) Tokenize the string to get a dict with attention_mask
    enc = tokenizer(
        chat_str,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    return enc  # dict with 'input_ids' and 'attention_mask'

@torch.inference_mode()
def generate_once(model, tokenizer, inputs, max_new_tokens=MAX_NEW_TOKENS):
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else pad_id

    gen_cfg = GenerationConfig(
        do_sample=False,             # greedy
        temperature=None, top_p=None, top_k=None,  # silence sampling warnings
        max_new_tokens=max_new_tokens,
        repetition_penalty=1.05,
        eos_token_id=eos_id,
        pad_token_id=pad_id,
    )

    out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,   # fixes the pad==eos ambiguity
        generation_config=gen_cfg,
        return_dict_in_generate=True,
    )

    # Robust slicing, supports batch
    sequences = out.sequences
    prompt_lens = attention_mask.sum(dim=1)

    texts = []
    for i in range(sequences.size(0)):
        gen_tokens = sequences[i, prompt_lens[i].item():]
        texts.append(tokenizer.decode(gen_tokens, skip_special_tokens=True).strip())

    return texts[0] if len(texts) == 1 else texts

# ---------- Load ----------
print("=" * 60)
print(f"Loading model: {MODEL_NAME} (device: {DEVICE})")
print("=" * 60)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, token=HF_TOKEN, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model.to(DEVICE)
model.config.use_cache = True

# ---------- Interactive Loop ----------
print("\nInteractive chat started. Type 'exit' or 'quit' to stop.\n")
history = []
while True:
    user_in = input("> ")
    if user_in.strip().lower() in {"exit", "quit"}:
        print("Goodbye!")
        break
    prompt_ids = render_prompt(tokenizer, user_in, history)
    reply = generate_once(model, tokenizer, prompt_ids)
    print(f"\nAssistant: {reply}\n")
    if USE_HISTORY:
        history.append({"role": "user", "content": f'Mission: "{user_in}"'})
        history.append({"role": "assistant", "content": reply})