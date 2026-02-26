import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"   # must be set before torch initializes CUDA

import re
import json
import math
import torch
torch.cuda.set_device(0)

import argparse
import numpy as np
from datasets import Dataset

from peft import LoraConfig
from datasets import load_dataset
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from typing import List, Dict, Any, Tuple

from trl.rewards import think_format_reward




# -----------------------------
# Helper functions (same as before)
# -----------------------------
SYSTEM_MSG = (
    "You are an expert in nonlinear control and model predictive control (MPC). "
    "You will propose MPC cost weights. "
    "CRITICAL OUTPUT RULE: "
    "-Return ONLY valid JSON and nothing else. "
    "-Output must be exactly one JSON object with key \"w\". "
    "-\"w\" must be a list of finite numbers. "
    "-All numbers must be >= 0. "
    "-Length must match the requested dimension. "
    "-No extra keys, no markdown, no prose."
)


def build_user_msg(prompt_sys: str, prompt_score: str, w_dim: int) -> str:
    # Keep this consistent with your downstream parser and weight sampler semantics.
    semantics = (
        "Weight vector format (length {w_dim}): "
        "w = [Ws_track, Ws_safe, Ws_u, Ws_smooth, Wt_track, Wt_safe, Wt_u] "
        "-Ws_* are stage (running) weights. "
        "-Wt_* are terminal weights. "
        "-track penalizes ||x - x_goal||^2 "
        "-safe penalizes max(0, ||x||_inf - R_safe)^2 "
        "-u penalizes ||u||^2 normalized by u_max "
        "-smooth penalizes ||u - u_prev||^2 (stage only) "
        "Weight prior/range hint (for stability; do not output negatives): "
        "Ws_track, Ws_safe in about [10^-1.5, 10^2] (may scale with 1/rho). "
        "Ws_u, Ws_smooth in about [10^-3, 10^0.5]. "
        "Wt_track, Wt_safe typically 3x–30x larger than stage weights. "
        "Wt_u typically similar scale to Ws_u. "
    )
    return (
        # "INSTANCE SUMMARY:\n" #already in dataset
        f"{prompt_sys} "
        # "SCORE PREFERENCES:\n"
        f"{prompt_score} "
        f"{semantics} "
        "TASK: Choose w to maximize the score. "
        "OUTPUT FORMAT (STRICT): Return ONLY JSON: {{\"w\": [..{w_dim} numbers..]}}"
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


# def parse_w(text: str, w_dim: int) -> np.ndarray:
#     obj = _extract_json_obj(text)
#     w = np.asarray(obj["w"], dtype=float).reshape(-1)
#     if w.size != w_dim:
#         raise ValueError(f"Expected w_dim={w_dim}, got {w.size}")
#     if not np.all(np.isfinite(w)):
#         raise ValueError("Non-finite weights")
#     if np.any(w < 0):
#         raise ValueError("Negative weight(s)")
#     return w

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
    feature = row.get("feat_vec", "")

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

    return {"prompt": messages, "solution": assistant, "rho_goal": feature[7]}

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

from reward_MLP_offline import MLPRegressor
X = np.load("data_500_dt01_N100/run1_W.npy")
Y = np.load("data_500_dt01_N100/run1_Y.npy")
mlp = MLPRegressor(hidden=(256,256), epochs=300, lr=1e-3, device="cuda")
mlp.fit(X, Y)

def reward_num_unique_letters(completions, **kwargs):
    """Reward function that rewards completions with more unique letters."""
    completion_contents = [completion[0]["content"] for completion in completions]
    return [float(len(set(content))) for content in completion_contents]

def weights_reward(completions, **kwargs):
    """
    Strict format reward for GRPO:
      - returns -1.0 if not valid JSON {"w":[...]} or wrong dim/non-finite/negatives/out of bounds
      - else returns 0.0 (format reward only)

    Expects:
      completions: list[list[dict]]  (TRL format)
      kwargs contains rho_goal: list[float] or float
    """
    # TRL gives completions as: completions[i][0]["content"]
    texts = [c[0]["content"] for c in completions]

    # rho_goal can arrive as a list (batched) or scalar
    rho_goal = kwargs.get("rho_goal", None)
    if isinstance(rho_goal, (list, tuple, np.ndarray)):
        rho_list = list(rho_goal)
    else:
        rho_list = [rho_goal] * len(texts)

    dev = next(mlp.model.parameters()).device
    mlp.model.eval()

    rewards = []
    for text, rg in zip(texts, rho_list):
        r = 0
        try:
            # 1) parse JSON -> w
            w = parse_w(text, w_dim=7)  # <- YOUR parse_w
            r += 0.2

            # 2) non-negativity (parse_w checks finite + dim already)
            if np.any(w < 0.0):
                rewards.append(r)
                continue
            r += 0.1

            # 3) bound check from sampler limits
            if rg is None:
                # if you *require* rho_goal, treat missing as invalid
                rewards.append(r)
                continue
            r += 0.1

            lo, hi = get_weight_bounds_from_sampler(float(rg), w_dim=7)
            if np.any(w < lo) or np.any(w > hi):
                rewards.append(r)
                continue
            r += 0.2

            # valid
            w_t = torch.tensor(w, dtype=torch.float32, device=dev).unsqueeze(0)  # [1,7]
            # w_t = torch.tensor(w, dtype=torch.float32, device="cuda").unsqueeze(0)  # [1,7]
            # mlp.freeze()
            with torch.no_grad():
                r += 10 * mlp.predict_torch(w_t).item()
            rewards.append(r)

        except Exception:
            r = -0.1
            rewards.append(r)

    return rewards


# -----------------------------
# Main GRPO training
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", type=str, default="data_500_dt01_N100/dataset_all.jsonl")
    # ap.add_argument("--base_model", type=str, default="unsloth/Llama-3.2-1B-Instruct")
    ap.add_argument("--base_model", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--out_dir", type=str, default="./grpo-out")
    ap.add_argument("--w_dim", type=int, default=7)
    ap.add_argument("--w_clip_lo", type=float, default=0.0)
    ap.add_argument("--w_clip_hi", type=float, default=1e6)
    ap.add_argument("--test_split", type=float, default=0.2)

    # GRPO-specific
    ap.add_argument("--num_generations", type=int, default=4, help="Number of completions per prompt")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.7)

    # Training
    ap.add_argument("--num_train_epochs", type=int, default=3)
    ap.add_argument("--per_device_train_batch_size", type=int, default=2)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=0)

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

    # print(train_ds[0]['prompt'])

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args.base_model,
        # pretrained_model_name_or_path=model_id,
        attn_implementation="sdpa",  # Change to Flash Attention if GPU has support
        dtype="float32",  # Change to bfloat16 if GPU has support
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,  # Load the model in 4-bit precision to save memory
            bnb_4bit_compute_dtype=torch.float16,  # Data type used for internal computations in quantization
            bnb_4bit_use_double_quant=True,  # Use double quantization to improve accuracy
            bnb_4bit_quant_type="nf4"  # Type of quantization. "nf4" is recommended for recent LLMs
        ),
        device_map={"": 0},                # <-- critical: pin model to GPU0
    )

    # You may need to update `target_modules` depending on the architecture of your chosen model.
    # For example, different LLMs might have different attention/projection layer names.
    peft_config = LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", ],
    )

    # ---- GRPO Config ----
    # Configure training arguments using GRPOConfig
    training_args = GRPOConfig(
        # Training schedule / optimization
        learning_rate=2e-5,  # Learning rate for the optimizer
        # num_train_epochs=5,
        max_steps=500,  # Number of dataset passes. For full trainings, use `num_train_epochs` instead

        # Parameters that control GRPO training (you can adapt them)
        per_device_train_batch_size=8,
        max_completion_length=256,  # default: 256               # Max completion length produced during training
        num_generations=8,
        # default: 8                         # Number of generations produced during trainig for comparison

        # Optimizations
        optim="paged_adamw_8bit",  # Optimizer
        use_liger_kernel=True,  # Enable Liger kernel optimizations for faster training

        # Parameters related to reporting and saving
        output_dir=args.out_dir,  # Where to save model checkpoints and logs
        logging_steps=10,  # Log training metrics every N steps
        report_to="trackio",  # Experiment tracking tool
        # trackio_space_id=args.out_dir,  # HF Space where the experiment tracking will be saved
        log_completions=False,  # Return model completions during training

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
    trainer = GRPOTrainer(
        model=model,
        # reward_funcs=[think_format_reward, reasoning_accuracy_reward],
        # reward_funcs=[reward_num_unique_letters],
        reward_funcs = [weights_reward],
        args=training_args,
        train_dataset=train_ds,
        peft_config=peft_config,
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
    trainer.push_to_hub(dataset_name='data_500_dt01_N100')

if __name__ == "__main__":
    main()