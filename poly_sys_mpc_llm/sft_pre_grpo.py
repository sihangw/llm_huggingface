import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"   # must be set before torch initializes CUDA
import re
import dill
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
from trl import SFTConfig, SFTTrainer

from ilqr_mpc import PolyDynamicsWithJac, QuadCostConfig
from instance_sampler import ExperimentSampler, ExperimentSamplerConfig
from weights_search_ilqr import run_weight_search_for_instance_fast_ilqr, make_fast_cost_from_w
from CompositeScore import CompositeScoreConfig, CompositeTrajectoryScorer
from instance_prompt import build_instance_prompt, build_beta_prompt_simple
from ilqr_mpc import ILQRMPCConfig, FastILQRMPC, run_closed_loop_fast_ilqr_mpc, FastQuadraticCost

SYSTEM_MSG = (
    "You are an expert in nonlinear control and MPC."
    "You MUST return ONLY valid JSON and nothing else."
    "No explanation, no markdown, no code blocks, no extra text."
    "Your ENTIRE response must be exactly this format:"
    '{"w": [w1, w2, w3, w4, w5, w6, w7]}'
    "Where w1-w7 are non-negative numbers."
    "DO NOT add any text before or after the JSON."
)

def build_user_msg(prompt_sys: str, prompt_score: str, w_dim: int) -> str:
    semantics = (
        f"w = [Ws_track, Ws_safe, Ws_u, Ws_smooth, Wt_track, Wt_safe, Wt_u]"
        f"All {w_dim} values must be >= 0."
    )
    return (
        "INSTANCE SUMMARY:"
        f"{prompt_sys}"
        "SCORE PREFERENCES:"
        f"{prompt_score}"
        f"{semantics}"
        "Choose w to maximize the final score."
        "OUTPUT FORMAT (STRICT):"
        f"Return ONLY JSON: {{\"w\": [..{w_dim} numbers..]}}"
    )

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
        {"role": "assistant", "content": assistant},
    ]
    return {"messages": messages}

def main():
    ap = argparse.ArgumentParser()
    # ap.add_argument("--train_jsonl", type=str, default="data_500_dt05_N20_grpo/dataset_all.jsonl")
    # ap.add_argument("--train_instances", type=str, default="data_500_dt05_N20_grpo/instances.dill")
    ap.add_argument("--train_jsonl", type=str, default="data_10000_dt05_N20_sampling/dataset_all.jsonl")
    ap.add_argument("--train_instances", type=str, default="data_10000_dt05_N20_sampling/instances.dill")
    # ap.add_argument("--base_model", type=str, default="unsloth/Llama-3.2-1B-Instruct")
    ap.add_argument("--base_model", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    # ap.add_argument("--base_model", type=str, default="./sft-weights-10000-run2")
    # ap.add_argument("--out_dir", type=str, default="./sft-pre-grpo-10000-ranked")
    ap.add_argument("--out_dir", type=str, default="./sft-pre-grpo-10000-sampling")
    # ap.add_argument("--out_dir", type=str, default="./sft-grpo-500-for-comp")
    ap.add_argument("--w_dim", type=int, default=7)
    ap.add_argument("--w_clip_lo", type=float, default=0.0)
    ap.add_argument("--w_clip_hi", type=float, default=1e6)
    ap.add_argument("--test_split", type=float, default=0.1)

    # GRPO-specific
    ap.add_argument("--num_generations", type=int, default=8, help="Number of completions per prompt")
    ap.add_argument("--max_new_tokens", type=int, default=12)
    ap.add_argument("--temperature", type=float, default=0.7)

    # Training
    ap.add_argument("--num_train_epochs", type=int, default=5)
    ap.add_argument("--per_device_train_batch_size", type=int, default=8)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=4)
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-5)
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


    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        attn_implementation="sdpa",  # Change to Flash Attention if GPU has support
        dtype=torch.float16,  # Change to bfloat16 if GPU has support
        use_cache=True,  # Whether to cache attention outputs to speed up inference
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,  # Load the model in 4-bit precision to save memory
            bnb_4bit_compute_dtype=torch.float16,  # Data type used for internal computations in quantization
            bnb_4bit_use_double_quant=True,  # Use double quantization to improve accuracy
            bnb_4bit_quant_type="nf4"  # Type of quantization. "nf4" is recommended for recent LLMs
        )
    )

    peft_config = LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", ],
    )

    training_args = SFTConfig(
        # Training schedule / optimization
        per_device_train_batch_size=1,  # Batch size per GPU
        gradient_accumulation_steps=4,
        # Gradients are accumulated over multiple steps → effective batch size = 2 * 8 = 16
        warmup_steps=5,
        # num_train_epochs = 1,               # Number of full dataset passes. For shorter training, use `max_steps` instead (this case)
        max_steps=80,
        learning_rate=2e-4,  # Learning rate for the optimizer
        optim="paged_adamw_8bit",  # Optimizer

        # Logging / reporting
        logging_steps=1,  # Log training metrics every N steps
        report_to="trackio",  # Experiment tracking tool
        # trackio_space_id=args.out_dir,  # HF Space where the experiment tracking will be saved
        output_dir=args.out_dir,  # Where to save model checkpoints and logs

        max_length=32,  # Maximum input sequence length
        # use_liger_kernel=True,  # Enable Liger kernel optimizations for faster training
        # activation_offloading=True,  # Offload activations to CPU to reduce GPU memory usage

        # Hub integration
        push_to_hub=True,  # Automatically push the trained model to the Hugging Face Hub
        # The model will be saved under your Hub account in the repository named `output_dir`
    )


    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        peft_config=peft_config
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

    # Save merged model
    trainer.save_model(args.out_dir)
    trainer.push_to_hub(dataset_name="sft-pre-grpo-500")

if __name__ == "__main__":
    main()




