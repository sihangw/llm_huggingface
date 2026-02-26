# sft_weights_unsloth.py
# from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"   # must be set before torch initializes CUDA
import torch
torch.cuda.set_device(0)

from typing import Any, Dict, List, Tuple

import numpy as np
import wandb, re
import argparse, os, torch, json
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only, standardize_sharegpt

from trl import SFTTrainer
from datasets import load_dataset
from transformers import TrainingArguments, DataCollatorForSeq2Seq

# local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
# is_ddp = local_rank != -1  # True when launched via torchrun/accelerate

# ----------------------------
# Prompts
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

def build_user_msg(prompt_sys: str, prompt_score: str, w_dim: int) -> str:
    # Keep this consistent with your downstream parser and weight sampler semantics.
    semantics = (
        f"Weight vector format (length {w_dim}):\n"
        "w = [Ws_track, Ws_safe, Ws_u, Ws_smooth, Wt_track, Wt_safe, Wt_u]\n"
        "- Ws_* are stage (running) weights.\n"
        "- Wt_* are terminal weights.\n"
        "- track penalizes ||x - x_goal||^2\n"
        "- safe penalizes max(0, ||x||_inf - R_safe)^2\n"
        "- u penalizes ||u||^2 normalized by u_max\n"
        "- smooth penalizes ||u - u_prev||^2 (stage only)\n\n"
        "Weight prior/range hint (for stability; do not output negatives):\n"
        "Ws_track, Ws_safe in about [10^-1.5, 10^2] (may scale with 1/rho).\n"
        "Ws_u, Ws_smooth in about [10^-3, 10^0.5].\n"
        "Wt_track, Wt_safe typically 3x–30x larger than stage weights.\n"
        "Wt_u typically similar scale to Ws_u.\n"
    )
    return (
        "INSTANCE SUMMARY:\n"
        f"{prompt_sys}\n\n"
        "SCORE PREFERENCES:\n"
        f"{prompt_score}\n\n"
        f"{semantics}\n"
        "TASK:\n"
        "Choose w to maximize the composite score.\n\n"
        "OUTPUT FORMAT (STRICT):\n"
        f"Return ONLY JSON: {{\"w\": [..{w_dim} numbers..]}}\n"
    )


# ----------------------------
# Data helpers
# ----------------------------

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


def format_to_text(batch: Dict[str, Any], tokenizer) -> Dict[str, Any]:
    # batch["messages"] is a list of message lists
    texts = []
    for msgs in batch["messages"]:
        txt = tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append(txt)
    return {"text": texts}


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    # ap.add_argument("--train_jsonl", type=str, default="data_10000_dt05_N20_grpo/dataset_all.jsonl")
    # ap.add_argument("--train_jsonl", type=str, default="data_10000_dt05_N20/dataset_all.jsonl")
    ap.add_argument("--train_jsonl", type=str, default="data_10000_dt05_N20_sampling/dataset_all.jsonl")

    ap.add_argument("--model", type=str, default="unsloth/Llama-3.2-1B-Instruct")
    ap.add_argument("--output_dir", type=str, default="./sft-weights-10000-sampling")

    ap.add_argument("--max_seq_len", type=int, default=2048)
    ap.add_argument("--qlora", action="store_true", default=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)

    ap.add_argument("--w_dim", type=int, default=7)
    ap.add_argument("--w_clip_lo", type=float, default=0.0)
    ap.add_argument("--w_clip_hi", type=float, default=1e6)

    ap.add_argument("--test_split", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)

    # IMPORTANT: response-only loss
    ap.add_argument("--responses_only", action="store_true", default=True)

    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    run = wandb.init(project="Llama_sft_poly_sys")

    # -------- Model + (Q)LoRA + Tokenizer --------
    print(f"Loading model: {args.model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_len,
        dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        load_in_4bit=bool(args.qlora),
        device_map="auto",
    )

    # LoRA config based on QLoRA paper & Sebastian Raschka experiment
    if args.qlora:
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=32,
            lora_dropout=0.0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=args.seed,
            use_rslora=False,
            loftq_config=None,
        )
    print(f"Loading tokenizer: {args.model}")
    # Llama chat template (important)
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.2")

    # -------- Dataset --------
    ds = load_dataset("json", data_files=args.train_jsonl, split="train")
    ds = ds.train_test_split(test_size=args.test_split, seed=args.seed)
    train_ds = ds["train"]
    eval_ds  = ds["test"]

    w_clip = (args.w_clip_lo, args.w_clip_hi)

    # train_ds = standardize_sharegpt(train_ds)
    # train_ds = train_ds.map(formatting_prompts_func, batched=True)
    # test_ds = standardize_sharegpt(test_ds)
    # test_ds = test_ds.map(formatting_prompts_func, batched=True, )

    # Build messages from your existing fields
    train_ds = train_ds.map(
        lambda row: make_messages_from_row(row, w_dim=args.w_dim, w_clip=w_clip),
        remove_columns=train_ds.column_names,
    )
    eval_ds = eval_ds.map(
        lambda row: make_messages_from_row(row, w_dim=args.w_dim, w_clip=w_clip),
        remove_columns=eval_ds.column_names,
    )

    # Convert to text field using chat template
    train_ds = train_ds.map(lambda b: format_to_text(b, tokenizer), batched=True)
    eval_ds  = eval_ds.map(lambda b: format_to_text(b, tokenizer), batched=True)

    # -------- Trainer --------
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        packing=False,
        args=TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            num_train_epochs=args.epochs,
            warmup_ratio=0.05,
            logging_steps=10,
            bf16=torch.cuda.is_available(),
            fp16=False,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=args.seed,
            report_to="none",
        ),
    )

    if args.responses_only:
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
            response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
        )

    # Show current memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    trainer_stats = trainer.train()
    # Finish wandb run
    # The detailed run history is generated when we finish the Weights & Biases run.
    run.finish()

    # @title Show final memory and time stats
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

    # Save final artifacts (including LoRA adapters if used)
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("\nTraining complete. Model saved to:", args.output_dir)

    # FastLanguageModel.for_training(trainer.model)
    #
    # trainer.train()
    #
    # trainer.model.save_pretrained(args.output_dir)
    # tokenizer.save_pretrained(args.output_dir)
    # print("Saved to:", args.output_dir)


if __name__ == "__main__":
    main()
