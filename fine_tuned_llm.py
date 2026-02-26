#!/usr/bin/env python
# -*- coding: utf-8 -*-
import wandb, re
import argparse, os, torch, json
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, standardize_sharegpt, train_on_responses_only

from trl import SFTTrainer
from datasets import load_dataset
from transformers import TrainingArguments, DataCollatorForSeq2Seq

local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
is_ddp = local_rank != -1  # True when launched via torchrun/accelerate

# def formatting_prompts_func(examples):
#     convos = examples["messages"]
#     texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
#     return {"text": texts, }

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
#     "disturbance_radius_eps": "<if worst_case/minmax>",
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
#
# ## Rejection rule
# If the mission is too vague to choose families, output a single-family baseline (tracking or energy_min) AND state one missing cue in "parameters.notes".
#
# Return ONLY the JSON object.
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

# def _record_to_messages(rec, include_system=True):
#     """Build chat messages from a structured training record."""
#     # 1) user turn (mission only — short and clean)
#     user_msg = {
#         "role": "user",
#         "content": f"Mission: {rec['mission']}"
#     }
#
#     # 2) assistant target: strict one-line JSON composed from fields
#     assistant_obj = {
#         "objective_function": rec["objective_function"],
#         "constraints": rec["constraints"],
#         "risk_model": rec["risk_model"],
#         "objective_families": rec["objective_families"],
#         "composition": rec["composition"],
#         "parameters": rec.get("parameters", {}),
#         "structure": rec.get("structure", {}),
#         # keep the short rationale if you want it as part of target;
#         # remove this field if you prefer the model NOT to emit explanations in training.
#         "explanation": rec.get("explanation", "")
#     }
#     assistant_text = json.dumps(assistant_obj, ensure_ascii=False)
#
#     messages = []
#     if include_system:
#         messages.append({"role": "system", "content": SYSTEM_PROMPT})
#     messages.append(user_msg)
#     messages.append({"role": "assistant", "content": assistant_text})
#     return messages
#
# def formatting_prompts_func(examples):
#     """
#     Converts either:
#       - examples with `messages` already present, or
#       - structured records with fields: mission, objective_function, constraints, ...
#     into a single `text` string using tokenizer.apply_chat_template.
#     """
#     out = []
#     for i in range(len(examples[next(iter(examples))])):  # iterate row-wise
#         if "messages" in examples and examples["messages"][i] is not None:
#             # Already chat-shaped → use as-is
#             conv = examples["messages"][i]
#         else:
#             # Structured → build messages
#             rec = {k: examples[k][i] for k in examples.keys()}
#             conv = _record_to_messages(rec, include_system=False)
#
#         # IMPORTANT: For SFT (teacher forcing), add_generation_prompt=False
#         # (At inference you’ll set it to True.)
#         text = tokenizer.apply_chat_template(
#             conv,
#             tokenize=False,
#             add_generation_prompt=False
#         )
#         out.append(text)
#     return {"text": out}

# def formatting_prompts_func(examples, include_system=True):
#     texts = []
#     n = len(examples["mission"])
#     for i in range(n):
#         msgs = []
#         if include_system:
#             msgs.append({"role": "system", "content": SYSTEM_PROMPT})
#         msgs.append({"role": "user", "content": f"Mission: {examples['mission'][i]}"})
#
#         target = {
#             "objective_function": examples["objective_function"][i],
#             "constraints": examples["constraints"][i],
#         }
#         msgs.append({"role": "assistant", "content": json.dumps(target, ensure_ascii=False)})
#
#         text = tokenizer.apply_chat_template(
#             msgs, tokenize=False, add_generation_prompt=False
#         )
#         texts.append(text)
#     return {"text": texts}

def _oneline(s: str) -> str:
    """Collapse whitespace/newlines inside a field."""
    return re.sub(r"\s+", " ", s.strip())

def formatting_prompts_func(examples, include_system=True):
    texts = []
    n = len(examples["mission"])
    for i in range(n):
        msgs = []
        if include_system:
            msgs.append({"role": "system", "content": SYSTEM_PROMPT})

        # user turn
        mission_i = _oneline(examples["mission"][i])
        msgs.append({"role": "user", "content": f"Mission: {mission_i}"})

        # assistant target: ONE-LINE JSON with the three keys
        target = {
            "objective_function": _oneline(examples["objective_function"][i]),
            "constraints": _oneline(examples["constraints"][i]),
            "areas": list(map(int, examples["areas"][i])),
        }

        # compact one-line JSON (no spaces after commas/colons)
        assistant_json = json.dumps(target, ensure_ascii=False, separators=(",", ":"))

        msgs.append({"role": "assistant", "content": assistant_json})

        text = tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=False,  # for SFT data creation
        )
        texts.append(text)
    return {"text": texts}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--train_jsonl", type=str, required=True)
    # parser.add_argument("--eval_jsonl", type=str, required=True)
    parser.add_argument("--train_jsonl", type=str, default="data/train_missions_1000_balanced.jsonl")
    # parser.add_argument("--train_jsonl", type=str, default="data/robotics_train.jsonl")
    # parser.add_argument("--train_jsonl", type=str, default="data/minimal_train.jsonl")
    # parser.add_argument("--train_jsonl", type=str, default="data/minimal_clean.jsonl")
    # parser.add_argument("--eval_jsonl", type=str, default="data/missions_val.jsonl")
    parser.add_argument("--model", type=str, default="unsloth/Llama-3.2-1B-Instruct")
    parser.add_argument("--output_dir", type=str, default="./sft-output")
    parser.add_argument("--bf16", action="store_true", default=True, help="Use bfloat16 if available")
    parser.add_argument("--fp16", action="store_true", default=False, help="Use float16")
    # parser.add_argument("--use_lora", action="store_true", default=True, help="Enable LoRA/QLoRA")
    parser.add_argument("--qlora", action="store_true", default=True, help="Enable 4-bit QLoRA with bitsandbytes")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--save_steps", type=int, default=50)
    # NEW: choose training target
    parser.add_argument("--responses_only", action="store_true", default=False,
                        help="If set, mask loss to assistant responses only.")
    args = parser.parse_args()

    run = wandb.init(project="Llama-fine_tuned_missions")

    # -------- Model + (Q)LoRA + Tokenizer --------
    print(f"Loading model: {args.model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,  # or choose "unsloth/Llama-3.2-1B-Instruct"
        max_seq_length=args.max_seq_len,
        dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None),
        # QLoRA / bitsandbytes options:
        load_in_4bit=bool(args.qlora),          # == bnb_config.load_in_4bit
        # device_map=None if is_ddp else "auto",
        device_map="cuda:0",
    )
    # LoRA config based on QLoRA paper & Sebastian Raschka experiment
    if args.qlora:
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj", ],
            lora_alpha=32,
            lora_dropout=0,  # Supports any, but = 0 is optimized
            bias="none",  # Supports any, but = "none" is optimized
            # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
            use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
            random_state=47,
            use_rslora=False,  # We support rank stabilized LoRA
            loftq_config=None,  # And LoftQ
        )
    print(f"Loading tokenizer: {args.model}")
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

    # -------- Dataset --------
    dataset = load_dataset("json", data_files=args.train_jsonl, split="train")
    # Create a new 80/20 train/test split
    train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]
    train_dataset = standardize_sharegpt(train_dataset)
    train_dataset = train_dataset.map(formatting_prompts_func, batched=True)
    test_dataset = standardize_sharegpt(test_dataset)
    test_dataset = test_dataset.map(formatting_prompts_func, batched=True)

    # print(train_dataset[799]["messages"])

    # -------- Trainer --------
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        validation_data=test_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        dataset_num_proc=2,
        packing=False,  # Can make training 5x faster for short sequences.
        args=TrainingArguments(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            warmup_ratio=0.05,
            num_train_epochs=args.epochs,  # Set this for 1 full training run.
            # max_steps = 60,
            learning_rate=args.lr,
            fp16=args.fp16 and not args.bf16,
            bf16=args.bf16,
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=args.output_dir,
            # report_to="wandb"
        ),
    )

    # ---- OPTIONAL: mask to responses only  ----
    if args.responses_only:
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
            response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
        )

        # Verify masking is actually done:
        print(tokenizer.decode(trainer.train_dataset[5]["input_ids"]))
        space = tokenizer(" ", add_special_tokens=False).input_ids[0]
        print(tokenizer.decode([space if x == -100 else x for x in trainer.train_dataset[5]["labels"]]))

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
