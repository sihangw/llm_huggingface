#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Minimal interactive inference with a trained (LoRA) model checkpoint.
- Works with base models or your fine-tuned folder (SFT or LoRA/QLoRA).
- If the folder contains PEFT adapters (adapter_config.json), they’ll be loaded automatically.
"""

import argparse, os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel

CONTROL_SPEC_SYSTEM = """
Output JSON only (one line): {"area":"<composition-encoded labels>","objective_function":"<ONE LINE>"}.
Labels: {stability_robustness, optimality_efficiency, reachability_controllability, safety_risk}.
Composition: A > B...[hierarchy]; (A || B) ...[parallel]; A -> B ...[feedback]; or a single label.
Choose 1–4 labels in priority order; objective is one LaTeX-like line, no constraints/dynamics.
"""

def render_prompt(tokenizer, user_text):
    msgs = [
        {"role": "system", "content": CONTROL_SPEC_SYSTEM},
        {"role": "user", "content": f'Mission: "{user_text}"'}
    ]
    # Ensure pad token is set (typical for Llama: pad == eos)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    chat_str = tokenizer.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=False
    )
    enc = tokenizer(
        chat_str, return_tensors="pt", padding=True, truncation=True
    )
    return enc  # dict: input_ids, attention_mask

def load_model_and_tokenizer(model_path_or_id: str):
    tok = AutoTokenizer.from_pretrained(
        model_path_or_id, token=os.environ.get("HF_TOKEN"), trust_remote_code=True
    )
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        model_path_or_id,
        token=os.environ.get("HF_TOKEN"),
        trust_remote_code=True,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    # Auto-load LoRA/QLoRA adapters if present in the folder
    adapters_path = os.path.join(model_path_or_id, "adapter_config.json")
    if os.path.exists(adapters_path):
        base = PeftModel.from_pretrained(base, model_path_or_id)
        base.eval()

    return tok, base

def main():
    ap = argparse.ArgumentParser()
    # ap.add_argument("--model", type=str, required=True, help="Path to fine-tuned folder or HF model id")
    ap.add_argument("--model", type=str, default="./sft-output",
                    help="Path to fine-tuned folder or HF model id")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model: {args.model} (device: {device})")
    tokenizer, model = load_model_and_tokenizer(args.model)

    # Global default: greedy decoding (prevents sampling surprises)
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = True

    # Per-call config (overrides always win; keeps things explicit)
    gen_cfg = GenerationConfig(
        bos_token_id=tokenizer.bos_token_id, do_sample=False, temperature=None, top_p=None, top_k=None,
        max_new_tokens=args.max_new_tokens, repetition_penalty=1.05,
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
    )

    print("\nInteractive chat. Type 'exit' or 'quit' to end.\n")
    while True:
        try:
            user_in = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if user_in.lower() in {"exit", "quit"}:
            print("Bye!")
            break

        batch = render_prompt(tokenizer, user_in)
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)

        with torch.inference_mode():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=gen_cfg,
                return_dict_in_generate=True,
            )

        # Slice off the prompt robustly (handles padding)
        prompt_len = attention_mask.sum(dim=1)[0].item()
        gen_tokens = out.sequences[0, prompt_len:]
        text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        print("\nAssistant:", text, "\n")


if __name__ == "__main__":
    main()