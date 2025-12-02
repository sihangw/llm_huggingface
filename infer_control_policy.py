#!/usr/bin/env python3
import os
import unsloth
import argparse, torch
from transformers import TextStreamer
from unsloth import FastLanguageModel

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

SYSTEM_PROMPT = r"""
you read control specification and objective function and constraints, give me a control policy
"""

def _get_eos_ids(tokenizer):
    ids = {tokenizer.eos_token_id}
    for tok in ["<|eot_id|>", "<|eom_id|>", "<|end|>", "<|im_end|>"]:
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is not None and tid != tokenizer.unk_token_id and tid != -1:
            ids.add(tid)
    return list(ids)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, default="./sft-output")
    ap.add_argument("--max_seq_len", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.7)   # was 0.2
    ap.add_argument("--top_p", type=float, default=0.9)         # replace min_p
    ap.add_argument("--max_new_tokens", type=int, default=256)
    args = ap.parse_args()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=args.max_seq_len,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    eos_ids = _get_eos_ids(tokenizer)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    print("Interactive mode. Type mission text. Commands: /exit")
    while True:
        try:
            user_text = input("Mission> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not user_text or user_text.lower() == "/exit":
            print("Bye.")
            break

        # match training: NO quotes around mission
        messages.append({"role": "user", "content": f"Mission: {user_text}"})

        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)

        print("Assistant:")
        streamer = TextStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,   # hide <|eot_id|> etc.
        )
        with torch.no_grad():
            _ = model.generate(
                input_ids=inputs,
                streamer=streamer,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,                 # was False → can instantly emit EOT
                temperature=args.temperature,
                top_p=args.top_p,
                use_cache=True,
                eos_token_id=eos_ids,           # allow stopping on chat EOT tokens
                pad_token_id=tokenizer.pad_token_id,
            )
        print("-" * 60)

        # keep only system after each round (tiny context)
        messages = [messages[0]]

if __name__ == "__main__":
    main()