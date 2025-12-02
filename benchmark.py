#!/usr/bin/env python3
import unsloth
import argparse, torch
from transformers import TextStreamer
from unsloth import FastLanguageModel

SYSTEM_PROMPT = (
    # 'Output JSON only (one line): {"area":"<composition-encoded labels>",'
    # '"objective_function":"<ONE LINE>", "explanation":"<2–3 sentences, optional>"}. '
    # "Labels: {stability_robustness, optimality_efficiency, reachability_controllability, safety_risk}. "
    # "Composition: A > B...[hierarchy]; (A || B) ...[parallel]; A -> B ...[feedback]; or a single label. "
    # "Choose 1–4 labels in priority order; objective is one LaTeX-like line, no constraints/dynamics."
    # "Explanations may be 2–3 sentences and should justify the label composition and the objective's emphasis."
    ''' You are a control area expert'''
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, default="unsloth/Llama-3.2-1B-Instruct")
    ap.add_argument("--max_seq_len", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_new_tokens", type=int, default=4096)
    args = ap.parse_args()

    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=args.max_seq_len,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    # Conversation history: keep just system + last exchange (simple!)
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

        # Wrap exactly like training examples
        messages.append({"role": "user", "content": f'Mission: "{user_text}"'})

        # Build input and stream output
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,   # important for generation
            return_tensors="pt",
        ).to(model.device)

        print("Assistant:")
        streamer = TextStreamer(tokenizer, skip_prompt=True)
        with torch.no_grad():
            _ = model.generate(
                input_ids=inputs,
                streamer=streamer,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                use_cache=True,
            )
        print("-" * 60)

        # Keep only system + last assistant to keep context tiny
        if len(messages) >= 3:
            messages = [messages[0]]  # keep system only

    return

if __name__ == "__main__":
    main()