import argparse, json, sys
from typing import Optional
import torch
from transformers import TextStreamer
from unsloth import FastLanguageModel  # adjust import if needed


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

def _get_eos_ids(tokenizer):
    ids = {tokenizer.eos_token_id}
    for tok in ["<|eot_id|>", "<|eom_id|>", "<|end|>", "<|im_end|>"]:
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is not None and tid != tokenizer.unk_token_id and tid != -1:
            ids.add(tid)
    return list(ids)

def build_inputs(tokenizer, system_prompt: str, mission_text: str, device):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": f"Mission: {mission_text}"},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)
    return inputs

def generate_answer(model, tokenizer, eos_ids, mission_text: str,
                    max_new_tokens: int, temperature: float, top_p: float,
                    do_sample: bool) -> str:
    inputs = build_inputs(tokenizer, SYSTEM_PROMPT, mission_text, model.device)
    input_len = inputs.shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            use_cache=True,
            eos_token_id=eos_ids,
            pad_token_id=tokenizer.pad_token_id,
        )

    gen_ids = output_ids[0, input_len:]               # strip the prompt
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return text

def run_file(model, tokenizer, eos_ids, inp_path: str, out_path: str,
             max_new_tokens: int, temperature: float, top_p: float,
             do_sample: bool):
    # Write JSONL as we go (good for long runs)
    with open(inp_path, "r", encoding="utf-8") as fin, \
            open(out_path, "w", encoding="utf-8") as fout:
        n = 0
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                # If it isn't JSON (rare), treat the whole line as the mission text
                ex = {"mission": line}

            # Accept several common keys; default to "mission"
            mission: Optional[str] = ex.get("mission") or ex.get("Mission") or ex.get("text") or ex.get("prompt")
            if mission is None:
                continue

            answer_text = generate_answer(
                model, tokenizer, eos_ids,
                mission_text=mission,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
            )

            # --- merge answer fields into top-level (no "answer" key) ---
            ans_obj = json.loads(answer_text)  # model output must be JSON
            out_row = {"mission": mission}
            out_row.update(ans_obj)  # <-- merge fields

            # passthrough optional id if present
            if "id" in ex:
                out_row["id"] = ex["id"]

            fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            n += 1

    print(f"Generated {n} answers → {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, default="./sft-output")
    ap.add_argument("--max_seq_len", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--do_sample", action="store_true", help="Enable sampling (default off in dataset mode)")
    ap.add_argument("--input_jsonl", type=str, default="data/test_missions_200.jsonl", help="If set, run in dataset mode reading missions from this JSONL.")
    ap.add_argument("--output_jsonl", type=str, default="data/test_missions_pred.jsonl", help="Where to write answers JSONL in dataset mode.")
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

    # ---------- Dataset mode ----------
    if args.input_jsonl:
        run_file(
            model=model,
            tokenizer=tokenizer,
            eos_ids=eos_ids,
            inp_path=args.input_jsonl,
            out_path=args.output_jsonl,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.do_sample,          # default False unless you pass --do_sample
        )
        return

    # ---------- Interactive mode ----------
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    print("Interactive mode. Type mission text. Commands: /exit")
    while True:
        try:
            user_text = input("Mission> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye."); break
        if not user_text or user_text.lower() == "/exit":
            print("Bye."); break

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
            skip_special_tokens=True,
        )
        with torch.no_grad():
            _ = model.generate(
                input_ids=inputs,
                streamer=streamer,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                use_cache=True,
                eos_token_id=eos_ids,
                pad_token_id=tokenizer.pad_token_id,
            )
        print("-" * 60)
        messages = [messages[0]]

if __name__ == "__main__":
    main()
