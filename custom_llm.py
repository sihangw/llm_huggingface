#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional

# ---------------- Config ----------------
@dataclass
class LLMConfig:
    provider: str = "unsloth"           # {"unsloth", "hf"}
    model_path: str = "unsloth/Llama-3.2-1B-Instruct"
    max_seq_len: int = 4096
    max_new_tokens: int = 1024
    temperature: float = 0.2
    top_p: float = 0.95
    load_in_4bit: bool = True          # Unsloth only

# ---------------- Provider base ----------------
class _Provider:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.cfg: Optional[LLMConfig] = None
    def load(self, cfg: LLMConfig) -> None: raise NotImplementedError
    def build_inputs(self, messages: List[Dict[str,str]]):
        raise NotImplementedError
    def generate(self, inputs, stream: bool = False):
        raise NotImplementedError
    def decode(self, outputs) -> str:
        raise NotImplementedError

# ---------------- Unsloth provider ----------------
class _UnslothProvider(_Provider):
    def load(self, cfg: LLMConfig) -> None:
        from unsloth import FastLanguageModel
        import torch  # noqa: F401
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=cfg.model_path,
            max_seq_length=cfg.max_seq_len,
            dtype="bfloat16",
            load_in_4bit=cfg.load_in_4bit,
        )
        FastLanguageModel.for_inference(self.model)
        self.cfg = cfg
    def build_inputs(self, messages):
        return self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
    def generate(self, inputs, stream: bool = False):
        import torch
        from transformers import TextStreamer
        streamer = TextStreamer(self.tokenizer, skip_prompt=True) if stream else None
        with torch.no_grad():
            out = self.model.generate(
                input_ids=inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                use_cache=True,
                streamer=streamer,
            )
        return out
    def decode(self, outputs) -> str:
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# ---------------- HF provider ----------------
class _HFProvider(_Provider):
    def load(self, cfg: LLMConfig) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch  # noqa: F401
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path, torch_dtype="bfloat16", device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
        if getattr(self.tokenizer, "pad_token", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.cfg = cfg
    def build_inputs(self, messages):
        if hasattr(self.tokenizer, "apply_chat_template"):
            toks = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            )
        else:
            text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages) + "\nASSISTANT:"
            toks = self.tokenizer(text, return_tensors="pt")
        return toks.to(self.model.device)
    def generate(self, inputs, stream: bool = False):
        import torch
        from transformers import TextStreamer
        streamer = TextStreamer(self.tokenizer, skip_prompt=True) if stream else None
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                do_sample=True,
                streamer=streamer,
            )
        return out
    def decode(self, outputs) -> str:
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

_REGISTRY = {"unsloth": _UnslothProvider, "hf": _HFProvider}

# ---------------- Chat session ----------------
class GenerationResult:
    def __init__(self, text: str, raw=None):
        self.text = text
        self.raw = raw

class ChatSession:
    def __init__(self, cfg: LLMConfig):
        if cfg.provider not in _REGISTRY:
            raise ValueError(f"Unknown provider {cfg.provider}")
        self.cfg = cfg
        self.provider = _REGISTRY[cfg.provider]()
        self.provider.load(cfg)
        self.messages: List[Dict[str,str]] = []
        self.set_system("You are a control area expert.")
    def set_system(self, content: str):
        if self.messages and self.messages[0]["role"] == "system":
            self.messages[0]["content"] = content
        else:
            self.messages.insert(0, {"role":"system", "content": content})
    def add_user(self, content: str):
        self.messages.append({"role":"user", "content": content})
    def add_assistant(self, content: str):
        self.messages.append({"role":"assistant", "content": content})
    def clear_history(self, keep_system=True):
        self.messages = self.messages[:1] if (keep_system and self.messages) else []
    # def generate(self, extra_messages: Optional[List[Dict[str,str]]] = None, stream: bool = False) -> GenerationResult:
    #     msgs = self.messages + (extra_messages or [])
    #     inputs = self.provider.build_inputs(msgs)
    #     out = self.provider.generate(inputs, stream=stream)
    #     text = self.provider.decode(out)
    #     if "ASSISTANT:" in text:
    #         text = text.split("ASSISTANT:")[-1].strip()
    #     return GenerationResult(text=text, raw=out)
    def generate(self, extra_messages: Optional[List[Dict[str,str]]] = None, stream: bool = False) -> GenerationResult:
        msgs = self.messages + (extra_messages or [])
        inputs = self.provider.build_inputs(msgs)

        # 1) Length of the prompt (in tokens)
        if isinstance(inputs, dict):
            prompt_len = inputs["input_ids"].shape[-1]
        else:
            prompt_len = inputs.shape[-1]

        # 2) Generate (streaming prints only the completion, but the returned tensor still
        #    contains prompt+completion; we'll slice next)
        out = self.provider.generate(inputs, stream=stream)

        # 3) Get the full token sequence from HF/Unsloth
        seq = out.sequences[0] if hasattr(out, "sequences") else out[0]

        # 4) Keep ONLY generated tokens (drop the prompt)
        gen_ids = seq[prompt_len:]

        # 5) Decode only completion tokens
        text = self.provider.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        return GenerationResult(text=text, raw=out)