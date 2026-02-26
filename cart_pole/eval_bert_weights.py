import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, AutoTokenizer
from datasets import load_dataset
from bert_weights import EncoderToVector, build_input_text


# load
ckpt_dir = "./bert-weights_10000"   # your output_dir
# ckpt_dir = "./bert-weights_500"   # your output_dir
# model = EncoderToVector(model_name="microsoft/deberta-v3-base", w_dim=7, nonnegative=False, pooling="mean")
model = EncoderToVector(model_name="distilbert-base-uncased", w_dim=7, pooling="mean")

# model.load_state_dict(torch.load(f"{ckpt_dir}/model.safetensors", map_location="cpu"))  # if saved this way

from safetensors.torch import load_file

# state = load_file("./bert-weights_500/model.safetensors")
state = load_file("./bert-weights_10000/model.safetensors")

model.load_state_dict(state, strict=False)

model.eval()

tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, use_fast=True)

# load eval ds the same way you built it (IMPORTANT)
# ds = load_dataset("json", data_files="data_500_dt05_N20_sampling/dataset_all.jsonl", split="train")
ds = load_dataset("json", data_files="data_10000_dt05_N20_grpo/dataset_all.jsonl", split="train")

ds = ds.train_test_split(test_size=0.2, seed=42)
eval_raw = ds["test"]

def build_text(row, w_dim=7):
    return build_input_text(row["prompt_sys"], row["prompt_score"], w_dim=w_dim)

def get_w(row):
    return np.asarray(row.get("best_w", row.get("w")), dtype=np.float32)

# take a few examples
for i in range(10):
    row = eval_raw[i]
    text = build_text(row)
    enc = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")

    with torch.no_grad():
        y_hat = model(**enc)["logits"].cpu().numpy()[0]

    # If you trained in log-space:
    eps = 1e-12
    # w_hat = np.exp(y_hat) - eps
    w_hat = y_hat
    w_hat = np.clip(w_hat, 0.0, 1.0)

    w_true = get_w(row)

    print("----", i, "----")
    print("true:", np.round(w_true, 4))
    print("pred:", np.round(w_hat, 4))
    print("abs err:", np.round(np.abs(w_hat - w_true), 4))