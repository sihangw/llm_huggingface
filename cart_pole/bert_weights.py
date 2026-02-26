import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"   # must be set before torch initializes CUDA
import torch
torch.cuda.set_device(0)



# from typing import Any, Dict, List, Tuple
# import os, json, argparse
# import numpy as np
# import torch
# import torch.nn as nn
# import wandb
#
# from datasets import load_dataset
# from transformers import (
#     AutoTokenizer,
#     AutoModel,
#     Trainer,
#     TrainingArguments,
#     DataCollatorWithPadding,
# )
# from transformers.trainer_utils import set_seed
# from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
#
#
#
# # ----------------------------
# # Prompt formatting (same semantics as yours)
# # ----------------------------
#
# def build_input_text(prompt_sys: str, prompt_score: str, w_dim: int) -> str:
#     semantics = (
#         f"Weight vector format (length {w_dim}):\n"
#         "w = [Ws_track, Ws_safe, Ws_u, Ws_smooth, Wt_track, Wt_safe, Wt_u]\n"
#         "- Ws_* are stage (running) weights.\n"
#         "- Wt_* are terminal weights.\n"
#         "- track penalizes ||x - x_goal||^2\n"
#         "- safe penalizes max(0, ||x||_inf - R_safe)^2\n"
#         "- u penalizes ||u||^2 normalized by u_max\n"
#         "- smooth penalizes ||u - u_prev||^2 (stage only)\n\n"
#         "Weight prior/range hint (for stability; do not output negatives):\n"
#         "Ws_track, Ws_safe in about [10^-1.5, 10^2] (may scale with 1/rho).\n"
#         "Ws_u, Ws_smooth in about [10^-3, 10^0.5].\n"
#         "Wt_track, Wt_safe typically 3x–30x larger than stage weights.\n"
#         "Wt_u typically similar scale to Ws_u.\n"
#     )
#
#     return (
#         "INSTANCE SUMMARY:\n"
#         f"{prompt_sys}\n\n"
#         "SCORE PREFERENCES:\n"
#         f"{prompt_score}\n\n"
#         f"{semantics}\n"
#         "TASK:\n"
#         "Choose MPC cost weights.\n"
#     )
#
#
# # ----------------------------
# # Label helpers (same as yours)
# # ----------------------------
#
# def _get_w_from_row(row: Dict[str, Any]) -> List[float]:
#     if "best_w" in row:
#         w = row["best_w"]
#     elif "w" in row:
#         w = row["w"]
#     # 2) Nested under "llm"
#     elif isinstance(row.get("llm"), dict):
#         llm = row["llm"]
#         print(llm)
#         if "w" in llm and llm["w"] is not None:
#             w = llm["w"]
#         else:
#             raise KeyError("Row missing weight vector in 'llm'")
#     else:
#         raise KeyError("Row missing weight vector key: expected 'best_w' or 'w' or 'llm'")
#     w = np.asarray(w, dtype=float).reshape(-1)
#     return w.tolist()
#
# def _sanitize_w(w: List[float], w_dim: int, w_clip: Tuple[float, float]) -> List[float]:
#     w = np.asarray(w, dtype=float).reshape(-1)
#     if w.size != w_dim:
#         raise ValueError(f"Expected w_dim={w_dim}, got {w.size}")
#     if not np.all(np.isfinite(w)):
#         raise ValueError("Non-finite weights")
#     lo, hi = w_clip
#     w = np.clip(w, lo, hi)
#     return w.tolist()
#
#
# # ----------------------------
# # Encoder regression model
# # ----------------------------
#
# # class EncoderToVector(nn.Module):
# #     def __init__(self, model_name: str, w_dim: int, dropout: float = 0.1, nonnegative: bool = True, pooling: str = "mean"):
# #         super().__init__()
# #         self.encoder = AutoModel.from_pretrained(model_name)
# #         hidden = self.encoder.config.hidden_size
# #         self.pooling = pooling
# #
# #         self.head = nn.Sequential(
# #             nn.Dropout(dropout),
# #             nn.Linear(hidden, 512),
# #             nn.LayerNorm(512),
# #             nn.ReLU(),
# #             nn.Dropout(dropout),
# #             nn.Linear(512, 256),
# #             nn.LayerNorm(256),
# #             nn.ReLU(),
# #             nn.Dropout(dropout),
# #             nn.Linear(256, w_dim),
# #             nn.Tanh(),  # Output [-1, 1]
# #         )
#
#
#
# class EncoderToVector(nn.Module):
#     """
#     Encoder (BERT/DeBERTa/SciBERT/etc.) -> pooled rep -> MLP -> w_dim outputs.
#     Enforce nonnegativity with softplus (optional).
#     """
#     def __init__(
#         self,
#         model_name: str,
#         w_dim: int,
#         dropout: float = 0.1,
#         nonnegative: bool = True,
#         pooling: str = "cls",  # "cls" or "mean"
#     ):
#         super().__init__()
#         self.encoder = AutoModel.from_pretrained(model_name)
#         hidden = self.encoder.config.hidden_size
#         self.pooling = pooling
#         self.nonnegative = nonnegative
#
#         self.head = nn.Sequential(
#             nn.Dropout(dropout),
#             nn.Linear(hidden, hidden),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden, w_dim),
#         )
#         # self.softplus = nn.Softplus()
#         self.out_act = nn.Sigmoid()
#
#     def _pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
#         if self.pooling == "cls":
#             return last_hidden_state[:, 0, :]  # CLS token
#         elif self.pooling == "mean":
#             # masked mean pooling
#             mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
#             summed = (last_hidden_state * mask).sum(dim=1)
#             denom = mask.sum(dim=1).clamp(min=1e-6)
#             return summed / denom
#         else:
#             raise ValueError(f"Unknown pooling: {self.pooling}")
#
#     # # def forward(self, input_ids=None, attention_mask=None, labels=None):
#     # def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
#     #
#     #     out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True, **kwargs)
#     #     pooled = self._pool(out.last_hidden_state, attention_mask)
#     #     pred = self.head(pooled)
#     #
#     #     if self.nonnegative:
#     #         pred = self.softplus(pred)
#     #
#     #     loss = None
#     #     if labels is not None:
#     #         # Huber is often better than plain MSE
#     #         loss_fn = nn.SmoothL1Loss(beta=1.0)  # Huber
#     #         loss = loss_fn(pred, labels)
#     #
#     #     return {"loss": loss, "logits": pred}
#
#     def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
#         # Trainer may pass things like num_items_in_batch; encoder doesn't accept it.
#         # Keep only keys that transformer encoders typically accept.
#         encoder_kwargs = {}
#         for k in ["token_type_ids", "position_ids", "inputs_embeds", "head_mask"]:
#             if k in kwargs and kwargs[k] is not None:
#                 encoder_kwargs[k] = kwargs[k]
#
#         out = self.encoder(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             return_dict=True,
#             **encoder_kwargs,
#         )
#
#         pooled = self._pool(out.last_hidden_state, attention_mask)
#         pred = self.head(pooled)
#
#         # pred = self.head(pooled)  # [-1, 1]
#         # pred = (pred + 1.0) / 2.0  # Scale to [0, 1]
#         #
#         # loss = None
#         # if labels is not None:
#         #     labels = labels.to(pred.dtype)
#         #     loss = nn.MSELoss()(pred, labels)
#         #
#         # # # if self.nonnegative:
#         # # pred = self.out_act(pred)
#
#         loss = None
#         if labels is not None:
#             labels = labels.to(pred.dtype)
#             loss = nn.SmoothL1Loss(beta=1.0)(pred, labels)
#
#         if loss is not None and loss.dim() == 0:
#             loss = loss.unsqueeze(0)  # make it shape [1]
#         return {"loss": loss, "logits": pred}
#
#
# # class EncoderToVector(nn.Module):
# #     def __init__(self, model_name: str, w_dim: int, pooling: str = "mean"):
# #         super().__init__()
# #         self.encoder = AutoModel.from_pretrained(model_name)
# #         hidden = self.encoder.config.hidden_size
# #         self.pooling = pooling
# #
# #         self.head = nn.Sequential(
# #             nn.Dropout(0.1),
# #             nn.Linear(hidden, 512),
# #             nn.LayerNorm(512),
# #             nn.GELU(),
# #             nn.Dropout(0.1),
# #             nn.Linear(512, 256),
# #             nn.LayerNorm(256),
# #             nn.GELU(),
# #             nn.Linear(256, w_dim),
# #         )
# #
# #     def _pool(self, last_hidden_state, attention_mask):
# #         if self.pooling == "cls":
# #             return last_hidden_state[:, 0]
# #         else:  # mean
# #             mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
# #             summed = (last_hidden_state * mask).sum(dim=1)
# #             denom = mask.sum(dim=1).clamp(min=1e-6)
# #             return summed / denom
# #
# #     def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
# #         encoder_kwargs = {k: v for k, v in kwargs.items()
# #                           if k in ["token_type_ids", "position_ids"]}
# #
# #         out = self.encoder(
# #             input_ids=input_ids,
# #             attention_mask=attention_mask,
# #             **encoder_kwargs
# #         )
# #
# #         pooled = self._pool(out.last_hidden_state, attention_mask)
# #         logits = self.head(pooled)
# #         pred = torch.sigmoid(logits)  # each in (0,1), no sum constraint
# #
# #         loss = None
# #         if labels is not None:
# #             labels = labels.to(pred.dtype)
# #             bce = nn.functional.binary_cross_entropy(pred, labels)
# #             mse = nn.functional.mse_loss(pred, labels)
# #             loss = bce + 0.5 * mse
# #
# #         # # Softmax to ensure sum to 1 (like true weights)
# #         # pred = torch.softmax(logits, dim=-1)
# #         #
# #         # loss = None
# #         # if labels is not None:
# #         #     labels = labels.to(pred.dtype)
# #         #
# #         #     # Use KL divergence (better for distributions)
# #         #     # Treat labels as distribution too
# #         #     labels_normalized = labels / (labels.sum(dim=-1, keepdim=True) + 1e-8)
# #         #
# #         #     # KL divergence
# #         #     kl_loss = torch.nn.functional.kl_div(
# #         #         torch.log(pred + 1e-8),
# #         #         labels_normalized,
# #         #         reduction='batchmean'
# #         #     )
# #         #
# #         #     # Also add MSE to keep predictions close
# #         #     mse_loss = nn.functional.mse_loss(pred, labels)
# #         #
# #         #     loss = kl_loss + 0.5 * mse_loss
# #
# #         return {"loss": loss, "logits": pred}
#
# # ----------------------------
# # Main
# # ----------------------------
#
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--train_jsonl", type=str, default="data_500_dt05_N20_grpo/dataset_all.jsonl")
#     # ap.add_argument("--train_jsonl", type=str, default="data_10000_dt05_N20_grpo/dataset_all.jsonl")
#
#     # ap.add_argument("--train_jsonl", type=str, default="data_500_dt05_N20_sampling/dataset_all.jsonl")
#     # ap.add_argument("--model", type=str, default="microsoft/deberta-v3-base")  # good default
#     ap.add_argument("--model", type=str, default="distilbert-base-uncased")  # good default
#
#     ap.add_argument("--output_dir", type=str, default="./bert-weights_10000")
#
#     ap.add_argument("--max_len", type=int, default=512)  # encoder limit; increase only if needed
#     ap.add_argument("--epochs", type=int, default=50)
#     ap.add_argument("--batch_size", type=int, default=16)
#     ap.add_argument("--lr", type=float, default=2e-5)
#     ap.add_argument("--weight_decay", type=float, default=0.01)
#
#     ap.add_argument("--w_dim", type=int, default=7)
#     ap.add_argument("--w_clip_lo", type=float, default=0.0)
#     ap.add_argument("--w_clip_hi", type=float, default=1.0)
#
#     ap.add_argument("--test_split", type=float, default=0.2)
#     ap.add_argument("--seed", type=int, default=42)
#
#     ap.add_argument("--nonnegative", action="store_true", default=True)
#     ap.add_argument("--pooling", type=str, default="mean", choices=["cls", "mean"])
#
#     args = ap.parse_args()
#     os.makedirs(args.output_dir, exist_ok=True)
#     set_seed(args.seed)
#
#     wandb.init(project="BERT_regress_MPC_weights")
#
#     tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
#
#     # Load dataset
#     ds = load_dataset("json", data_files=args.train_jsonl, split="train")
#     ds = ds.train_test_split(test_size=args.test_split, seed=args.seed)
#     train_ds = ds["train"]
#     eval_ds  = ds["test"]
#
#     w_clip = (args.w_clip_lo, args.w_clip_hi)
#
#     def preprocess(row: Dict[str, Any]) -> Dict[str, Any]:
#         prompt_sys = row.get("prompt_sys", "")
#         prompt_score = row.get("prompt_score", "")
#         trial_id = row.get("trial_id", "")
#         if not prompt_sys:
#             raise KeyError("Row missing 'prompt_sys'")
#         if not prompt_score:
#             raise KeyError("Row missing 'prompt_score'")
#
#         text = build_input_text(prompt_sys, prompt_score, w_dim=args.w_dim)
#
#         w = _sanitize_w(_get_w_from_row(row), w_dim=args.w_dim, w_clip=w_clip)
#         # labels must be float tensor later; store list for datasets
#         enc = tokenizer(
#             text,
#             truncation=True,
#             max_length=args.max_len,
#         )
#         enc["labels"] = w
#         # enc["trial_id"] = trial_id
#         return enc
#
#     train_ds = train_ds.map(preprocess, remove_columns=train_ds.column_names)
#     eval_ds  = eval_ds.map(preprocess, remove_columns=eval_ds.column_names)
#     # print(train_ds[0])
#
#
#     data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
#
#     model = EncoderToVector(
#         model_name=args.model,
#         w_dim=args.w_dim,
#         # dropout=0.1,
#         # nonnegative=args.nonnegative,
#         pooling=args.pooling,
#     )
#
#     # Tell Trainer that labels are floats (vector regression)
#     def compute_metrics(eval_pred):
#         preds, labels = eval_pred
#         preds = np.asarray(preds)
#         labels = np.asarray(labels)
#         mse = float(np.mean((preds - labels) ** 2))
#         mae = float(np.mean(np.abs(preds - labels)))
#         # relative error (avoid div by 0)
#         rel = float(np.mean(np.abs(preds - labels) / (np.abs(labels) + 1e-6)))
#         return {"mse": mse, "mae": mae, "rel_mae": rel}
#
#     training_args = TrainingArguments(
#         output_dir=args.output_dir,
#         per_device_train_batch_size=args.batch_size,
#         per_device_eval_batch_size=args.batch_size,
#         num_train_epochs=args.epochs,
#         learning_rate=args.lr,
#         weight_decay=args.weight_decay,
#         warmup_ratio=0.05,
#         lr_scheduler_type="linear",
#         logging_steps=10,
#         eval_strategy="steps",
#         eval_steps=200,
#         save_strategy="steps",
#         save_steps=200,
#         save_total_limit=2,
#         report_to=["wandb"],
#         fp16=torch.cuda.is_available(),   # ok for encoder training; or bf16 if supported
#         bf16=False,
#         seed=args.seed,
#         remove_unused_columns=False,      # IMPORTANT because our model forward expects labels
#     )
#
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_ds,
#         eval_dataset=eval_ds,
#         tokenizer=tokenizer,
#         data_collator=data_collator,
#         compute_metrics=compute_metrics,
#     )
#
#     trainer.train()
#     trainer.save_model(args.output_dir)
#     tokenizer.save_pretrained(args.output_dir)
#     wandb.finish()
#
#     print("Training complete. Saved to:", args.output_dir)
#
#
# if __name__ == "__main__":
#     main()


from typing import Any, Dict, List, Tuple
import os, json, argparse
import numpy as np
import torch
import torch.nn as nn
import wandb

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from transformers.trainer_utils import set_seed


# ----------------------------
# Prompt formatting (same semantics as yours)
# ----------------------------

def build_input_text(prompt_sys: str, prompt_score: str, w_dim: int) -> str:
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
        "Choose MPC cost weights.\n"
    )


# ----------------------------
# Label helpers (same as yours)
# ----------------------------

def _get_w_from_row(row: Dict[str, Any]) -> List[float]:
    if "best_w" in row:
        w = row["best_w"]
    elif "w" in row:
        w = row["w"]
    else:
        raise KeyError("Row missing weight vector key: expected 'best_w' or 'w'")
    w = np.asarray(w, dtype=float).reshape(-1)
    return w.tolist()

def _sanitize_w(w: List[float], w_dim: int, w_clip: Tuple[float, float]) -> List[float]:
    w = np.asarray(w, dtype=float).reshape(-1)
    if w.size != w_dim:
        raise ValueError(f"Expected w_dim={w_dim}, got {w.size}")
    if not np.all(np.isfinite(w)):
        raise ValueError("Non-finite weights")
    lo, hi = w_clip
    w = np.clip(w, lo, hi)
    return w.tolist()


# ----------------------------
# Encoder regression model
# ----------------------------

class EncoderToVector(nn.Module):
    """
    Encoder (BERT/DeBERTa/SciBERT/etc.) -> pooled rep -> MLP -> w_dim outputs.
    Enforce nonnegativity with softplus (optional).
    """
    def __init__(
        self,
        model_name: str,
        w_dim: int,
        dropout: float = 0.1,
        nonnegative: bool = True,
        pooling: str = "cls",  # "cls" or "mean"
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.pooling = pooling
        self.nonnegative = nonnegative

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, w_dim),
        )
        self.softplus = nn.Softplus()

    def _pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            return last_hidden_state[:, 0, :]  # CLS token
        elif self.pooling == "mean":
            # masked mean pooling
            mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
            summed = (last_hidden_state * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1e-6)
            return summed / denom
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        pooled = self._pool(out.last_hidden_state, attention_mask)
        pred = self.head(pooled)

        if self.nonnegative:
            pred = self.softplus(pred)

        loss = None
        if labels is not None:
            # Huber is often better than plain MSE
            loss_fn = nn.SmoothL1Loss(beta=1.0)  # Huber
            loss = loss_fn(pred, labels)

        return {"loss": loss, "logits": pred}


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", type=str, default="data_500_dt05_N20_grpo/dataset_all.jsonl")
    # ap.add_argument("--model", type=str, default="microsoft/deberta-v3-base")  # good default
    ap.add_argument("--model", type=str, default="distilbert-base-uncased")  # good default

    ap.add_argument("--output_dir", type=str, default="./bert-reg-weights")

    ap.add_argument("--max_len", type=int, default=512)  # encoder limit; increase only if needed
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)

    ap.add_argument("--w_dim", type=int, default=7)
    ap.add_argument("--w_clip_lo", type=float, default=1e-3)
    ap.add_argument("--w_clip_hi", type=float, default=1e3)

    ap.add_argument("--test_split", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--nonnegative", action="store_true", default=True)
    ap.add_argument("--pooling", type=str, default="mean", choices=["cls", "mean"])

    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    wandb.init(project="BERT_regress_MPC_weights")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    # Load dataset
    ds = load_dataset("json", data_files=args.train_jsonl, split="train")
    ds = ds.train_test_split(test_size=args.test_split, seed=args.seed)
    train_ds = ds["train"]
    eval_ds  = ds["test"]

    w_clip = (args.w_clip_lo, args.w_clip_hi)

    def preprocess(row: Dict[str, Any]) -> Dict[str, Any]:
        prompt_sys = row.get("prompt_sys", "")
        prompt_score = row.get("prompt_score", "")
        if not prompt_sys:
            raise KeyError("Row missing 'prompt_sys'")
        if not prompt_score:
            raise KeyError("Row missing 'prompt_score'")

        text = build_input_text(prompt_sys, prompt_score, w_dim=args.w_dim)

        w = _sanitize_w(_get_w_from_row(row), w_dim=args.w_dim, w_clip=w_clip)
        # labels must be float tensor later; store list for datasets
        enc = tokenizer(
            text,
            truncation=True,
            max_length=args.max_len,
        )
        enc["labels"] = w
        return enc

    train_ds = train_ds.map(preprocess, remove_columns=train_ds.column_names)
    eval_ds  = eval_ds.map(preprocess, remove_columns=eval_ds.column_names)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = EncoderToVector(
        model_name=args.model,
        w_dim=args.w_dim,
        dropout=0.1,
        nonnegative=args.nonnegative,
        pooling=args.pooling,
    )

    # Tell Trainer that labels are floats (vector regression)
    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        preds = np.asarray(preds)
        labels = np.asarray(labels)
        mse = float(np.mean((preds - labels) ** 2))
        mae = float(np.mean(np.abs(preds - labels)))
        # relative error (avoid div by 0)
        rel = float(np.mean(np.abs(preds - labels) / (np.abs(labels) + 1e-6)))
        return {"mse": mse, "mae": mae, "rel_mae": rel}

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=0.05,
        lr_scheduler_type="linear",
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        report_to=["wandb"],
        fp16=torch.cuda.is_available(),   # ok for encoder training; or bf16 if supported
        bf16=False,
        seed=args.seed,
        remove_unused_columns=False,      # IMPORTANT because our model forward expects labels
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    wandb.finish()

    print("Training complete. Saved to:", args.output_dir)


if __name__ == "__main__":
    main()
