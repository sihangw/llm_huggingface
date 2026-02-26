import dill
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# LOAD instances directly for reward
def load_instances(path: str) -> dict:
    with open(path, "rb") as f:
        instances = dill.load(f)
    print(f"Loaded {len(instances)} instances")
    return instances


# X = np.load("grpo_eval_scores/X_2.npy")
# y = np.load("grpo_eval_scores/Y_llm.npy").reshape(-1)
# valid_idx = np.where(y > 0)[0]
# W = np.load("grpo_eval_scores/W_llm.npy")
# X_valid = X[valid_idx]
# W_valid = W[valid_idx]

X = np.load("sft_eval_scores/X.npy")
y = np.load("sft_eval_scores/Y_llm.npy").reshape(-1)
valid_idx = np.where(y > 0)[0]
W = np.load("sft_eval_scores/W_llm.npy")
X_valid = X[valid_idx]
W_valid = W[valid_idx]


W = np.load("grpo_eval_scores/W_llm.npy")
print(W[:5])



# ===========================
# X (N,17) -> W (N,7) Multi-output Regression with MLP (PyTorch)
# Includes preprocessing:
#   - Standardization (z-score) for X and W (fit on TRAIN only)
#   - Optional Min-Max normalization after standardization (disabled by default)
# Training:
#   - AdamW, Huber loss, gradient clipping
#   - Early stopping + ReduceLROnPlateau scheduler
# Metrics:
#   - MSE (original units), R2 avg + per-dim (original units)
# ===========================
#
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.utils.data import TensorDataset, DataLoader
#
# # ---------------------------
# # Reproducibility
# # ---------------------------
# def set_seed(seed: int = 0):
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#
# # ---------------------------
# # Metrics
# # ---------------------------
# def r2_per_dim(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
#     ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
#     ss_tot = np.sum((y_true - np.mean(y_true, axis=0, keepdims=True)) ** 2, axis=0) + 1e-12
#     return 1.0 - ss_res / ss_tot
#
# def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
#     return float(np.mean((y_true - y_pred) ** 2))
#
# # ---------------------------
# # Preprocessing: Standardize + (optional) MinMax Normalize
# # Fit ONLY on training set to avoid leakage.
# # ---------------------------
# class StandardizeThenOptionalMinMax:
#     """
#     Step 1: Standardize: x -> (x - mean) / std
#     Step 2 (optional): MinMax normalize standardized values to [a,b] using train-set min/max.
#     """
#     def __init__(self, use_minmax: bool = False, a: float = -1.0, b: float = 1.0, eps: float = 1e-8):
#         self.use_minmax = use_minmax
#         self.a = a
#         self.b = b
#         self.eps = eps
#
#         self.mean_ = None
#         self.std_ = None
#         self.zmin_ = None
#         self.zmax_ = None
#
#     def fit(self, X: np.ndarray):
#         X = np.asarray(X, dtype=np.float32)
#         self.mean_ = X.mean(axis=0, keepdims=True)
#         self.std_ = X.std(axis=0, keepdims=True)
#         self.std_[self.std_ < self.eps] = 1.0
#
#         if self.use_minmax:
#             Z = (X - self.mean_) / self.std_
#             self.zmin_ = Z.min(axis=0, keepdims=True)
#             self.zmax_ = Z.max(axis=0, keepdims=True)
#             # avoid division by zero for constant dims
#             span = (self.zmax_ - self.zmin_)
#             span[span < self.eps] = 1.0
#             self._span_ = span
#
#         return self
#
#     def transform(self, X: np.ndarray) -> np.ndarray:
#         X = np.asarray(X, dtype=np.float32)
#         Z = (X - self.mean_) / self.std_
#         if not self.use_minmax:
#             return Z
#         # map standardized Z to [a,b]
#         U = (Z - self.zmin_) / self._span_
#         return self.a + (self.b - self.a) * U
#
#     def inverse_transform(self, Xn: np.ndarray) -> np.ndarray:
#         Xn = np.asarray(Xn, dtype=np.float32)
#         if self.use_minmax:
#             # invert minmax first back to standardized Z
#             U = (Xn - self.a) / (self.b - self.a + self.eps)
#             Z = self.zmin_ + U * self._span_
#         else:
#             Z = Xn
#         # invert standardization
#         return Z * self.std_ + self.mean_
#
# # ---------------------------
# # Model
# # ---------------------------
# class MLP(nn.Module):
#     def __init__(self, in_dim=17, out_dim=7, hidden=(128, 64), dropout=0.2):
#         super().__init__()
#         layers = []
#         d = in_dim
#         for h in hidden:
#             layers += [
#                 nn.Linear(d, h),
#                 nn.ReLU(),
#                 nn.Dropout(dropout),
#             ]
#             d = h
#         layers += [nn.Linear(d, out_dim)]
#         self.net = nn.Sequential(*layers)
#
#     def forward(self, x):
#         return self.net(x)
#
# # ---------------------------
# # Split helpers
# # ---------------------------
# def split_random(X, Y, test_frac=0.15, val_frac=0.15, seed=0):
#     set_seed(seed)
#     n = X.shape[0]
#     idx = np.random.permutation(n)
#     n_test = int(test_frac * n)
#     n_val = int(val_frac * n)
#     test_idx = idx[:n_test]
#     val_idx = idx[n_test:n_test + n_val]
#     train_idx = idx[n_test + n_val:]
#     return train_idx, val_idx, test_idx
#
# def split_blocked(X, Y, train_frac=0.7, val_frac=0.15):
#     # preserves order (useful for time-series / trajectories if ordered)
#     n = X.shape[0]
#     cut_tr = int(train_frac * n)
#     cut_va = int((train_frac + val_frac) * n)
#     train_idx = np.arange(0, cut_tr)
#     val_idx = np.arange(cut_tr, cut_va)
#     test_idx = np.arange(cut_va, n)
#     return train_idx, val_idx, test_idx
#
# # ---------------------------
# # Training loop
# # ---------------------------
# def train_mlp_with_preprocessing(
#     X_valid,
#     W_valid,
#     seed=0,
#     split="random",  # "random" or "blocked"
#     test_frac=0.15,
#     val_frac=0.15,
#     batch_size=256,
#     lr=5e-4,
#     weight_decay=5e-3,
#     max_epochs=1000,
#     patience=200,
#     use_minmax=False,          # set True if you really want standardize + minmax
#     minmax_range=(-1.0, 1.0),  # range used if use_minmax=True
# ):
#
#     set_seed(seed)
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#
#     X = np.asarray(X_valid, dtype=np.float32)
#     Y = np.asarray(W_valid, dtype=np.float32)
#
#     # ----- split -----
#     if split == "random":
#         tr_idx, va_idx, te_idx = split_random(X, Y, test_frac=test_frac, val_frac=val_frac, seed=seed)
#     elif split == "blocked":
#         tr_idx, va_idx, te_idx = split_blocked(X, Y, train_frac=1.0 - test_frac - val_frac, val_frac=val_frac)
#     else:
#         raise ValueError("split must be 'random' or 'blocked'")
#
#     X_tr, Y_tr = X[tr_idx], Y[tr_idx]
#     X_va, Y_va = X[va_idx], Y[va_idx]
#     X_te, Y_te = X[te_idx], Y[te_idx]
#
#     # ----- preprocessing (fit on train only) -----
#     xproc = StandardizeThenOptionalMinMax(
#         use_minmax=use_minmax, a=minmax_range[0], b=minmax_range[1]
#     ).fit(X_tr)
#
#     yproc = StandardizeThenOptionalMinMax(
#         use_minmax=use_minmax, a=minmax_range[0], b=minmax_range[1]
#     ).fit(Y_tr)
#
#     X_tr_n = xproc.transform(X_tr)
#     X_va_n = xproc.transform(X_va)
#     X_te_n = xproc.transform(X_te)
#
#     Y_tr_n = yproc.transform(Y_tr)
#     Y_va_n = yproc.transform(Y_va)
#
#     # ----- baselines -----
#     mean_pred = np.tile(Y_tr.mean(axis=0, keepdims=True), (Y_va.shape[0], 1))
#     print("Val baseline (predict train-mean) MSE:", mse(Y_va, mean_pred))
#
#     # ----- loaders -----
#     train_loader = DataLoader(
#         TensorDataset(torch.from_numpy(X_tr_n), torch.from_numpy(Y_tr_n)),
#         batch_size=batch_size, shuffle=True, drop_last=False
#     )
#     val_loader = DataLoader(
#         TensorDataset(torch.from_numpy(X_va_n), torch.from_numpy(Y_va_n)),
#         batch_size=batch_size, shuffle=False, drop_last=False
#     )
#
#     # ----- model -----
#     model = MLP(in_dim=X.shape[1], out_dim=Y.shape[1], hidden=(128, 64), dropout=0.2).to(device)
#     opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
#     loss_fn = nn.SmoothL1Loss(beta=1.0)  # robust to outliers
#
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         opt, mode="min", factor=0.5, patience=6, min_lr=1e-6,
#     )
#
#     best_val = float("inf")
#     best_state = None
#     bad = 0
#
#     for epoch in range(1, max_epochs + 1):
#         # ---- train ----
#         model.train()
#         tr_losses = []
#         for xb, yb in train_loader:
#             xb, yb = xb.to(device), yb.to(device)
#             pred = model(xb)
#             loss = loss_fn(pred, yb)
#
#             opt.zero_grad()
#             loss.backward()
#             nn.utils.clip_grad_norm_(model.parameters(), 5.0)
#             opt.step()
#
#             tr_losses.append(loss.item())
#
#         tr_loss = float(np.mean(tr_losses))
#
#         # ---- val ----
#         model.eval()
#         va_losses = []
#         with torch.no_grad():
#             for xb, yb in val_loader:
#                 xb, yb = xb.to(device), yb.to(device)
#                 pred = model(xb)
#                 va_losses.append(loss_fn(pred, yb).item())
#         va_loss = float(np.mean(va_losses))
#
#         scheduler.step(va_loss)
#
#         # ---- early stopping ----
#         if va_loss < best_val - 1e-6:
#             best_val = va_loss
#             best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
#             bad = 0
#         else:
#             bad += 1
#
#         if epoch == 1 or epoch % 10 == 0:
#             print(f"Epoch {epoch:03d} | train {tr_loss:.5f} | val {va_loss:.5f} | best {best_val:.5f}")
#
#         if bad >= patience:
#             print(f"Early stopping at epoch {epoch} (best val {best_val:.5f})")
#             break
#
#     if best_state is not None:
#         model.load_state_dict(best_state)
#
#     # ----- test eval (convert back to original units) -----
#     model.eval()
#     with torch.no_grad():
#         pred_te_n = model(torch.from_numpy(X_te_n).to(device)).cpu().numpy()
#
#     # inverse preprocessing for Y
#     pred_te = yproc.inverse_transform(pred_te_n)
#
#     test_mse = mse(Y_te, pred_te)
#     test_r2_dim = r2_per_dim(Y_te, pred_te)
#     test_r2_avg = float(np.mean(test_r2_dim))
#
#     print("\n=== Test metrics (original units) ===")
#     print("MSE:", test_mse)
#     print("R2 (per-dim):", test_r2_dim)
#     print("R2 (avg):", test_r2_avg)
#
#     return {
#         "model": model,
#         "device": device,
#         "xproc": xproc,
#         "yproc": yproc,
#         "indices": {"train": tr_idx, "val": va_idx, "test": te_idx},
#         "test": {"Y_true": Y_te, "Y_pred": pred_te},
#         "metrics": {"mse": test_mse, "r2_per_dim": test_r2_dim, "r2_avg": test_r2_avg},
#     }
#
# # ===========================
# # USAGE
# # ===========================
# # Assumes you already have:
# #   X_valid shape (9260,17)
# #   W_valid shape (9260,7)
# #
# # 1) Standardization only (recommended)
# res = train_mlp_with_preprocessing(
#     X_valid, W_valid,
#     seed=0,
#     split="random",     # try "blocked" if time-series / trajectory ordered
#     use_minmax=False    # standardization only
# )

# 2) If you REALLY want standardize + minmax:
# res = train_mlp_with_preprocessing(
#     X_valid, W_valid,
#     seed=0,
#     split="random",
#     use_minmax=True,
#     minmax_range=(-1.0, 1.0)
# )






