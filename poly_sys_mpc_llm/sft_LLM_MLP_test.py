import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# -----------------------------
# Model: phi(x) from encoder, then y = W phi(x)
# -----------------------------
class MLPEncoder(nn.Module):
    def __init__(self, d_in: int, d_phi: int = 128, hidden=(256, 256), dropout=0.0):
        super().__init__()
        layers = []
        prev = d_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, d_phi)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # (B, d_phi)


class PhiLinearModel(nn.Module):
    """
    yhat = head(phi(x)) = W^T phi(x) (+ optional bias)
    """
    def __init__(self, d_in: int, d_phi: int = 128, hidden=(256, 256), y_dim: int = 1,
                 dropout=0.0, head_bias: bool = False):
        super().__init__()
        self.encoder = MLPEncoder(d_in, d_phi=d_phi, hidden=hidden, dropout=dropout)
        self.head = nn.Linear(d_phi, y_dim, bias=head_bias)

    def phi(self, x):
        return self.encoder(x)

    def forward(self, x):
        return self.head(self.encoder(x))  # (B, y_dim)


# -----------------------------
# Utility: normalization
# -----------------------------
class Standardizer:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std < 1e-12] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        return (X - self.mean) / self.std

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# -----------------------------
# Trainer wrapper: offline pretrain, then freeze encoder, then online update W
# -----------------------------
class ScoreModel:
    def __init__(
        self,
        d_in: int,
        d_phi: int = 128,
        hidden=(256, 256),
        y_dim: int = 1,
        device: str = "cuda",
        dropout: float = 0.0,
        head_bias: bool = False
    ):
        self.device = device
        self.model = PhiLinearModel(
            d_in=d_in, d_phi=d_phi, hidden=hidden, y_dim=y_dim,
            dropout=dropout, head_bias=head_bias
        ).to(device)

        self.x_scaler = Standardizer()
        self.y_mean = None
        self.y_std = None

    # -------- offline training (encoder + head) --------
    # def pretrain(
    #         self,
    #         X: np.ndarray,
    #         Y: np.ndarray,
    #         epochs: int = 300,
    #         batch_size: int = 256,
    #         lr: float = 1e-3,
    #         weight_decay: float = 0.0,
    #         normalize_y: bool = True,
    #         loss: str = "mse",
    #         log_every: int = 10,  # <-- add this
    # ):
    #     X = np.asarray(X, dtype=np.float32)
    #     Y = np.asarray(Y, dtype=np.float32)
    #     if Y.ndim == 1:
    #         Y = Y.reshape(-1, 1)
    #
    #     Xn = self.x_scaler.fit_transform(X)
    #
    #     if normalize_y:
    #         self.y_mean = Y.mean(axis=0)
    #         self.y_std = Y.std(axis=0)
    #         self.y_std[self.y_std < 1e-12] = 1.0
    #         Yn = (Y - self.y_mean) / self.y_std
    #     else:
    #         self.y_mean = np.zeros((Y.shape[1],), dtype=np.float32)
    #         self.y_std = np.ones((Y.shape[1],), dtype=np.float32)
    #         Yn = Y
    #
    #     ds = TensorDataset(torch.from_numpy(Xn), torch.from_numpy(Yn))
    #     dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
    #
    #     opt = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
    #     if loss == "mse":
    #         loss_fn = nn.MSELoss()
    #     elif loss == "huber":
    #         loss_fn = nn.SmoothL1Loss()
    #     else:
    #         raise ValueError("loss must be 'mse' or 'huber'")
    #
    #     self.model.train()
    #     for ep in range(1, epochs + 1):
    #         total = 0.0
    #         count = 0
    #         for xb, yb in dl:
    #             xb = xb.to(self.device)
    #             yb = yb.to(self.device)
    #             pred = self.model(xb)
    #             l = loss_fn(pred, yb)
    #
    #             opt.zero_grad()
    #             l.backward()
    #             opt.step()
    #
    #             bs = xb.shape[0]
    #             total += float(l.detach().cpu().item()) * bs
    #             count += bs
    #
    #         avg = total / max(count, 1)
    #         if (ep % log_every) == 0 or ep == 1 or ep == epochs:
    #             print(f"[pretrain] epoch {ep:4d}/{epochs}  avg_loss={avg:.6f}")
    #
    #     return self

    @staticmethod
    def _split_train_val(X, Y, val_frac=0.2, seed=0):
        rng = np.random.default_rng(seed)
        idx = np.arange(X.shape[0])
        rng.shuffle(idx)
        n_val = int(round(val_frac * len(idx)))
        va = idx[:n_val]
        tr = idx[n_val:]
        return X[tr], Y[tr], X[va], Y[va]

    def pretrain(
            self,
            X: np.ndarray,
            Y: np.ndarray,
            epochs: int = 200,
            batch_size: int = 256,
            lr: float = 1e-3,
            weight_decay: float = 0.0,
            normalize_y: bool = True,
            loss: str = "mse",  # "mse" or "huber"
            val_frac: float = 0.2,
            split_seed: int = 42,
            log_every: int = 10,
            # scheduler + early stopping
            scheduler_patience: int = 20,  # epochs without val improvement before LR drops
            scheduler_factor: float = 0.5,  # LR *= factor
            min_lr: float = 1e-6,
            early_stop_patience: int = 60,  # epochs without val improvement before stop
            min_delta: float = 1e-5,  # required improvement in val loss
    ):
        """
        Drop-in replacement for OptionBRewardModel.pretrain().
        Includes: train/val split, ReduceLROnPlateau scheduler, early stopping, and best checkpoint restore.
        """

        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.float32)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        # --- split before scaling (scale on train only) ---
        Xtr, Ytr, Xva, Yva = self._split_train_val(X, Y, val_frac=val_frac, seed=split_seed)

        # --- fit X scaler on train only ---
        Xtr_n = self.x_scaler.fit_transform(Xtr)
        Xva_n = self.x_scaler.transform(Xva)

        # --- normalize Y using train only ---
        if normalize_y:
            self.y_mean = Ytr.mean(axis=0)
            self.y_std = Ytr.std(axis=0)
            self.y_std[self.y_std < 1e-12] = 1.0
            Ytr_n = (Ytr - self.y_mean) / self.y_std
            Yva_n = (Yva - self.y_mean) / self.y_std
        else:
            self.y_mean = np.zeros((Y.shape[1],), dtype=np.float32)
            self.y_std = np.ones((Y.shape[1],), dtype=np.float32)
            Ytr_n, Yva_n = Ytr, Yva

        tr_ds = TensorDataset(torch.from_numpy(Xtr_n), torch.from_numpy(Ytr_n))
        va_ds = TensorDataset(torch.from_numpy(Xva_n), torch.from_numpy(Yva_n))
        tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, drop_last=False)
        va_dl = DataLoader(va_ds, batch_size=batch_size, shuffle=False, drop_last=False)

        # optimizer + scheduler
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        if loss == "mse":
            loss_fn = nn.MSELoss()
        elif loss == "huber":
            loss_fn = nn.SmoothL1Loss()
        else:
            raise ValueError("loss must be 'mse' or 'huber'")

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=scheduler_factor, patience=scheduler_patience,
            min_lr=min_lr,
            # verbose=True
        )

        # early stopping state
        best_val = float("inf")
        best_state = None
        bad_epochs = 0

        def _eval(dl):
            self.model.eval()
            tot, n = 0.0, 0
            with torch.no_grad():
                for xb, yb in dl:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    pred = self.model(xb)
                    l = loss_fn(pred, yb)
                    bs = xb.shape[0]
                    tot += float(l.detach().cpu().item()) * bs
                    n += bs
            return tot / max(n, 1)

        for ep in range(1, epochs + 1):
            # ---- train ----
            self.model.train()
            tr_tot, tr_n = 0.0, 0
            for xb, yb in tr_dl:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                pred = self.model(xb)
                l = loss_fn(pred, yb)

                opt.zero_grad()
                l.backward()
                opt.step()

                bs = xb.shape[0]
                tr_tot += float(l.detach().cpu().item()) * bs
                tr_n += bs
            tr_loss = tr_tot / max(tr_n, 1)

            # ---- val ----
            va_loss = _eval(va_dl)

            # scheduler step based on val
            scheduler.step(va_loss)
            cur_lr = opt.param_groups[0]["lr"]

            # logging
            if (ep % log_every) == 0 or ep == 1 or ep == epochs:
                print(f"[pretrain] ep {ep:4d}/{epochs}  train={tr_loss:.6f}  val={va_loss:.6f}  lr={cur_lr:.2e}")

            # early stopping check
            if va_loss < best_val - min_delta:
                best_val = va_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= early_stop_patience:
                    print(f"[pretrain] Early stopping at epoch {ep} (best val={best_val:.6f}).")
                    break

        # restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)
            self.model.eval()
            print(f"[pretrain] Restored best checkpoint (val={best_val:.6f}).")

        return self

    # -------- freeze encoder so only W is learnable --------
    def freeze_encoder(self):
        for p in self.model.encoder.parameters():
            p.requires_grad = False
        return self

    # -------- online update: SGD on head only --------
    def online_update_head_sgd(
        self,
        X_new: np.ndarray,
        Y_new: np.ndarray,
        steps: int = 1,
        lr: float = 1e-2,
        loss: str = "mse",
        l2: float = 0.0
    ):
        """
        Update only the head W using SGD on new data.
        """
        X_new = np.asarray(X_new, dtype=np.float32)
        Y_new = np.asarray(Y_new, dtype=np.float32)
        if Y_new.ndim == 1:
            Y_new = Y_new.reshape(-1, 1)

        Xn = self.x_scaler.transform(X_new)
        Yn = (Y_new - self.y_mean) / self.y_std

        xb = torch.from_numpy(Xn).to(self.device)
        yb = torch.from_numpy(Yn).to(self.device)

        params = list(self.model.head.parameters())
        opt = torch.optim.SGD(params, lr=lr, weight_decay=l2)

        if loss == "mse":
            loss_fn = nn.MSELoss()
        elif loss == "huber":
            loss_fn = nn.SmoothL1Loss()
        else:
            raise ValueError("loss must be 'mse' or 'huber'")

        self.model.train()
        for _ in range(steps):
            pred = self.model(xb)
            l = loss_fn(pred, yb)
            opt.zero_grad()
            l.backward()
            opt.step()

        return float(l.detach().cpu().item())

    # -------- online update: closed-form ridge on head (mini-batch) --------
    @torch.no_grad()
    def online_update_head_ridge(
        self,
        X_new: np.ndarray,
        Y_new: np.ndarray,
        lam: float = 1e-3
    ):
        """
        Closed-form update of head W on new batch using ridge regression:
        minimize ||Phi W - Y||^2 + lam||W||^2, with Phi fixed = encoder(X)
        Replaces the head weight with the ridge solution on this batch.
        """
        self.model.eval()

        X_new = np.asarray(X_new, dtype=np.float32)
        Y_new = np.asarray(Y_new, dtype=np.float32)
        if Y_new.ndim == 1:
            Y_new = Y_new.reshape(-1, 1)

        Xn = self.x_scaler.transform(X_new)
        Yn = (Y_new - self.y_mean) / self.y_std

        xb = torch.from_numpy(Xn).to(self.device)
        yb = torch.from_numpy(Yn).to(self.device)

        Phi = self.model.phi(xb)  # (N, d_phi)
        # Ridge: W = (Phi^T Phi + lam I)^-1 Phi^T Y
        d_phi = Phi.shape[1]
        A = Phi.T @ Phi + lam * torch.eye(d_phi, device=self.device)
        B = Phi.T @ yb
        W = torch.linalg.solve(A, B)  # (d_phi, y_dim)

        # Set head weight to W^T because nn.Linear stores (out_dim, in_dim)
        self.model.head.weight.copy_(W.T)
        if self.model.head.bias is not None:
            self.model.head.bias.zero_()

        return self

    # -------- prediction / reward --------
    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        Xn = self.x_scaler.transform(X)
        xb = torch.from_numpy(Xn).to(self.device)
        self.model.eval()
        pred_n = self.model(xb).detach().cpu().numpy()  # normalized Y
        pred = pred_n * self.y_std + self.y_mean
        if pred.shape[1] == 1:
            # pred = np.clip(pred, 0.0, 1.0)
            return pred[:, 0]
        # pred = np.clip(pred, 0.0, 1.0)
        return pred

    # -------- save/load --------
    # def save(self, path: str):
    #     ckpt = {
    #         "state_dict": self.model.state_dict(),
    #         "x_mean": torch.tensor(self.x_scaler.mean, dtype=torch.float32),
    #         "x_std": torch.tensor(self.x_scaler.std, dtype=torch.float32),
    #         "y_mean": torch.tensor(self.y_mean, dtype=torch.float32),
    #         "y_std": torch.tensor(self.y_std, dtype=torch.float32),
    #     }
    #     torch.save(ckpt, path)  # safe (no numpy pickles)
    #
    # def load(self, path: str):
    #     ckpt = torch.load(path, map_location=self.device)  # safe
    #     self.model.load_state_dict(ckpt["state_dict"])
    #
    #     self.x_scaler.mean = ckpt["x_mean"].cpu().numpy()
    #     self.x_scaler.std = ckpt["x_std"].cpu().numpy()
    #     self.y_mean = ckpt["y_mean"].cpu().numpy()
    #     self.y_std = ckpt["y_std"].cpu().numpy()
    #
    #     self.model.to(self.device)
    #     self.model.eval()
    #     return self

    def save(self, path: str):
        if self.x_scaler.mean is None or self.y_mean is None:
            raise RuntimeError("Model/scalers not initialized. Train/pretrain first.")
        ckpt = {
            "state_dict": self.model.state_dict(),
            "x_mean": torch.tensor(self.x_scaler.mean, dtype=torch.float32),
            "x_std": torch.tensor(self.x_scaler.std, dtype=torch.float32),
            "y_mean": torch.tensor(self.y_mean, dtype=torch.float32),
            "y_std": torch.tensor(self.y_std, dtype=torch.float32),
        }
        torch.save(ckpt, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)  # safe load
        self.model.load_state_dict(ckpt["state_dict"])

        self.x_scaler.mean = ckpt["x_mean"].cpu().numpy()
        self.x_scaler.std = ckpt["x_std"].cpu().numpy()
        self.y_mean = ckpt["y_mean"].cpu().numpy()
        self.y_std = ckpt["y_std"].cpu().numpy()

        self.model.to(self.device)
        self.model.eval()
        return self

    # expose W explicitly
    @property
    def W(self):
        # returns numpy array with shape (y_dim, d_phi)
        return self.model.head.weight.detach().cpu().numpy()

    def phi(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        Xn = self.x_scaler.transform(X)
        xb = torch.from_numpy(Xn).to(self.device)
        self.model.eval()
        with torch.no_grad():
            Phi = self.model.phi(xb).detach().cpu().numpy()
        return Phi

def main():
    X = np.load("data_10000_dt05_N20/run1_X.npy")
    W = np.load("data_10000_dt05_N20/run1_W.npy")
    # Y = np.load("data_10000_dt05_N20/run1_Y.npy")
    Y = np.load("sft_eval_scores/Y_llm.npy")

    mask = (Y != 0)
    X_new = X[mask]
    Y_new = Y[mask]
    W_new = W[mask]


    XW = np.concatenate([X_new, W_new], axis=1)


    # n_train, n_cal, n_test = 8500, 1000, 500
    # n_total = n_train + n_cal + n_test
    # X_train, Y_train = X[:n_train], Y[:n_train]
    # X_cal, Y_cal = X[n_train:n_train + n_cal], Y[n_train:n_train + n_cal]
    # X_test, Y_test = X[n_train + n_cal:n_total], Y[n_train + n_cal:n_total]

    # # 1) split
    # rng = np.random.default_rng(42)
    # idx = np.arange(len(X))
    # rng.shuffle(idx)
    # n_val = int(round(0.2 * len(idx)))
    # va = idx[:n_val]
    # tr = idx[n_val:]
    #
    # Xtr, Ytr = X[tr], Y[tr]
    # Xva, Yva = X[va], Y[va]
    #
    # # Xtr, Xva, Ytr, Yva = train_test_split(X, Y, test_size=0.2, random_state=42)
    # # 2) train teacher
    # teacher = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.01, max_iter=5000, random_state=0)
    # teacher.fit(Xtr, Ytr)
    #
    # # teacher val rmse
    # pred_t = teacher.predict(Xva)
    # rmse_t = np.sqrt(np.mean((pred_t - Yva) ** 2))
    # print("Teacher val RMSE:", rmse_t)
    #
    # rmse = mean_squared_error(Yva, pred_t)
    # print("HGBR val RMSE:", rmse)







    # gbr = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.01, max_iter=5000, random_state=0)
    # gbr.fit(Xtr, Ytr)
    # pred = gbr.predict(Xva)
    # rmse = mean_squared_error(Yva, pred)
    # print("HGBR val RMSE:", rmse)

    # # 3) create soft targets for distillation
    # Ytr_soft = teacher.predict(Xtr)
    # Yva_soft = teacher.predict(Xva)
    #
    # sm = ScoreModel(d_in=X.shape[1], d_phi=128, hidden=(128, 128), dropout=0.1, device="cuda")
    #
    # # train to match teacher outputs (soft labels)
    # sm.pretrain(Xtr, Ytr_soft, epochs=500, lr=1e-3, weight_decay=1e-4, loss="huber", normalize_y=False, val_frac=0.0)
    #
    # # evaluate distillation quality on val
    # pred_student = sm.predict(Xva)
    # rmse_student_vs_teacher = np.sqrt(np.mean((pred_student - Yva_soft) ** 2))
    # rmse_student_vs_true = np.sqrt(np.mean((pred_student - Yva) ** 2))
    # print("Student vs teacher RMSE:", rmse_student_vs_teacher)
    # print("Student vs true RMSE:", rmse_student_vs_true)







    # # 3) create soft targets for distillation
    # Ytr_soft = teacher.predict(Xtr)
    # Yva_soft = teacher.predict(Xva)
    #
    # sm = ScoreModel(d_in=X.shape[1], d_phi=64, hidden=(128, 128), device="cuda")
    # sm.pretrain(X_train, Y_train, epochs=1000, lr=1e-3, batch_size=256, loss="mse")
    # # # sm.pretrain(X, Y, epochs=1000, lr=1e-5, val_frac=0.2, early_stop_patience=200, log_every=10)
    #
    sm = ScoreModel(d_in=XW.shape[1], d_phi=512, hidden=(512, 512), dropout=0.2, device="cuda")
    sm.pretrain(XW, Y_new, lr=1e-3, weight_decay=5e-4, loss="huber", val_frac=0.2)
    sm.save("sft_eval_scores/llm_mlp_test.pt")

if __name__ == "__main__":
    main()