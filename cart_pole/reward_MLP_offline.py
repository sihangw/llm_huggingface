import torch
import torch.nn as nn
import numpy as np

class MLPRegressor:
    def __init__(self, hidden=(256,256), lr=1e-3, epochs=200, batch_size=256, device="cuda"):
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.model = None
        self.x_mean = None
        self.x_std = None
        self.x_mean_t = None
        self.x_std_t = None

    def _build(self, d):
        layers = []
        prev = d
        for h in self.hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, 1)]
        return nn.Sequential(*layers)

    def fit(self, X, Y):
        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.float32).reshape(-1)

        self.x_mean = X.mean(axis=0).astype(np.float32)
        self.x_std  = X.std(axis=0).astype(np.float32)
        self.x_std[self.x_std < 1e-12] = 1.0

        Xn = (X - self.x_mean) / self.x_std
        d = Xn.shape[1]

        self.model = self._build(d).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        X_t = torch.from_numpy(Xn).to(self.device)
        Y_t = torch.from_numpy(Y).to(self.device)

        n = len(Xn)
        self.model.train()
        for ep in range(self.epochs):
            perm = torch.randperm(n, device=self.device)
            for i in range(0, n, self.batch_size):
                idx = perm[i:i+self.batch_size]
                pred = self.model(X_t[idx]).squeeze(-1)
                loss = loss_fn(pred, Y_t[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()

        # cache torch stats for fast inference
        self.x_mean_t = torch.tensor(self.x_mean, device=self.device)
        self.x_std_t  = torch.tensor(self.x_std,  device=self.device)

        return self

    def freeze(self):
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        return self

    @torch.no_grad()
    def predict_torch(self, X_t: torch.Tensor) -> torch.Tensor:
        """
        X_t: torch tensor [B, d] on CPU or GPU
        returns: torch tensor [B] on same device as X_t
        """
        if X_t.dtype != torch.float32:
            X_t = X_t.float()

        dev = X_t.device
        self.model = self.model.to(dev)

        # ensure cached stats exist on correct device
        if self.x_mean_t is None or self.x_mean_t.device != dev:
            self.x_mean_t = torch.tensor(self.x_mean, device=dev)
            self.x_std_t  = torch.tensor(self.x_std,  device=dev)

        Xn = (X_t - self.x_mean_t) / self.x_std_t
        return self.model(Xn).squeeze(-1)

    @torch.no_grad()
    def predict(self, X):
        """
        Accepts numpy or torch. Returns numpy (for convenience outside RL).
        """
        if torch.is_tensor(X):
            return self.predict_torch(X).cpu().numpy()

        X = np.asarray(X, dtype=np.float32)
        Xn = (X - self.x_mean) / self.x_std
        X_t = torch.from_numpy(Xn).to(self.device)
        return self.model(X_t).squeeze(-1).cpu().numpy()
