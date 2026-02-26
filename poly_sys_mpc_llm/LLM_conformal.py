import numpy as np
from reward_MLP_offline import MLPRegressor
from sft_LLM_MLP import ScoreModel


# Y_pred = np.load("sft_cal_scores/Y_llm.npy")
# Y_gt = np.load("sft_cal_scores/Y_oracle.npy")


Y_pred = np.load("sft_cal_scores/Y_llm.npy").reshape(-1)
Y_gt   = np.load("sft_cal_scores/Y_oracle.npy").reshape(-1)

assert Y_pred.shape == Y_gt.shape, (Y_pred.shape, Y_gt.shape)

def conformal_split_intervals(yhat, y, alpha=0.1, cal_size=1000, seed=42, clip=(0.0, 1.0)):
    """
    Split conformal using absolute residuals.
    Returns: (q, yhat_test, lo, hi, coverage, idx_cal, idx_test)
    """
    n = len(y)
    assert cal_size < n, f"cal_size={cal_size} must be < n={n}"

    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    idx_cal  = idx[:cal_size]
    idx_test = idx[cal_size:]

    yhat_cal = yhat[idx_cal]
    y_cal    = y[idx_cal]
    res = np.abs(y_cal - yhat_cal)

    # Conformal quantile (finite-sample, split conformal)
    # q = k-th smallest residual where k = ceil((n_cal+1)*(1-alpha))
    n_cal = len(res)
    k = int(np.ceil((n_cal + 1) * (1 - alpha)))
    k = min(max(k, 1), n_cal)
    q = np.partition(res, k - 1)[k - 1]

    # Test intervals
    yhat_test = yhat[idx_test]
    lo = yhat_test - q
    hi = yhat_test + q

    if clip is not None:
        lo = np.clip(lo, clip[0], clip[1])
        hi = np.clip(hi, clip[0], clip[1])
        yhat_test = np.clip(yhat_test, clip[0], clip[1])

    y_test = y[idx_test]
    coverage = float(np.mean((y_test >= lo) & (y_test <= hi)))

    return q, yhat_test, lo, hi, coverage, idx_cal, idx_test

# ---- run it ----
alpha = 0.3 # 60% intervals
cal_size = 800      # you can change this
q, yhat_te, lo, hi, cov, idx_cal, idx_te = conformal_split_intervals(
    Y_pred, Y_gt, alpha=alpha, cal_size=cal_size, seed=42, clip=(0.0, 1.0)
)

print(f"alpha={alpha}  cal_size={cal_size}  test_size={len(idx_te)}")
print("conformal q:", q)
print("empirical test coverage:", cov)

# Example: show first 10 test intervals
for i in range(20):
    j = idx_te[i]
    print(f"idx {j}: pred={Y_pred[j]:.4f}, gt={Y_gt[j]:.4f}, interval=[{lo[i]:.4f}, {hi[i]:.4f}]")