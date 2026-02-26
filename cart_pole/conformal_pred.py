import numpy as np
from reward_MLP_offline import MLPRegressor
from sft_LLM_MLP import ScoreModel
from sft_conformal import SplitConformalRegressor

X = np.load("data_10000_dt05_N20/run1_X.npy")
Y = np.load("sft_eval_scores/Y_llm.npy")
Y_gt = np.load("data_10000_dt05_N20/run1_Y.npy")

# print(Y[:10])
# print(Y_gt[:10])

n_train, n_cal, n_test = 8500, 1000, 500
n_total = n_train + n_cal + n_test
X_train, Y_train, Y_gt_train = X[:n_train], Y[:n_train], Y_gt[:n_train]
X_cal, Y_cal, Y_gt_cal = X[n_train:n_train + n_cal], Y[n_train:n_train + n_cal], Y_gt[n_train:n_train + n_cal]
X_test, Y_test, Y_gt_test = X[n_train + n_cal:n_total], Y[n_train + n_cal:n_total], Y_gt[n_train + n_cal:n_total]


print(Y_cal[:10])
print(Y_gt_cal[:10])

res = np.abs(Y_cal - Y_gt_cal)
n = res.shape[0]
alpha = 0.9
# conformal quantile index: ceil((n+1)*(1-alpha)) / n
k = int(np.ceil((n + 1) * (1.0 - alpha)))
k = min(max(k, 1), n)  # clamp
qhat = np.sort(res)[k - 1]

# 5) prediction intervals
lo = Y_test - qhat
hi = Y_test + qhat
print(lo[23:35])
print(hi[23:35])
print(Y_gt_test[23:35])

def summarize_intervals(lo, hi, y_true, alpha):
    covered = (y_true >= lo) & (y_true <= hi)
    coverage = covered.mean()
    widths = hi - lo
    print(f"Target coverage (1-alpha): {alpha:.3f}")
    print(f"Empirical coverage:        {coverage:.3f}")
    print(f"Mean width:               {widths.mean():.4f}")
    print(f"Median width:             {np.median(widths):.4f}")
    print(f"90% width quantile:       {np.quantile(widths, 0.9):.4f}")
    return covered, widths

import matplotlib.pyplot as plt

covered, widths = summarize_intervals(lo, hi, Y_gt_test, alpha)

mid = Y_test
half = (hi - lo) / 2.0

plt.figure(figsize=(7, 6))
sc = plt.scatter(mid, Y_gt_test, s=8, alpha=0.25, c=half, rasterized=True)
mn = min(mid.min(), Y_gt_test.min())
mx = max(mid.max(), Y_gt_test.max())
plt.plot([mn, mx], [mn, mx], linewidth=2)  # y=x reference
plt.xlabel("Point prediction (Y_test)")
plt.ylabel("Ground truth (Y_gt_test)")
cbar = plt.colorbar(sc)
cbar.set_label("Interval half-width (q̂)")
plt.title(f"Pred vs GT (n={len(Y_gt_test)} test points), color = uncertainty")
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
m = min(500, len(Y_gt_test))  # show 500 intervals
idx = rng.choice(len(Y_gt_test), size=m, replace=False)

order = np.argsort(Y_test[idx])  # sort for a clean visual
idx = idx[order]

plt.figure(figsize=(8, 6))
ypos = np.arange(m)

plt.hlines(ypos, lo[idx], hi[idx], linewidth=1.2, alpha=0.8)
plt.plot(Y_gt_test[idx], ypos, '.', markersize=6, alpha=0.9, label="GT")
plt.plot(Y_test[idx], ypos, 'x', markersize=4, alpha=0.7, label="Pred")

plt.xlabel("Score/value")
plt.ylabel("Example (sorted by prediction)")
plt.title(f"Conformal intervals on {m} random test samples")
plt.legend()
plt.tight_layout()
plt.show()



# import numpy as np
# import matplotlib.pyplot as plt
#
# resid = Y_gt_test - Y_test
#
# fig, ax = plt.subplots(figsize=(7.6, 6.2))
#
# hb = ax.hexbin(
#     Y_test, resid,
#     gridsize=65, mincnt=1,
#     cmap="viridis"   # colorful density
# )
#
# # Shaded band for ±qhat
# ax.axhspan(-qhat, +qhat, alpha=0.15, label=r"Conformal band $\pm \hat{q}$")
#
# # Reference lines
# ax.axhline(0, color="black", linewidth=2, label="Zero residual")
# ax.axhline(+qhat, color="crimson", linestyle="--", linewidth=2.5, label=r"$+\hat{q}$")
# ax.axhline(-qhat, color="crimson", linestyle="--", linewidth=2.5, label=r"$-\hat{q}$")
#
# ax.set_xlabel("Prediction (Y_test)")
# ax.set_ylabel("Residual (Y_gt_test - Y_test)")
# ax.set_title(rf"Residual density (hexbin), band = $\pm \hat{{q}}$  ($\hat{{q}}$={qhat:.4f})")
#
# cbar = fig.colorbar(hb, ax=ax)
# cbar.set_label("Count per hex")
#
# ax.grid(True, linestyle="--", alpha=0.35)
# ax.legend(loc="upper right", framealpha=0.95)
# plt.tight_layout()
# plt.show()


import matplotlib.pyplot as plt

resid = Y_gt_test - Y_test

plt.figure(figsize=(7, 6))
hb = plt.hexbin(Y_test, resid, gridsize=60, mincnt=1)
plt.axhline(0, linewidth=2)
plt.axhline(+qhat, linestyle='--', linewidth=2)
plt.axhline(-qhat, linestyle='--', linewidth=2)
plt.xlabel("Prediction (Y_test)")
plt.ylabel("Residual (Y_gt_test - Y_test)")
plt.title(f"Residual density (hexbin), dashed = ± q̂ (q̂={qhat:.4f})")
plt.colorbar(hb, label="count")
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import numpy as np

alphas = np.linspace(0.05, 0.95, 19)  # nominal coverage levels
n = len(Y_cal)

res_cal = np.abs(Y_cal - Y_gt_cal)
res_cal_sorted = np.sort(res_cal)

emp_cov = []
for a in alphas:
    k = int(np.ceil((n + 1) * (1.0 - a)))
    k = min(max(k, 1), n)
    q = res_cal_sorted[k - 1]
    lo_a = Y_test - q
    hi_a = Y_test + q
    cov = ((Y_gt_test >= lo_a) & (Y_gt_test <= hi_a)).mean()
    emp_cov.append(cov)

plt.figure(figsize=(6, 6))
plt.plot(1 - alphas, emp_cov, marker='o')
plt.plot([0, 1], [0, 1], linewidth=2)  # ideal
plt.xlabel("Nominal coverage (1 - alpha)")
plt.ylabel("Empirical coverage (test)")
plt.title("Conformal calibration curve of LLM predictor")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

alphas = np.linspace(0.05, 0.95, 19)      # miscoverage levels
n = len(Y_cal)

res_cal = np.abs(Y_cal - Y_gt_cal)
res_cal_sorted = np.sort(res_cal)

emp_cov = []
for a in alphas:
    k = int(np.ceil((n + 1) * (1.0 - a)))
    k = min(max(k, 1), n)
    q = res_cal_sorted[k - 1]
    lo_a = Y_test - q
    hi_a = Y_test + q
    cov = ((Y_gt_test >= lo_a) & (Y_gt_test <= hi_a)).mean()
    emp_cov.append(cov)

nom_cov = 1 - alphas  # nominal coverage on x-axis

fig, ax = plt.subplots(figsize=(6.4, 6.2))

# Ideal diagonal
ax.plot([0, 1], [0, 1], linestyle="--", linewidth=2.5, color="gray", label="Ideal (y=x)")

# Light shading around ideal (optional; looks nice in talks)
ax.fill_between([0, 1], [0, 1], [0, 1], alpha=0.0)  # no-op placeholder; keep simple

# Empirical curve
ax.plot(
    nom_cov, emp_cov,
    marker="o", linewidth=2.5,
    color="tab:blue",
    label="Empirical (test)"
)

# Highlight your main operating point (e.g., 90% or 95%)
target = 0.95
idx = np.argmin(np.abs(nom_cov - target))
# ax.scatter([nom_cov[idx]], [emp_cov[idx]], s=120, color="crimson", zorder=5,
#            label=f"Highlight @ nominal={nom_cov[idx]:.2f}")

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel("Nominal coverage (1 - alpha)")
ax.set_ylabel("Empirical coverage (test)")
ax.set_title("Conformal calibration curve of LLM predictor")
ax.grid(True, linestyle="--", alpha=0.35)
ax.legend(loc="lower right", framealpha=0.95)

plt.tight_layout()
plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
#
# # --- pick which split you want to visualize ---
# abs_err = np.abs(Y_test - Y_gt_test)     # histogram of test errors
# abs_err_cal = np.abs(Y_cal - Y_gt_cal)   # calibration errors used for qhat
#
# # --- define nominal coverage (like the figure: 95%) ---
# coverage_nominal = 0.90
# alpha = 1.0 - coverage_nominal  # miscoverage
#
# # --- conformal threshold from CAL (split conformal) ---
# n = len(abs_err_cal)
# k = int(np.ceil((n + 1) * (1.0 - alpha)))
# k = min(max(k, 1), n)
# qhat = np.sort(abs_err_cal)[k - 1]  # conformal radius (empirical threshold)
#
# # --- "desired threshold" (optional) ---
# # If you have some target tolerance from domain knowledge, set it here.
# # Otherwise, a common choice is the same nominal quantile computed on TEST just for comparison (NOT for conformal).
# desired_thr = np.quantile(abs_err, coverage_nominal)
#
# # --- plot like the example ---
# plt.figure(figsize=(7.2, 4.6))
# plt.hist(abs_err, bins=30, edgecolor="k", linewidth=1.0, alpha=0.35, label="Error Distribution")
#
# plt.axvline(qhat, linestyle="--", linewidth=2.5, label=f"Empirical Threshold ({coverage_nominal*100:.1f}%)")
# plt.axvline(desired_thr, linestyle="-.", linewidth=2.5, label=f"Desired Threshold ({coverage_nominal*100:.1f}%)")
#
# plt.grid(True, which="both", linestyle="--", alpha=0.6)
# plt.xlabel(r"$|Y_{\mathrm{pred}} - Y_{\mathrm{gt}}|$")
# plt.ylabel("Frequency")
# plt.title(f"Error Distribution with Thresholds (n={len(abs_err)})")
# plt.legend(loc="upper right", framealpha=0.95)
# plt.tight_layout()
# plt.show()


import numpy as np
import matplotlib.pyplot as plt

# --- data ---
abs_err_test = np.abs(Y_test - Y_gt_test)
abs_err_cal  = np.abs(Y_cal  - Y_gt_cal)

# --- nominal coverage like your example (e.g., 95%) ---
coverage_nominal = 0.90
alpha = 1.0 - coverage_nominal

# --- split conformal threshold from CAL ---
n = len(abs_err_cal)
k = int(np.ceil((n + 1) * (1.0 - alpha)))
k = min(max(k, 1), n)
qhat = np.sort(abs_err_cal)[k - 1]

# --- "desired threshold" (set this to your domain tolerance if you have one) ---
# Example option A (domain tolerance):
# desired_thr = 0.20
# Example option B (for comparison only): same nominal quantile on TEST:
desired_thr = np.quantile(abs_err_test, coverage_nominal)

# --- plot ---
fig, ax = plt.subplots(figsize=(7.4, 4.8))

# Histogram with colored bars
counts, bins, patches = ax.hist(
    abs_err_test, bins=35, alpha=0.85, edgecolor="white", linewidth=0.8,
    label="Error Distribution"
)

# Gradient-like coloring across bins
for i, p in enumerate(patches):
    p.set_facecolor(plt.cm.viridis(i / max(1, len(patches) - 1)))

# Vertical lines (match your example: red dashed, green dash-dot)
ax.axvline(qhat, color="crimson", linestyle="--", linewidth=2.8,
           label=f"Empirical Threshold ({coverage_nominal*100:.1f}%)")
ax.axvline(desired_thr, color="forestgreen", linestyle="-.", linewidth=2.8,
           label=f"Desired Threshold ({coverage_nominal*100:.1f}%)")

# Shaded regions to emphasize "good" vs "tail"
xmax = max(abs_err_test.max(), qhat, desired_thr) * 1.05
ax.set_xlim(0, xmax)

# Shade left of qhat (covered-ish) and right tail (misses-ish)
ax.axvspan(0, qhat, color="forestgreen", alpha=0.08, label="Within q̂ (visual)")
ax.axvspan(qhat, xmax, color="crimson", alpha=0.06, label="Beyond q̂ (visual)")

# Labels / style
ax.grid(True, which="both", linestyle="--", alpha=0.45)
ax.set_xlabel(r"$|Y_{\mathrm{pred}} - Y_{\mathrm{gt}}|$")
ax.set_ylabel("Frequency")
ax.set_title(f"Test Error of LLM predictor with Thresholds (n={len(abs_err_test)})")

# Cleaner legend (avoid duplicates from shading)
handles, labels = ax.get_legend_handles_labels()
seen = set()
h2, l2 = [], []
for h, l in zip(handles, labels):
    if l not in seen:
        h2.append(h); l2.append(l); seen.add(l)
ax.legend(h2, l2, loc="upper right", framealpha=0.95)

plt.tight_layout()
plt.show()



# # Load some X (must have same feature dimension as training)
# # Create the model with the SAME architecture as training
# sm = ScoreModel(d_in=X.shape[1], d_phi=64, hidden=(128, 128), device="cuda")
# sm.load("sft_eval_scores/llm_mlp.pt")
# # Get predicted reward and clip to [0,1]
# r = sm.predict(X[23:33])



# print(r)
# print("true:", Y[23:33])

# # rm is your trained OptionBRewardModel (has predict(X)->(N,))
# cp = SplitConformalRegressor(predictor=sm, alpha=0.1, clip=(0.0, 1.0))
# cp.fit_calibration(X_cal, Y_cal)
#
# yhat, lo, hi = cp.predict_interval(X_test)
# print("pred:", yhat[:5])
# print("true:", Y_test[:5])
# print("lo:", lo[:5])
# print("hi:", hi[:5])
#
# print("empirical coverage:", cp.empirical_coverage(X_test, Y_test))