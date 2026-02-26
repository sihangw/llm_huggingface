import numpy as np
import matplotlib.pyplot as plt

def pca_2d(Phi):
    Phi0 = Phi - Phi.mean(axis=0, keepdims=True)
    # SVD-based PCA
    U, S, Vt = np.linalg.svd(Phi0, full_matrices=False)
    Z = Phi0 @ Vt[:2].T
    return Z

def random_sampling_indices(N, T, seed=0):
    rng = np.random.default_rng(seed)
    return rng.choice(N, size=T, replace=False)

def dispersion_sampling_indices(Phi, T, seed=0):
    """
    Farthest-point / max-min distance sampling in feature space (Algorithm 5 idea).
    """
    rng = np.random.default_rng(seed)
    N = Phi.shape[0]
    chosen = []
    # start from a random point
    first = rng.integers(0, N)
    chosen.append(first)

    # keep track of min distance to chosen set for each point
    dmin = np.full(N, np.inf)
    for t in range(1, T):
        last = chosen[-1]
        # update distances using the last chosen point
        diff = Phi - Phi[last]
        dist2 = np.einsum("ij,ij->i", diff, diff)
        dmin = np.minimum(dmin, dist2)
        dmin[chosen] = -np.inf  # don't re-pick chosen points
        nxt = int(np.argmax(dmin))
        chosen.append(nxt)
    return np.array(chosen, dtype=int)

def plot_heatmap_compare(Phi, T=200, bins=80, seed=0, title_suffix=""):
    """
    Heatmap = 2D density of all pool points (after PCA projection),
    overlay = selected samples (random vs dispersion).
    """
    Z = pca_2d(Phi)
    N = Z.shape[0]

    idx_rand = random_sampling_indices(N, T, seed=seed)
    idx_disp = dispersion_sampling_indices(Phi, T, seed=seed)

    # 2D histogram (heatmap background)
    x, y = Z[:, 0], Z[:, 1]
    H, xedges, yedges = np.histogram2d(x, y, bins=bins)
    H = H.T  # for imshow orientation

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    for ax, idx, name in [
        (axes[0], idx_rand, "Random sampling"),
        (axes[1], idx_disp, "Selective sampling (dispersion / farthest-point)"),
    ]:
        im = ax.imshow(H, origin="lower", extent=extent, aspect="auto", cmap="magma", alpha=0.95)
        ax.scatter(Z[:, 0], Z[:, 1], s=3, alpha=0.08, color="white")  # faint pool points

        # overlay selected points; color by time (early->late)
        t = np.arange(len(idx))
        sc = ax.scatter(Z[idx, 0], Z[idx, 1], c=t, s=18, cmap="viridis", edgecolor="none")

        ax.set_title(f"{name} (T={T}){title_suffix}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(True, linestyle="--", alpha=0.25)

        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("Selection step (early → late)")

    cbar2 = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cbar2.set_label("Pool density per bin")
    plt.show()

# Example usage:

import numpy as np

X = np.load("data_10000_dt05_N20/run1_X.npy")
Phi = X  # if X is your instance feature vectors
# plot_heatmap_compare(Phi, T=300, bins=90, seed=0)


import numpy as np
import matplotlib.pyplot as plt

def pca_2d_standardized(Phi, eps=1e-12):
    Phi0 = Phi - Phi.mean(axis=0, keepdims=True)
    std = Phi0.std(axis=0, keepdims=True) + eps
    Phi0 = Phi0 / std
    U, S, Vt = np.linalg.svd(Phi0, full_matrices=False)
    return Phi0 @ Vt[:2].T

def clip_for_display(Z, q=0.01):
    lo = np.quantile(Z, q, axis=0)
    hi = np.quantile(Z, 1-q, axis=0)
    return np.clip(Z, lo, hi), lo, hi

Z = pca_2d_standardized(Phi)          # use your Phi/X here
Zc, lo, hi = clip_for_display(Z, q=0.01)

plt.figure(figsize=(7,5))
plt.scatter(Zc[:,0], Zc[:,1], s=3, alpha=0.2)
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.title("PCA (standardized) with 1% clipping for visualization")
plt.grid(True, linestyle="--", alpha=0.3)
plt.show()


def pca2_standardized(Phi, eps=1e-12):
    Phi0 = Phi - Phi.mean(axis=0, keepdims=True)
    Phi0 = Phi0 / (Phi0.std(axis=0, keepdims=True) + eps)
    U, S, Vt = np.linalg.svd(Phi0, full_matrices=False)
    return Phi0 @ Vt[:2].T

def quantile_limits(z, q=0.01):
    lo, hi = np.quantile(z, [q, 1-q])
    return lo, hi

def heatmap_with_overlay(Z, values, idx, title, bins=80):
    x, y = Z[:, 0], Z[:, 1]
    xlo, xhi = quantile_limits(x, 0.01)
    ylo, yhi = quantile_limits(y, 0.01)

    mask = (x>=xlo)&(x<=xhi)&(y>=ylo)&(y<=yhi)
    x, y, v = x[mask], y[mask], values[mask]

    # grid bins
    xb = np.linspace(xlo, xhi, bins+1)
    yb = np.linspace(ylo, yhi, bins+1)

    # average "information" in each bin
    sum_v, _, _ = np.histogram2d(x, y, bins=[xb, yb], weights=v)
    cnt,  _, _ = np.histogram2d(x, y, bins=[xb, yb])
    avg_v = sum_v / (cnt + 1e-12)          # avoid /0
    avg_v = avg_v.T                         # for imshow

    extent = [xlo, xhi, ylo, yhi]

    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    im = ax.imshow(avg_v, origin="lower", extent=extent, aspect="auto", cmap="magma")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Avg information score (per bin)")

    # overlay selected points colored by selection time
    t = np.arange(len(idx))
    ax.scatter(Z[idx,0], Z[idx,1], c=t, cmap="viridis", s=22, edgecolor="none",
               label="selected (colored by step)")

    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(loc="upper right", framealpha=0.95)
    plt.tight_layout()
    plt.show()

# ---------- selection baselines ----------
def random_idx(N, T, seed=0):
    rng = np.random.default_rng(seed)
    return rng.choice(N, size=T, replace=False)

def topk_info_idx(info, T):
    return np.argsort(-info)[:T]  # pick highest info points first

# ---------- if you DON'T have information gain yet: proxy ----------
def knn_rarity_proxy(Phi, k=20):
    """
    Larger = more isolated/rare (often good for exploration / info gain proxy).
    O(N^2) memory-safe version for moderate N; if N is huge, we can batch it.
    """
    N = Phi.shape[0]
    norms = np.sum(Phi**2, axis=1, keepdims=True)
    D2 = norms + norms.T - 2*Phi@Phi.T
    np.fill_diagonal(D2, np.inf)
    # kNN distance = mean of k smallest distances
    knn = np.partition(D2, kth=k-1, axis=1)[:, :k]
    return np.sqrt(np.mean(knn, axis=1))


# ====== Use like this ======
Z = pca2_standardized(Phi)
# info = ig if you have it else knn_rarity_proxy(Phi, k=20)
info = knn_rarity_proxy(Phi, k=20)
T = 300
idx_r = random_idx(len(Phi), T, seed=0)
idx_i = topk_info_idx(info, T)
heatmap_with_overlay(Z, info, idx_r, "Random sampling on info heatmap")
heatmap_with_overlay(Z, info, idx_i, "Information-gain sampling on info heatmap")









def knn_uncertainty_proxy(Phi, k=20):
    N = Phi.shape[0]
    norms = np.sum(Phi**2, axis=1, keepdims=True)
    D2 = norms + norms.T - 2*Phi@Phi.T
    np.fill_diagonal(D2, np.inf)
    knn = np.partition(D2, kth=k-1, axis=1)[:, :k]
    return np.sqrt(np.mean(knn, axis=1))  # larger = more isolated/uncertain

def random_idx(N, T, seed=0):
    rng = np.random.default_rng(seed)
    return rng.choice(N, size=T, replace=False)

def info_topk_idx(u, T):
    return np.argsort(-u)[:T]  # highest uncertainty first

def pca2_standardized(Phi, eps=1e-12):
    Phi0 = Phi - Phi.mean(axis=0, keepdims=True)
    Phi0 = Phi0 / (Phi0.std(axis=0, keepdims=True) + eps)
    U, S, Vt = np.linalg.svd(Phi0, full_matrices=False)
    return Phi0 @ Vt[:2].T

X = np.load("data_10000_dt05_N20/run1_X.npy")
Phi = X

# Choose uncertainty:
u = knn_uncertainty_proxy(Phi, k=20)   # or your real uncertainty vector

T = 300
idx_rand = random_idx(len(Phi), T, seed=0)
idx_info = info_topk_idx(u, T)



import matplotlib.pyplot as plt
import numpy as np

def cumulative_uncertainty(u, idx_order):
    return np.cumsum(u[idx_order])

cum_r = cumulative_uncertainty(u, idx_rand)
cum_i = cumulative_uncertainty(u, idx_info)

plt.figure(figsize=(7,4.5))
plt.plot(cum_r, label="Random sampling")
plt.plot(cum_i, label="Information sampling")
plt.xlabel("Sampling step t")
plt.ylabel("Cumulative uncertainty captured")
plt.title("Information sampling targets high-uncertainty regions earlier")
plt.grid(True, linestyle="--", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1) Load data
# =========================
X = np.load("data_10000_dt05_N20/run1_X.npy")  # shape (N, d)
Phi = X  # treat X as feature vectors

# =========================
# 2) PCA to 2D (PC1/PC2)
# =========================
def pca2_standardized(Phi, eps=1e-12):
    """
    Returns:
      Z: (N,2) 2D coordinates (PC1, PC2)
      evr2: explained variance ratio for PC1, PC2
    """
    Phi0 = Phi - Phi.mean(axis=0, keepdims=True)
    Phi0 = Phi0 / (Phi0.std(axis=0, keepdims=True) + eps)  # standardize

    U, S, Vt = np.linalg.svd(Phi0, full_matrices=False)
    Z = Phi0 @ Vt[:2].T

    var = (S**2) / (Phi0.shape[0] - 1)
    evr = var / var.sum()
    return Z, evr[:2]

Z, evr2 = pca2_standardized(Phi)

# =========================
# 3) Uncertainty score u (proxy)
# =========================
def knn_uncertainty_proxy(Phi, k=20):
    """
    Proxy uncertainty: larger = more isolated = 'rarer' region.
    NOTE: O(N^2) memory/time. For N=10k it's usually okay on a decent machine.
    If it's too slow, tell me and I'll give a batched version.
    """
    N = Phi.shape[0]
    norms = np.sum(Phi**2, axis=1, keepdims=True)
    D2 = norms + norms.T - 2 * (Phi @ Phi.T)  # squared distances
    np.fill_diagonal(D2, np.inf)
    knn = np.partition(D2, kth=k-1, axis=1)[:, :k]
    return np.sqrt(np.mean(knn, axis=1))

u = knn_uncertainty_proxy(Phi, k=20)  # (N,)

# =========================
# 4) Build heatmap M over PCA plane
# =========================
def build_uncertainty_heatmap(Z, u, bins=55, clip_q=0.01):
    """
    M: (bins, bins) heatmap values = median uncertainty per bin
    extent: [x_min, x_max, y_min, y_max] for imshow to place M correctly.

    This is exactly what "M and extent" are:
      - M is the 2D grid of values
      - extent tells matplotlib how to map that grid onto PC1/PC2 axes
    """
    x, y = Z[:, 0], Z[:, 1]

    # robust plot window to avoid outliers ruining the view
    xlo, xhi = np.quantile(x, [clip_q, 1 - clip_q])
    ylo, yhi = np.quantile(y, [clip_q, 1 - clip_q])

    # keep only points inside window when computing heatmap
    mask = (x >= xlo) & (x <= xhi) & (y >= ylo) & (y <= yhi)
    x, y, v = x[mask], y[mask], u[mask]

    # bin edges
    xb = np.linspace(xlo, xhi, bins + 1)
    yb = np.linspace(ylo, yhi, bins + 1)

    # assign each point to a bin
    xi = np.digitize(x, xb) - 1
    yi = np.digitize(y, yb) - 1

    # collect values per bin and take median (median is less noisy than mean)
    grid = [[[] for _ in range(bins)] for __ in range(bins)]
    for a, b, val in zip(xi, yi, v):
        if 0 <= a < bins and 0 <= b < bins:
            grid[b][a].append(val)

    M = np.full((bins, bins), np.nan)
    for r in range(bins):
        for c in range(bins):
            if grid[r][c]:
                M[r, c] = np.median(grid[r][c])

    extent = [xlo, xhi, ylo, yhi]
    return M, extent, (xlo, xhi, ylo, yhi)

M, extent, limits = build_uncertainty_heatmap(Z, u, bins=55, clip_q=0.01)

# =========================
# 5) Sampling strategies
# =========================
def random_idx(N, T, seed=0):
    rng = np.random.default_rng(seed)
    return rng.choice(N, size=T, replace=False)

def info_topk_idx(u, T):
    # choose the T most "uncertain" points first
    return np.argsort(-u)[:T]

T = 300
idx_rand = random_idx(len(Phi), T, seed=0)
idx_info = info_topk_idx(u, T)

# =========================
# 6) Plot: two panels, obvious markers
# =========================
def plot_two_panels(Z, M, extent, idx_a, idx_b, title_a, title_b,
                    K_show=60, number_points=False):
    # make heatmap less overpowering
    vmin = np.nanpercentile(M, 15)
    vmax = np.nanpercentile(M, 85)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), constrained_layout=True)

    for ax, idx, ttl in [(axes[0], idx_a, title_a), (axes[1], idx_b, title_b)]:
        im = ax.imshow(M, origin="lower", extent=extent, aspect="auto",
                       cmap="magma", vmin=vmin, vmax=vmax, alpha=0.88)

        idx_show = idx[:K_show]

        # VERY visible markers: big white star with thick black edge
        ax.scatter(Z[idx_show, 0], Z[idx_show, 1],
                   s=240, marker="*", facecolor="white",
                   edgecolor="black", linewidth=1.6, zorder=6)

        # extra outline ring (optional but helps on busy backgrounds)
        ax.scatter(Z[idx_show, 0], Z[idx_show, 1],
                   s=110, marker="o", facecolor="none",
                   edgecolor="cyan", linewidth=1.4, zorder=7)

        if number_points:
            for j, ii in enumerate(idx_show):
                ax.text(Z[ii, 0], Z[ii, 1], str(j+1),
                        ha="center", va="center",
                        fontsize=8, weight="bold",
                        color="black", zorder=8)

        ax.set_title(ttl)
        ax.set_xlabel(f"PC1 (EVR={evr2[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 (EVR={evr2[1]*100:.1f}%)")
        ax.grid(True, linestyle="--", alpha=0.25)

    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cb.set_label("Uncertainty heatmap (median per bin)")

    plt.show()


import numpy as np

def heatmap_guided_sampling(Z, u, bins=60, T=300, seed=0, clip_q=0.01, beta=1.0):
    """
    Picks T samples by:
      1) binning Z into a grid
      2) assigning each bin a weight ~ (median uncertainty)^beta * (count in bin)
      3) sampling bins by weight, then sampling a point from that bin (weighted by u)
    """
    rng = np.random.default_rng(seed)

    x, y = Z[:,0], Z[:,1]
    xlo, xhi = np.quantile(x, [clip_q, 1-clip_q])
    ylo, yhi = np.quantile(y, [clip_q, 1-clip_q])

    xb = np.linspace(xlo, xhi, bins+1)
    yb = np.linspace(ylo, yhi, bins+1)

    xi = np.digitize(x, xb) - 1
    yi = np.digitize(y, yb) - 1

    # collect indices per bin
    bin_lists = [[[] for _ in range(bins)] for __ in range(bins)]
    for i in range(len(Z)):
        a, b = xi[i], yi[i]
        if 0 <= a < bins and 0 <= b < bins:
            bin_lists[b][a].append(i)

    # compute bin weights
    bin_weights = []
    bin_coords = []
    for r in range(bins):
        for c in range(bins):
            idxs = bin_lists[r][c]
            if len(idxs) == 0:
                continue
            med_u = np.median(u[idxs])
            w = (med_u ** beta) * len(idxs)  # uncertainty * density
            bin_weights.append(w)
            bin_coords.append((r, c))

    bin_weights = np.array(bin_weights, dtype=float)
    bin_weights = bin_weights / bin_weights.sum()

    chosen = set()
    chosen_list = []

    # sample bins, then sample a point from the bin
    while len(chosen_list) < T:
        r, c = bin_coords[rng.choice(len(bin_coords), p=bin_weights)]
        idxs = bin_lists[r][c]

        # within bin, sample points proportional to uncertainty
        p = u[idxs].astype(float)
        p = p / (p.sum() + 1e-12)
        pick = int(rng.choice(idxs, p=p))

        if pick not in chosen:
            chosen.add(pick)
            chosen_list.append(pick)

    return np.array(chosen_list, dtype=int)


idx_heat = heatmap_guided_sampling(Z, u, bins=60, T=300, beta=1.5, seed=0)


import numpy as np
import matplotlib.pyplot as plt

# =========================
# Load data
# =========================
X = np.load("data_10000_dt05_N20/run1_X.npy")
Phi = X  # feature vectors (N,d)

# =========================
# PCA to 2D
# =========================
def pca2_standardized(Phi, eps=1e-12):
    Phi0 = Phi - Phi.mean(axis=0, keepdims=True)
    Phi0 = Phi0 / (Phi0.std(axis=0, keepdims=True) + eps)
    U, S, Vt = np.linalg.svd(Phi0, full_matrices=False)
    Z = Phi0 @ Vt[:2].T
    # explained variance ratio (approx)
    var = (S**2) / (Phi0.shape[0] - 1)
    evr = var / var.sum()
    return Z, evr[:2]

Z, evr2 = pca2_standardized(Phi)

# =========================
# Uncertainty u (proxy)
# Replace this with your true uncertainty if you have it
# =========================
def knn_uncertainty_proxy(Phi, k=20):
    N = Phi.shape[0]
    norms = np.sum(Phi**2, axis=1, keepdims=True)
    D2 = norms + norms.T - 2*(Phi @ Phi.T)
    np.fill_diagonal(D2, np.inf)
    knn = np.partition(D2, kth=k-1, axis=1)[:, :k]
    return np.sqrt(np.mean(knn, axis=1))

u = knn_uncertainty_proxy(Phi, k=20)

# =========================
# Build heatmap M and extent
# =========================
def build_heatmap(Z, u, bins=55, clip_q=0.01):
    x, y = Z[:,0], Z[:,1]
    xlo, xhi = np.quantile(x, [clip_q, 1-clip_q])
    ylo, yhi = np.quantile(y, [clip_q, 1-clip_q])

    mask = (x>=xlo)&(x<=xhi)&(y>=ylo)&(y<=yhi)
    x, y, v = x[mask], y[mask], u[mask]

    xb = np.linspace(xlo, xhi, bins+1)
    yb = np.linspace(ylo, yhi, bins+1)
    xi = np.digitize(x, xb) - 1
    yi = np.digitize(y, yb) - 1

    grid = [[[] for _ in range(bins)] for __ in range(bins)]
    for a, b, val in zip(xi, yi, v):
        if 0 <= a < bins and 0 <= b < bins:
            grid[b][a].append(val)

    M = np.full((bins, bins), np.nan)
    for r in range(bins):
        for c in range(bins):
            if grid[r][c]:
                M[r, c] = np.median(grid[r][c])  # median is stable

    extent = [xlo, xhi, ylo, yhi]
    return M, extent

M, extent = build_heatmap(Z, u, bins=55, clip_q=0.01)

# =========================
# Sampling: random vs heatmap-guided
# =========================
def random_idx(N, T, seed=0):
    rng = np.random.default_rng(seed)
    return rng.choice(N, size=T, replace=False)

def heatmap_guided_top_bins(Z, u, bins=55, T=300, seed=0, clip_q=0.01, top_bin_pct=20):
    rng = np.random.default_rng(seed)

    x, y = Z[:,0], Z[:,1]
    xlo, xhi = np.quantile(x, [clip_q, 1-clip_q])
    ylo, yhi = np.quantile(y, [clip_q, 1-clip_q])

    xb = np.linspace(xlo, xhi, bins+1)
    yb = np.linspace(ylo, yhi, bins+1)

    xi = np.digitize(x, xb) - 1
    yi = np.digitize(y, yb) - 1

    # collect indices per bin + compute median uncertainty per bin
    bin_lists = [[[] for _ in range(bins)] for __ in range(bins)]
    for i in range(len(Z)):
        a, b = xi[i], yi[i]
        if 0 <= a < bins and 0 <= b < bins:
            bin_lists[b][a].append(i)

    bin_med = []
    coords = []
    for r in range(bins):
        for c in range(bins):
            idxs = bin_lists[r][c]
            if idxs:
                coords.append((r,c))
                bin_med.append(np.median(u[idxs]))
    bin_med = np.array(bin_med)

    # keep only top bins
    thr = np.percentile(bin_med, 100 - top_bin_pct)
    top_coords = [coords[i] for i in np.where(bin_med >= thr)[0]]

    chosen = set()
    chosen_list = []
    while len(chosen_list) < T:
        r, c = top_coords[rng.integers(0, len(top_coords))]
        idxs = bin_lists[r][c]
        # sample within bin proportional to uncertainty
        p = u[idxs].astype(float); p /= (p.sum() + 1e-12)
        pick = int(rng.choice(idxs, p=p))
        if pick not in chosen:
            chosen.add(pick)
            chosen_list.append(pick)

    return np.array(chosen_list, dtype=int)

def heatmap_guided_sampling(Z, u, bins=55, T=300, seed=0, clip_q=0.01, beta=1.5):
    """
    Sample bins proportional to (median uncertainty)^beta * (bin count),
    then sample a point from the chosen bin proportional to u.
    """
    rng = np.random.default_rng(seed)
    x, y = Z[:,0], Z[:,1]

    xlo, xhi = np.quantile(x, [clip_q, 1-clip_q])
    ylo, yhi = np.quantile(y, [clip_q, 1-clip_q])

    xb = np.linspace(xlo, xhi, bins+1)
    yb = np.linspace(ylo, yhi, bins+1)

    xi = np.digitize(x, xb) - 1
    yi = np.digitize(y, yb) - 1

    bin_lists = [[[] for _ in range(bins)] for __ in range(bins)]
    for i in range(len(Z)):
        a, b = xi[i], yi[i]
        if 0 <= a < bins and 0 <= b < bins:
            bin_lists[b][a].append(i)

    bin_coords, wts = [], []
    for r in range(bins):
        for c in range(bins):
            idxs = bin_lists[r][c]
            if not idxs:
                continue
            med_u = np.median(u[idxs])
            w = (med_u ** beta) * len(idxs)
            bin_coords.append((r, c))
            wts.append(w)

    wts = np.array(wts, dtype=float)
    wts = wts / wts.sum()

    chosen = set()
    chosen_list = []
    while len(chosen_list) < T:
        r, c = bin_coords[rng.choice(len(bin_coords), p=wts)]
        idxs = bin_lists[r][c]
        p = u[idxs].astype(float)
        p = p / (p.sum() + 1e-12)
        pick = int(rng.choice(idxs, p=p))
        if pick not in chosen:
            chosen.add(pick)
            chosen_list.append(pick)

    return np.array(chosen_list, dtype=int)

T = 300
idx_rand = random_idx(len(Phi), T, seed=0)
idx_heat = heatmap_guided_sampling(Z, u, bins=55, T=T, seed=0, beta=1.5)
idx_hot = heatmap_guided_top_bins(Z, u, T=300, top_bin_pct=15, seed=0)

# =========================
# Plot side-by-side: SAME heatmap + big markers
# =========================
def plot_compare(Z, M, extent, idx_a, idx_b, title_a, title_b, K_show=80):
    cmap = plt.cm.magma_r.copy()
    cmap.set_bad(color="lightgray")  # empty bins = gray not white

    vmin = np.nanpercentile(M, 10)
    vmax = np.nanpercentile(M, 90)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), constrained_layout=True)
    for ax, idx, ttl in [(axes[0], idx_a, title_a), (axes[1], idx_b, title_b)]:
        im = ax.imshow(M, origin="lower", extent=extent, aspect="auto",
                       cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.90)

        # lock axes to heatmap extent (prevents outliers creating blank space)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

        idx_show = idx[:K_show]
        ax.scatter(Z[idx_show,0], Z[idx_show,1],
                   s=220, marker="*", facecolor="white",
                   edgecolor="black", linewidth=1.6, zorder=6)
        ax.scatter(Z[idx_show,0], Z[idx_show,1],
                   s=100, marker="o", facecolor="none",
                   edgecolor="cyan", linewidth=1.4, zorder=7)

        ax.set_title(ttl + f" (first {K_show})")
        ax.set_xlabel(f"PC1 (EVR={evr2[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 (EVR={evr2[1]*100:.1f}%)")
        ax.grid(True, linestyle="--", alpha=0.25)

    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cb.set_label("Uncertainty heatmap (median per bin)")
    plt.show()

plot_compare(
    Z, M, extent,
    idx_rand, idx_hot,
    "Random sampling",
    "Information sampling"
)

# =========================
# Quantitative: cumulative uncertainty captured
# =========================
def cumulative_uncertainty(u, idx):
    return np.cumsum(u[idx])

plt.figure(figsize=(7,4.5))
plt.plot(cumulative_uncertainty(u, idx_rand), label="Random")
plt.plot(cumulative_uncertainty(u, idx_heat), label="Information-guided")
plt.xlabel("Sampling step")
plt.ylabel("Cumulative uncertainty captured")
plt.title("Information-guided sampling targets uncertain regions earlier")
plt.grid(True, linestyle="--", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()