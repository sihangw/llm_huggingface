# import numpy as np
#
# X = np.load("grpo_cal_scores_500/X.npy")
# Y_oracle = np.load("grpo_cal_scores_500/Y_oracle.npy")
# Y_llm_grpo = np.load("grpo_cal_scores_500/Y_llm.npy")
# Y_llm_sft = np.load("grpo_sft_cal_scores_500/Y_llm.npy")
# print(Y_llm_sft[:10])
# print(Y_llm_grpo[:10])

import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
from scipy import stats

# Load data
Y_oracle = np.load("grpo_cal_scores_500/Y_oracle.npy")
Y_llm_grpo = np.load("grpo_cal_scores_500/Y_llm.npy")
Y_llm_sft = np.load("grpo_sft_cal_scores_500/Y_llm.npy")

print("="*60)
print("COMPARISON: Oracle vs GRPO vs SFT")
print("="*60)

# ===== 1. Basic Statistics =====
print("\n1. BASIC STATISTICS")
print("-"*60)

methods = {
    "Oracle": Y_oracle,
    "GRPO": Y_llm_grpo,
    "SFT": Y_llm_sft,
}

for name, Y in methods.items():
    print(f"\n{name}:")
    print(f"  Mean:   {np.mean(Y):.4f}")
    print(f"  Median: {np.median(Y):.4f}")
    print(f"  Std:    {np.std(Y):.4f}")
    print(f"  Min:    {np.min(Y):.4f}")
    print(f"  Max:    {np.max(Y):.4f}")
    print(f"  Q25:    {np.percentile(Y, 25):.4f}")
    print(f"  Q75:    {np.percentile(Y, 75):.4f}")

# ===== 2. Performance Gap (vs Oracle) =====
print("\n2. PERFORMANCE GAP (vs Oracle)")
print("-"*60)

gap_grpo = Y_oracle - Y_llm_grpo
gap_sft = Y_oracle - Y_llm_sft

print(f"\nGRPO Gap:")
print(f"  Mean gap:   {np.mean(gap_grpo):.4f}")
print(f"  Median gap: {np.median(gap_grpo):.4f}")
print(f"  % of oracle: {np.mean(Y_llm_grpo) / np.mean(Y_oracle) * 100:.1f}%")

print(f"\nSFT Gap:")
print(f"  Mean gap:   {np.mean(gap_sft):.4f}")
print(f"  Median gap: {np.median(gap_sft):.4f}")
print(f"  % of oracle: {np.mean(Y_llm_sft) / np.mean(Y_oracle) * 100:.1f}%")

# ===== 3. Win Rate =====
print("\n3. WIN RATE")
print("-"*60)

grpo_wins = np.sum(Y_llm_grpo > Y_llm_sft)
sft_wins = np.sum(Y_llm_sft > Y_llm_grpo)
ties = np.sum(Y_llm_grpo == Y_llm_sft)

print(f"GRPO wins:  {grpo_wins}/{len(Y_oracle)} ({grpo_wins/len(Y_oracle)*100:.1f}%)")
print(f"SFT wins:   {sft_wins}/{len(Y_oracle)} ({sft_wins/len(Y_oracle)*100:.1f}%)")
print(f"Ties:       {ties}/{len(Y_oracle)} ({ties/len(Y_oracle)*100:.1f}%)")

# ===== 4. Statistical Tests =====
print("\n4. STATISTICAL TESTS")
print("-"*60)

# Paired t-test (GRPO vs SFT)
t_stat, p_value = stats.ttest_rel(Y_llm_grpo, Y_llm_sft)
print(f"\nPaired t-test (GRPO vs SFT):")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value:     {p_value:.4e}")
if p_value < 0.05:
    winner = "GRPO" if np.mean(Y_llm_grpo) > np.mean(Y_llm_sft) else "SFT"
    print(f"  Result: {winner} is significantly better (p < 0.05)")
else:
    print(f"  Result: No significant difference")

# Wilcoxon signed-rank test (non-parametric alternative)
w_stat, p_value_w = stats.wilcoxon(Y_llm_grpo, Y_llm_sft)
print(f"\nWilcoxon test (GRPO vs SFT):")
print(f"  W-statistic: {w_stat:.4f}")
print(f"  p-value:     {p_value_w:.4e}")

# ===== VISUALIZATIONS =====
print("\n" + "="*60)
print("GENERATING FIGURES...")
print("="*60)

fig = plt.figure(figsize=(16, 12))

# ===== Plot 1: Distribution Comparison (Box Plot) =====
ax1 = plt.subplot(3, 3, 1)
data_box = [Y_oracle, Y_llm_grpo, Y_llm_sft]
bp = ax1.boxplot(data_box, labels=['Oracle', 'GRPO', 'SFT'], patch_artist=True)
colors = ['gold', 'lightblue', 'lightgreen']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
ax1.set_ylabel('Score (Y)')
ax1.set_title('Score Distribution Comparison')
ax1.grid(True, alpha=0.3)

# ===== Plot 2: Violin Plot =====
ax2 = plt.subplot(3, 3, 2)
parts = ax2.violinplot(data_box, positions=[1, 2, 3], showmeans=True, showmedians=True)
ax2.set_xticks([1, 2, 3])
ax2.set_xticklabels(['Oracle', 'GRPO', 'SFT'])
ax2.set_ylabel('Score (Y)')
ax2.set_title('Score Distribution (Violin)')
ax2.grid(True, alpha=0.3)

# ===== Plot 3: Histogram Overlay =====
ax3 = plt.subplot(3, 3, 3)
ax3.hist(Y_oracle, bins=30, alpha=0.5, label='Oracle', color='gold', density=True)
ax3.hist(Y_llm_grpo, bins=30, alpha=0.5, label='GRPO', color='blue', density=True)
ax3.hist(Y_llm_sft, bins=30, alpha=0.5, label='SFT', color='green', density=True)
ax3.set_xlabel('Score (Y)')
ax3.set_ylabel('Density')
ax3.set_title('Score Distribution Overlay')
ax3.legend()
ax3.grid(True, alpha=0.3)

# ===== Plot 4: GRPO vs Oracle Scatter =====
ax4 = plt.subplot(3, 3, 4)
ax4.scatter(Y_oracle, Y_llm_grpo, alpha=0.5, s=20)
lim = [min(Y_oracle.min(), Y_llm_grpo.min()),
       max(Y_oracle.max(), Y_llm_grpo.max())]
ax4.plot(lim, lim, 'r--', label='y=x (perfect match)')
ax4.set_xlabel('Oracle Score')
ax4.set_ylabel('GRPO Score')
ax4.set_title(f'GRPO vs Oracle (corr={np.corrcoef(Y_oracle, Y_llm_grpo)[0,1]:.3f})')
ax4.legend()
ax4.grid(True, alpha=0.3)

# ===== Plot 5: SFT vs Oracle Scatter =====
ax5 = plt.subplot(3, 3, 5)
ax5.scatter(Y_oracle, Y_llm_sft, alpha=0.5, s=20, color='green')
ax5.plot(lim, lim, 'r--', label='y=x (perfect match)')
ax5.set_xlabel('Oracle Score')
ax5.set_ylabel('SFT Score')
ax5.set_title(f'SFT vs Oracle (corr={np.corrcoef(Y_oracle, Y_llm_sft)[0,1]:.3f})')
ax5.legend()
ax5.grid(True, alpha=0.3)

# ===== Plot 6: GRPO vs SFT Scatter =====
ax6 = plt.subplot(3, 3, 6)
ax6.scatter(Y_llm_sft, Y_llm_grpo, alpha=0.5, s=20, color='purple')
lim2 = [min(Y_llm_sft.min(), Y_llm_grpo.min()),
        max(Y_llm_sft.max(), Y_llm_grpo.max())]
ax6.plot(lim2, lim2, 'r--', label='y=x (equal)')
ax6.set_xlabel('SFT Score')
ax6.set_ylabel('GRPO Score')
ax6.set_title(f'GRPO vs SFT (corr={np.corrcoef(Y_llm_sft, Y_llm_grpo)[0,1]:.3f})')
ax6.legend()
ax6.grid(True, alpha=0.3)

# ===== Plot 7: Gap Distribution =====
ax7 = plt.subplot(3, 3, 7)
ax7.hist(gap_grpo, bins=30, alpha=0.6, label='GRPO gap', color='blue')
ax7.hist(gap_sft, bins=30, alpha=0.6, label='SFT gap', color='green')
ax7.axvline(0, color='red', linestyle='--', label='Zero gap')
ax7.set_xlabel('Gap (Oracle - Method)')
ax7.set_ylabel('Count')
ax7.set_title('Performance Gap Distribution')
ax7.legend()
ax7.grid(True, alpha=0.3)

# ===== Plot 8: Cumulative Distribution =====
ax8 = plt.subplot(3, 3, 8)
x_oracle = np.sort(Y_oracle)
x_grpo = np.sort(Y_llm_grpo)
x_sft = np.sort(Y_llm_sft)
y = np.arange(1, len(Y_oracle)+1) / len(Y_oracle)
ax8.plot(x_oracle, y, label='Oracle', linewidth=2, color='gold')
ax8.plot(x_grpo, y, label='GRPO', linewidth=2, color='blue')
ax8.plot(x_sft, y, label='SFT', linewidth=2, color='green')
ax8.set_xlabel('Score (Y)')
ax8.set_ylabel('Cumulative Probability')
ax8.set_title('Cumulative Distribution Function')
ax8.legend()
ax8.grid(True, alpha=0.3)

# ===== Plot 9: Bar Chart Summary =====
ax9 = plt.subplot(3, 3, 9)
metrics = ['Mean', 'Median', 'Max']
oracle_vals = [np.mean(Y_oracle), np.median(Y_oracle), np.max(Y_oracle)]
grpo_vals = [np.mean(Y_llm_grpo), np.median(Y_llm_grpo), np.max(Y_llm_grpo)]
sft_vals = [np.mean(Y_llm_sft), np.median(Y_llm_sft), np.max(Y_llm_sft)]

x = np.arange(len(metrics))
width = 0.25
ax9.bar(x - width, oracle_vals, width, label='Oracle', color='gold')
ax9.bar(x, grpo_vals, width, label='GRPO', color='blue')
ax9.bar(x + width, sft_vals, width, label='SFT', color='green')
ax9.set_xticks(x)
ax9.set_xticklabels(metrics)
ax9.set_ylabel('Score')
ax9.set_title('Summary Statistics')
ax9.legend()
ax9.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('comparison_results.png', dpi=300, bbox_inches='tight')
print("Saved: comparison_results.png")
plt.show()

# ===== SUMMARY =====
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

grpo_mean = np.mean(Y_llm_grpo)
sft_mean = np.mean(Y_llm_sft)
oracle_mean = np.mean(Y_oracle)

print(f"\nOverall Performance:")
print(f"  Oracle: {oracle_mean:.4f} (baseline)")
print(f"  GRPO:   {grpo_mean:.4f} ({grpo_mean/oracle_mean*100:.1f}% of oracle)")
print(f"  SFT:    {sft_mean:.4f} ({sft_mean/oracle_mean*100:.1f}% of oracle)")

if grpo_mean > sft_mean:
    improvement = (grpo_mean - sft_mean) / sft_mean * 100
    print(f"\n✅ GRPO is better by {improvement:.1f}%")
else:
    improvement = (sft_mean - grpo_mean) / grpo_mean * 100
    print(f"\n✅ SFT is better by {improvement:.1f}%")

print("\n" + "="*60)