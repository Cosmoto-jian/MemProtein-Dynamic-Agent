# -*- coding: utf-8 -*-
"""Explore relationships between ξ̂ and geometry across OK proteins."""

import glob, os, re, numpy as np
import matplotlib.pyplot as plt

from scipy import stats

RESULTS = "data/results"
FIGURES = "data/figures"

# ---- load all OK proteins ----
damage = {}
with open("data/batch.log") as fh:
    for line in fh:
        m = re.match(r"\[(\w+)\s+F=[\d.]+\].*ratio=[\d.]+\s+(\S+)", line)
        if m:
            damage[m.group(1)] = m.group(2)

records = []
for npz_path in sorted(glob.glob(f"{RESULTS}/scale_*_F0.01.npz")):
    pid = os.path.basename(npz_path).replace("scale_", "").replace("_F0.01.npz", "")
    if damage.get(pid) != "ok":
        continue
    d = np.load(npz_path)
    records.append({
        "pid": pid,
        "L_par": float(d["L_parallel"]),
        "L_perp": float(d["L_perp"]),
        "eta": float(d["eta_shape"]),       # L_par / L_perp
        "xi_raw_par": d["xi_parallel"][-1],   # raw correlation length (nm)
        "xi_raw_perp": d["xi_perp"][-1],
        "xi_hat_par": d["xi_hat_parallel"][-1],  # normalized ξ̂
        "xi_hat_perp": d["xi_hat_perp"][-1],
        "chi": d["chi_excess"][-1],
    })

# filter: remove very small proteins with L_par < 0.9 (xi likely saturated)
clean = [r for r in records if r["L_par"] >= 0.9]
print(f"OK proteins: {len(records)} (all), {len(clean)} (L_par >= 0.9 nm, reliable ξ̂)")

# ---- correlation analysis ----
pairs = [
    ("xi_hat_par", "ξ̂_∥"),
    ("xi_hat_perp", "ξ̂_⊥"),
    ("chi", "χ_excess"),
    ("xi_raw_par", "ξ_raw_∥ (nm)"),
    ("xi_raw_perp", "ξ_raw_⊥ (nm)"),
]
geo_pairs = [
    ("eta", "η = L_∥ / L_⊥"),
    ("L_par", "L_∥ (nm)"),
    ("L_perp", "L_⊥ (nm)"),
]

fig, axes = plt.subplots(len(pairs), len(geo_pairs), figsize=(16, 20))
for i, (yk, yl) in enumerate(pairs):
    for j, (xk, xl) in enumerate(geo_pairs):
        ax = axes[i, j]
        x_vals = np.array([r[xk] for r in clean])
        y_vals = np.array([r[yk] for r in clean])

        # Pearson r
        r_val, p_val = stats.pearsonr(x_vals, y_vals)
        # Spearman rho (rank correlation, more robust)
        rho, p_rho = stats.spearmanr(x_vals, y_vals)

        ax.scatter(x_vals, y_vals, c="#2a6496", alpha=0.7, s=40, edgecolors="white", linewidth=0.3)
        # Annotate significant correlations
        if p_val < 0.05:
            z = np.polyfit(x_vals, y_vals, 1)
            x_line = np.linspace(x_vals.min(), x_vals.max(), 50)
            ax.plot(x_line, np.polyval(z, x_line), color="#c0392b", lw=1.2, alpha=0.7)

        # Show r and rho
        sig_marker = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
        ax.text(0.05, 0.95, f"r={r_val:.2f}{sig_marker}\nρ={rho:.2f}",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7))

        if i == len(pairs) - 1:
            ax.set_xlabel(xl, fontsize=9)
        if j == 0:
            ax.set_ylabel(yl, fontsize=9)

fig.suptitle(f"Parameter correlations: OK proteins with L_∥ ≥ 0.9 nm  (n={len(clean)})\n"
             "r = Pearson  |  ρ = Spearman rank  |  *** p<0.001  ** p<0.01  * p<0.05  ns=not significant",
             fontsize=11, y=0.996)
fig.tight_layout(rect=(0, 0, 1, 0.99))
out = f"{FIGURES}/param_correlation.png"
fig.savefig(out, dpi=160, bbox_inches="tight")
print(f"\n-> {out}")

# ---- also print the strongest correlations ----
print(f"\n{'='*70}")
print("Key findings (Pearson r, Spearman ρ):")
print(f"{'':>3s} {'x':>18s} {'y':>18s} {'r':>7s} {'ρ':>7s} {'p':>10s}")
for yk, yl in pairs:
    for xk, xl in geo_pairs:
        x_vals = np.array([r[xk] for r in clean])
        y_vals = np.array([r[yk] for r in clean])
        r_val, p_val = stats.pearsonr(x_vals, y_vals)
        rho, _ = stats.spearmanr(x_vals, y_vals)
        if abs(r_val) > 0.2 or p_val < 0.05:
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            print(f"{'':>3s} {xl:>18s} {yl:>18s} {r_val:+.3f}{sig} {rho:+.3f}  p={p_val:.4f}")

# ---- list top outliers in xi_hat_par ----
clean.sort(key=lambda r: -r["xi_hat_par"])
print(f"\n{'='*70}")
print("Top 10 by ξ̂_∥ (filtered):")
for r in clean[:10]:
    print(f"  {r['pid']:8s}  L_∥={r['L_par']:.2f}  L_⊥={r['L_perp']:.2f}  "
          f"η={r['eta']:.2f}  ξ̂_∥={r['xi_hat_par']:.2f}  ξ̂_⊥={r['xi_hat_perp']:.2f}  χ={r['chi']:.2f}")

print(f"\nBottom 5 by ξ̂_∥ (weakest in-plane transmission):")
for r in clean[-5:]:
    print(f"  {r['pid']:8s}  L_∥={r['L_par']:.2f}  L_⊥={r['L_perp']:.2f}  "
          f"η={r['eta']:.2f}  ξ̂_∥={r['xi_hat_par']:.2f}  ξ̂_⊥={r['xi_hat_perp']:.2f}  χ={r['chi']:.2f}")

print("\nDONE")
