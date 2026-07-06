# -*- coding: utf-8 -*-
"""Shape η vs normalized force transmission (ξ̂_∥, ξ̂_⊥, χ_excess).

η = L_∥ / L_perp:  η > 1 = flat disk,  η < 1 = vertical rod.
All metrics are already size-normalized (ξ̂_∥/L_∥, ξ̂_⊥/L_⊥).
"""

import glob, os, re, numpy as np
import matplotlib.pyplot as plt
from scipy import stats

RESULTS = "data/results"; FIGURES = "data/figures"

# Load OK proteins
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
    Lp = float(d["L_parallel"]); Lpe = float(d["L_perp"])
    records.append({
        "pid": pid,
        "L_par": Lp, "L_perp": Lpe, "eta": float(d["eta_shape"]),
        "xih_par": d["xi_hat_parallel"][-1],
        "xih_perp": d["xi_hat_perp"][-1],
        "chi": d["chi_excess"][-1],
    })

# Remove unreliable: L_par < 0.9 nm (xi likely saturated)
clean = [r for r in records if r["L_par"] >= 0.9]
print(f"OK + reliable: {len(clean)} proteins")

# ---- 1. η bins: group proteins by shape ----
eta_vals = np.array([r["eta"] for r in clean])
flat = [r for r in clean if r["eta"] >= 1.0]       # disk-like
medium = [r for r in clean if 0.6 <= r["eta"] < 1.0]  # sphere-like
rod = [r for r in clean if r["eta"] < 0.6]            # rod-like
print(f"flat(η≥1.0): {len(flat)}, medium(0.6≤η<1.0): {len(medium)}, rod(η<0.6): {len(rod)}")

# ---- 2. plot ----
fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# 2a: ξ̂_∥ vs η
ax = axes[0, 0]
colors = ["#e74c3c", "#3498db", "#2ecc71"]
labels = [f"rod η<0.6 (n={len(rod)})", f"medium 0.6≤η<1 (n={len(medium)})", f"flat η≥1 (n={len(flat)})"]
for group, c, label in [(rod, colors[0], labels[0]), (medium, colors[1], labels[1]), (flat, colors[2], labels[2])]:
    xs = [r["eta"] for r in group]; ys = [r["xih_par"] for r in group]
    ax.scatter(xs, ys, c=c, alpha=0.7, s=50, edgecolors="white", linewidth=0.3, label=label)
# Stats per group
for group, c, name in [(rod, colors[0], "rod"), (medium, colors[1], "medium"), (flat, colors[2], "flat")]:
    yv = np.array([r["xih_par"] for r in group])
    ax.axhline(np.mean(yv), color=c, ls="--", alpha=0.4)
    ax.text(0.02 if name == "rod" else 0.98, np.mean(yv), f"{name} mean={np.mean(yv):.2f}", color=c, fontsize=8,
            transform=ax.get_yaxis_transform(), va="bottom", ha="left" if name == "rod" else "right")

# Fit log
x_all = np.array([r["eta"] for r in clean]); y_all = np.array([r["xih_par"] for r in clean])
r_val, p_val = stats.pearsonr(np.log(x_all), y_all)
ax.text(0.05, 0.95, f"r(log η, ξ̂_∥)={r_val:.2f} p={p_val:.3f}",
        transform=ax.transAxes, fontsize=10, va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7))
ax.set_xlabel("η = L_∥ / L_⊥  (shape factor)"); ax.set_ylabel("ξ̂_∥")
ax.set_title("In-plane force transmission vs shape"); ax.legend(fontsize=8)

# 2b: ξ̂_⊥ vs η
ax = axes[0, 1]
for group, c, label in [(rod, colors[0], labels[0]), (medium, colors[1], labels[1]), (flat, colors[2], labels[2])]:
    xs = [r["eta"] for r in group]; ys = [r["xih_perp"] for r in group]
    ax.scatter(xs, ys, c=c, alpha=0.7, s=50, edgecolors="white", linewidth=0.3)
for group, c, name in [(rod, colors[0], "rod"), (medium, colors[1], "medium"), (flat, colors[2], "flat")]:
    yv = np.array([r["xih_perp"] for r in group])
    ax.axhline(np.mean(yv), color=c, ls="--", alpha=0.4)
r2, p2 = stats.pearsonr(x_all, np.array([r["xih_perp"] for r in clean]))
ax.text(0.05, 0.95, f"r={r2:.2f} p={p2:.3f}", transform=ax.transAxes, fontsize=10, va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7))
ax.set_xlabel("η = L_∥ / L_⊥"); ax.set_ylabel("ξ̂_⊥")
ax.set_title("Perpendicular force transmission vs shape")

# 2c: χ_excess vs η
ax = axes[0, 2]
for group, c, label in [(rod, colors[0], labels[0]), (medium, colors[1], labels[1]), (flat, colors[2], labels[2])]:
    xs = [r["eta"] for r in group]; ys = [r["chi"] for r in group]
    ax.scatter(xs, ys, c=c, alpha=0.7, s=50, edgecolors="white", linewidth=0.3)
r3, p3 = stats.pearsonr(x_all, np.array([r["chi"] for r in clean]))
ax.axhline(1.0, color="gray", ls="--", alpha=0.5, label="χ=1 (isotropic)")
ax.text(0.05, 0.95, f"r={r3:.2f} p={p3:.3f}", transform=ax.transAxes, fontsize=10, va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7))
ax.set_xlabel("η = L_∥ / L_⊥"); ax.set_ylabel("χ_excess = ξ̂_∥ / ξ̂_⊥")
ax.set_title("Directional anisotropy vs shape")
ax.legend(fontsize=8)

# ---- 3. Bar chart: mean ξ̂ per shape group ----
ax = axes[1, 0]
groups_data = {"rod (η<0.6)": rod, "medium (0.6≤η<1)": medium, "flat (η≥1)": flat}
x_pos = np.arange(len(groups_data))
width = 0.25
for i, (metric, key, color_i) in enumerate([("ξ̂_∥", "xih_par", "#c0392b"), ("ξ̂_⊥", "xih_perp", "#2980b9")]):
    means = [np.mean([r[key] for r in g]) for g in groups_data.values()]
    sems = [np.std([r[key] for r in g]) / np.sqrt(max(len(g), 1)) for g in groups_data.values()]
    ax.bar(x_pos + i * width, means, width, yerr=sems, color=color_i, alpha=0.8,
           capsize=3, label=metric)
# Mann-Whitney U test: flat vs rod for ξ̂_∥
from scipy.stats import mannwhitneyu
u1, p_u1 = mannwhitneyu([r["xih_par"] for r in rod], [r["xih_par"] for r in flat])
ax.text(0.5, 0.95, f"rod vs flat ξ̂_∥: MWU p={p_u1:.3f}", transform=ax.transAxes, ha="center",
        fontsize=9, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))
ax.set_xticks(x_pos + width); ax.set_xticklabels(groups_data.keys())
ax.set_ylabel("normalized correlation length"); ax.set_title("ξ̂ by shape group (±SEM)")
ax.legend()

# 3e: example proteins at extremes
ax = axes[1, 1]
ax.axis("off")
text = ("Shape extremes examples:\n\n"
        "FLATTEST (η = L_∥/L_⊥ largest → wide disk):\n"
        + "\n".join(f"  {r['pid']:6s}  η={r['eta']:.2f}  L_∥={r['L_par']:.1f}nm  L_⊥={r['L_perp']:.1f}nm"
                     for r in sorted(clean, key=lambda x: -x["eta"])[:5])
        + "\n\nRODDIEST (η smallest → tall column):\n"
        + "\n".join(f"  {r['pid']:6s}  η={r['eta']:.2f}  L_∥={r['L_par']:.1f}nm  L_⊥={r['L_perp']:.1f}nm"
                     for r in sorted(clean, key=lambda x: x["eta"])[:5]))
ax.text(0, 1, text, fontfamily="monospace", fontsize=9, va="top", transform=ax.transAxes)

# 3f: force load effect (only 12 proteins with 3 forces)
ax = axes[1, 2]
ax.axis("off")
# Find proteins with data at all 3 forces
force_effect = []
for pid in sorted(set(r["pid"] for r in clean)):
    scales = []
    for F in ["0.01", "0.05", "0.1"]:
        path = f"{RESULTS}/scale_{pid}_F{F}.npz"
        if os.path.exists(path):
            d = np.load(path)
            scales.append((float(F), d["xi_hat_parallel"][-1], d["chi_excess"][-1]))
    if len(scales) == 3:
        force_effect.append((pid, scales))

if force_effect:
    text2 = "Force-dependence (proteins with 3 forces):\n\n"
    for pid, scales in force_effect[:8]:
        text2 += f"  {pid}:\n"
        for F, xih, chi in scales:
            text2 += f"    F={F:.2f}  ξ̂_∥={xih:.2f}  χ={chi:.2f}\n"
else:
    text2 = "Only 12/64 proteins have data at\nF=0.05 and F=0.1 (the original 12).\n\nRe-run with --forces 0.01 0.05 0.1 for the\nfull 64-protein set to explore load effects."
ax.text(0, 1, text2, fontfamily="monospace", fontsize=8.5, va="top", transform=ax.transAxes)

fig.suptitle(f"Shape factor η vs size-normalized force transmission  (F=0.01 pN, n={len(clean)} OK proteins, L_∥ ≥ 0.9 nm)",
             fontsize=12, y=0.996)
fig.tight_layout(rect=(0, 0, 1, 0.99))

out = f"{FIGURES}/shape_vs_transmission.png"
fig.savefig(out, dpi=160, bbox_inches="tight")
print(f"\n-> {out}")

# ---- 4. numerical summary ----
print(f"\n{'='*70}")
print("Shape group summary:")
for name, group in groups_data.items():
    xih_par_mean = np.mean([r["xih_par"] for r in group])
    xih_perp_mean = np.mean([r["xih_perp"] for r in group])
    chi_mean = np.mean([r["chi"] for r in group])
    eta_mean = np.mean([r["eta"] for r in group])
    print(f"  {name:25s} n={len(group):2d}  η̄={eta_mean:.2f}  ξ̂_∥̄={xih_par_mean:.2f}  ξ̂_⊥̄={xih_perp_mean:.2f}  χ̄={chi_mean:.2f}")

print(f"\nMann-Whitney U: rod vs flat ξ̂_∥: p={p_u1:.4f}")
print(f"Mann-Whitney U: rod vs flat ξ̂_⊥: ", end="")
u2, p_u2 = mannwhitneyu([r["xih_perp"] for r in rod], [r["xih_perp"] for r in flat])
print(f"p={p_u2:.4f}")
print(f"Mann-Whitney U: rod vs flat χ_excess: ", end="")
u3, p_u3 = mannwhitneyu([r["chi"] for r in rod], [r["chi"] for r in flat])
print(f"p={p_u3:.4f}")

print("\nDONE")
