# -*- coding: utf-8 -*-
"""Flat vs rod-like: senior's η = L_z / L_xy, comparison across 3 forces."""

import glob, os, re, numpy as np
import matplotlib.pyplot as plt
from scipy import stats

RESULTS = "data/results"; FIGURES = "data/figures"; FORCES = ["0.01", "0.05", "0.1"]

# ---- load ----
records = []
for npz_path in sorted(glob.glob(f"{RESULTS}/scale_*_F0.01.npz")):
    pid = os.path.basename(npz_path).replace("scale_", "").replace("_F0.01.npz", "")
    d0 = np.load(npz_path)
    Lxy = float(d0["L_parallel"]); Lz = float(d0["L_perp"])
    if Lxy < 0.9:
        continue
    eta_s = Lz / Lxy   # senior: η = L_z / L_xy, >1=rod, <1=flat
    rec = {"pid": pid, "eta": eta_s, "Lxy": Lxy, "Lz": Lz}
    for F in FORCES:
        path = f"{RESULTS}/scale_{pid}_F{F}.npz"
        if os.path.exists(path):
            d = np.load(path)
            rec[f"xih_par_F{F}"] = d["xi_hat_parallel"][-1]
            rec[f"xih_perp_F{F}"] = d["xi_hat_perp"][-1]
            rec[f"chi_F{F}"] = d["chi_excess"][-1]
    records.append(rec)

complete = [r for r in records if all(f"xih_par_F{F}" in r for F in FORCES)]
flat = [r for r in complete if r["eta"] <= 1.0]
rod = [r for r in complete if r["eta"] > 1.0]
print(f"Complete: {len(complete)} | Flat (η≤1): {len(flat)} | Rod (η>1): {len(rod)}")

# ---- figure ----
fig, axes = plt.subplots(2, 3, figsize=(22, 13))

# 1a: η (senior) vs ξ̂_∥_F0.01
ax = axes[0, 0]
for group, c, lbl in [(flat, "#2ecc71", f"flat η≤1 (n={len(flat)})"),
                        (rod, "#e74c3c", f"rod η>1 (n={len(rod)})")]:
    ax.scatter([r["eta"] for r in group], [r["xih_par_F0.01"] for r in group],
               c=c, alpha=0.7, s=50, edgecolors="white", linewidth=0.3, label=lbl)
ax.axvline(1.0, color="gray", ls="--", alpha=0.4)
x_all = np.array([r["eta"] for r in complete])
y_all = np.array([r["xih_par_F0.01"] for r in complete])
r_val, p_val = stats.pearsonr(np.log10(x_all), y_all)
ax.text(0.05, 0.95, f"r(log η, ξ̂_∥)={r_val:.2f} p={p_val:.3f}",
        transform=ax.transAxes, fontsize=10, va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))
ax.set_xlabel("η = L_z / L_xy  (>1 = tall rod, <1 = flat disk)")
ax.set_ylabel("ξ̂_∥ = ξ_xy / L_xy")
ax.set_title("In-plane transmission vs shape (senior's η)"); ax.legend(fontsize=9)

# 1b: ξ̂_⊥ vs η
ax = axes[0, 1]
for group, c, lbl in [(flat, "#2ecc71", f"flat η≤1"), (rod, "#e74c3c", f"rod η>1")]:
    ax.scatter([r["eta"] for r in group], [r["xih_perp_F0.01"] for r in group],
               c=c, alpha=0.7, s=50, edgecolors="white", linewidth=0.3)
ax.axvline(1.0, color="gray", ls="--", alpha=0.4)
y_perp = np.array([r["xih_perp_F0.01"] for r in complete])
r2, p2 = stats.pearsonr(x_all, y_perp)
ax.text(0.05, 0.95, f"r={r2:.2f} p={p2:.3f}", transform=ax.transAxes, fontsize=10, va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))
ax.set_xlabel("η = L_z / L_xy"); ax.set_ylabel("ξ̂_⊥ = ξ_z / L_z")
ax.set_title("Perpendicular transmission vs shape")

# 1c: ξ̂_⊥ vs ξ̂_∥ coloured by η
ax = axes[0, 2]
etas = [r["eta"] for r in complete]
sc = ax.scatter([r["xih_par_F0.01"] for r in complete],
                [r["xih_perp_F0.01"] for r in complete],
                c=etas, cmap="RdYlBu_r", s=60, edgecolors="grey", linewidth=0.3)
ax.set_xlabel("ξ̂_∥"); ax.set_ylabel("ξ̂_⊥")
ax.axhline(1.0, color="gray", ls="--", alpha=0.3); ax.axvline(1.0, color="gray", ls="--", alpha=0.3)
ax.set_title("ξ̂_⊥ vs ξ̂_∥  (colour = η_senior)")
plt.colorbar(sc, ax=ax, label="η = L_z / L_xy")

# ---- 2. force-dependent bar chart ----
ax = axes[1, 0]
groups_data = {"flat η≤1": flat, "rod η>1": rod}
x_pos = np.arange(len(groups_data)); width = 0.25
for i, (F, ci) in enumerate([("F0.01", "#2980b9"), ("F0.05", "#27ae60"), ("F0.1", "#c0392b")]):
    means = [np.mean([r[f"xih_par_{F}"] for r in g]) for g in groups_data.values()]
    sems = [np.std([r[f"xih_par_{F}"] for r in g]) / np.sqrt(max(len(g), 1)) for g in groups_data.values()]
    ax.bar(x_pos + i * width, means, width, yerr=sems, color=ci, alpha=0.8, capsize=3,
           label=f"F={F.replace('F0.','0.')}")
ax.set_xticks(x_pos + width); ax.set_xticklabels(groups_data.keys())
ax.set_ylabel("ξ̂_∥ (mean ± SEM)"); ax.set_title("In-plane transmission by shape × force")
ax.legend(fontsize=8)
u1, p1 = stats.mannwhitneyu([r["xih_par_F0.01"] for r in flat], [r["xih_par_F0.01"] for r in rod])
ax.text(0.5, 0.95, f"MWU flat vs rod: p={p1:.3f}", transform=ax.transAxes, ha="center",
        fontsize=9, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))

# 2e: force trajectories
ax = axes[1, 1]
force_vals = [0.01, 0.05, 0.1]
for group, c, m in [(flat, "#2ecc71", "o"), (rod, "#e74c3c", "s")]:
    for r in group:
        ys = [r[f"xih_par_F{F}"] for F in FORCES]
        ax.plot(force_vals, ys, m + "-", color=c, alpha=0.2, lw=0.5, markersize=3)
for group, c, lbl, m in [(flat, "#2ecc71", f"flat η≤1 (n={len(flat)})", "o"),
                           (rod, "#e74c3c", f"rod η>1 (n={len(rod)})", "s")]:
    means = [np.mean([r[f"xih_par_F{F}"] for r in group]) for F in FORCES]
    sems = [np.std([r[f"xih_par_F{F}"] for r in group]) / np.sqrt(len(group)) for F in FORCES]
    ax.errorbar(force_vals, means, yerr=sems, fmt=m + "-", color=c, lw=2.5, markersize=10,
                capsize=5, label=lbl, markeredgewidth=0)
ax.set_xlabel("Fmax (pN/node)"); ax.set_ylabel("ξ̂_∥"); ax.set_xscale("log")
ax.set_title("In-plane transmission: flat vs rod across forces"); ax.legend(fontsize=8)

# 2f: extreme examples
ax = axes[1, 2]; ax.axis("off")
lines = ["Shape extremes (senior's η = L_z / L_xy)", "", "FLATTEST (η ≤ 1, wider-than-tall):"]
for r in sorted(flat, key=lambda x: x["eta"])[:7]:
    lines.append(f"  {r['pid']:6s}  η={r['eta']:.2f}  Lxy={r['Lxy']:.1f}nm  Lz={r['Lz']:.1f}nm  ξ̂_∥={r.get('xih_par_F0.01',0):.2f}")
lines.append("")
lines.append("RODDIEST (η > 1, taller-than-wide):")
for r in sorted(rod, key=lambda x: -x["eta"])[:7]:
    lines.append(f"  {r['pid']:6s}  η={r['eta']:.2f}  Lxy={r['Lxy']:.1f}nm  Lz={r['Lz']:.1f}nm  ξ̂_∥={r.get('xih_par_F0.01',0):.2f}")
ax.text(0, 1, "\n".join(lines), fontfamily="monospace", fontsize=8.5, va="top", transform=ax.transAxes)

fig.suptitle(f"Flat vs rod membrane proteins: senior's η = L_z / L_xy  (F=0.01–0.1 pN, {len(complete)} proteins, L_xy≥0.9nm)",
             fontsize=12, y=0.996)
fig.tight_layout(rect=(0, 0, 1, 0.99))
out1 = f"{FIGURES}/flat_vs_rod.png"
fig.savefig(out1, dpi=160, bbox_inches="tight")
print(f"-> {out1}")

# ---- table ----
print(f"\n{'='*90}")
print(f"{'group':15s} {'n':>4s}  {'η̄':>7s}  {'ξ̂_∥₀.₀₁':>8s} {'ξ̂_∥₀.₀₅':>8s} {'ξ̂_∥₀.₁':>8s}  {'ξ̂_⊥₀.₀₁':>8s} {'ξ̂_⊥₀.₀₅':>8s} {'ξ̂_⊥₀.₁':>8s}  {'χ₀.₀₁':>7s} {'χ₀.₀₅':>7s} {'χ₀.₁':>7s}")
print("-" * 90)
for name, group in [("flat η≤1", flat), ("rod η>1", rod)]:
    n = len(group)
    if n == 0: continue
    vals = {}
    for k in ["xih_par_F0.01", "xih_par_F0.05", "xih_par_F0.1",
              "xih_perp_F0.01", "xih_perp_F0.05", "xih_perp_F0.1",
              "chi_F0.01", "chi_F0.05", "chi_F0.1"]:
        vals[k] = np.mean([r[k] for r in group])
    eta_m = np.mean([r["eta"] for r in group])
    print(f"{name:15s} {n:>4d}  {eta_m:>7.2f}  "
          f"{vals['xih_par_F0.01']:>8.2f} {vals['xih_par_F0.05']:>8.2f} {vals['xih_par_F0.1']:>8.2f}  "
          f"{vals['xih_perp_F0.01']:>8.2f} {vals['xih_perp_F0.05']:>8.2f} {vals['xih_perp_F0.1']:>8.2f}  "
          f"{vals['chi_F0.01']:>7.2f} {vals['chi_F0.05']:>7.2f} {vals['chi_F0.1']:>7.2f}")

print(f"\nflat vs rod ξ̂_∥: MWU p={p1:.4f}")

# MWU for perp
u2, p2 = stats.mannwhitneyu([r["xih_perp_F0.01"] for r in flat], [r["xih_perp_F0.01"] for r in rod])
print(f"flat vs rod ξ̂_⊥: MWU p={p2:.4f}")
print("DONE")
