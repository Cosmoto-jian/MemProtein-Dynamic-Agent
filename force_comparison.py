# -*- coding: utf-8 -*-
"""Compare ξ̂ and χ across F=0.01, 0.05, 0.1 for all OK proteins."""

import glob, os, re, numpy as np
import matplotlib.pyplot as plt
from scipy import stats

RESULTS = "data/results"; FIGURES = "data/figures"
FORCES = ["0.01", "0.05", "0.1"]

# ---- load all proteins with all 3 forces ----
damage = {}
for log_file in ("data/batch.log", "data/batch_forces.log"):
    try:
        with open(log_file) as fh:
            for line in fh:
                m = re.match(r"\[(\w+)\s+F=[\d.]+\].*ratio=[\d.]+\s+(\S+)", line)
                if m:
                    # Keep the latest status (F=0.1 overrides F=0.01)
                    damage[m.group(1)] = m.group(2)
    except FileNotFoundError:
        pass

# Load F=0.01 data as reference
proteins = {}
for npz_path in glob.glob(f"{RESULTS}/scale_*_F0.01.npz"):
    pid = os.path.basename(npz_path)
    pid = pid.replace("scale_", "")
    pid = pid.replace("_F0.01.npz", "")
    d0 = np.load(npz_path)
    Lp = float(d0["L_parallel"])
    if Lp < 0.9:
        continue
    proteins[pid] = {"eta": float(d0["eta_shape"]), "L_par": Lp,
                     "L_perp": float(d0["L_perp"])}
    for F in FORCES:
        path = f"{RESULTS}/scale_{pid}_F{F}.npz"
        if os.path.exists(path):
            d = np.load(path)
            proteins[pid][f"xih_par_F{F}"] = d["xi_hat_parallel"][-1]
            proteins[pid][f"xih_perp_F{F}"] = d["xi_hat_perp"][-1]
            proteins[pid][f"chi_F{F}"] = d["chi_excess"][-1]

# Filter to only those with all 3 forces
complete = {
    pid: data for pid, data in proteins.items()
    if all(f"xih_par_F{F}" in data for F in FORCES)
}
print(f"Complete data (3 forces): {len(complete)} proteins")
missing = {pid for pid in proteins if pid not in complete}
if missing:
    print(f"Missing some force data: {len(missing)} ({', '.join(sorted(missing)[:10])}...)")

if len(complete) < 10:
    print("Not enough complete data — batch may still be running. Exiting.")
    print("Complete so far:", sorted(complete.keys()))
    exit()
# ---- shape groups ----
rod = {p: d for p, d in complete.items() if d["eta"] < 0.6}
medium = {p: d for p, d in complete.items() if 0.6 <= d["eta"] < 1.0}
flat = {p: d for p, d in complete.items() if d["eta"] >= 1.0}

# ---- 1. Trajectory plot: xih_par for each protein across forces ----
fig, axes = plt.subplots(2, 3, figsize=(22, 14))

# 1a: ξ̂_∥ trajectory per protein
ax = axes[0, 0]
force_vals = [0.01, 0.05, 0.1]
colors_map = {"rod": "#e74c3c", "medium": "#3498db", "flat": "#2ecc71"}
for group, color, label in [
    (rod, colors_map["rod"], f"rod η<0.6 (n={len(rod)})"),
    (medium, colors_map["medium"], f"medium (n={len(medium)})"),
    (flat, colors_map["flat"], f"flat η≥1 (n={len(flat)})"),
]:
    for pid, data in group.items():
        ys = [data[f"xih_par_F{F}"] for F in FORCES]
        ax.plot(force_vals, ys, "o-", color=color, alpha=0.25, lw=0.6, markersize=3)
ax.set_xlabel("Fmax (pN/node)"); ax.set_ylabel("ξ̂_∥")
ax.set_title("In-plane transmission vs load (individual proteins)")
ax.set_xscale("log")

# 1b: ξ̂_∥ group mean ± SEM
ax = axes[0, 1]
for group, color, label in [
    (rod, colors_map["rod"], f"rod η<0.6 (n={len(rod)})"),
    (medium, colors_map["medium"], f"medium (n={len(medium)})"),
    (flat, colors_map["flat"], f"flat η≥1 (n={len(flat)})"),
]:
    means = [np.nanmean([d.get(f"xih_par_F{F}", np.nan) for d in group.values()]) for F in FORCES]
    sems = [np.std([d[f"xih_par_F{F}"] for d in group.values()]) / np.sqrt(len(group))
            for F in FORCES]
    ax.errorbar(force_vals, means, yerr=sems, fmt="o-", color=color, lw=2, markersize=8,
                capsize=5, label=label, markeredgewidth=0)
ax.set_xlabel("Fmax (pN/node)"); ax.set_ylabel("ξ̂_∥ (mean ± SEM)")
ax.set_title("In-plane transmission: group average vs load")
ax.legend(fontsize=8); ax.set_xscale("log")

# 1c: change ratio ξ̂_∥(0.1)/ξ̂_∥(0.01) histogram
ax = axes[1, 0]
for group, color, label in [
    (rod, colors_map["rod"], f"rod"),
    (medium, colors_map["medium"], f"medium"),
    (flat, colors_map["flat"], f"flat"),
]:
    ratios = []
    for d in group.values():
        if d.get("xih_par_F0.01", 0) > 0.01 and d.get("xih_par_F0.1", 0) > 0:
            ratios.append(d["xih_par_F0.1"] / d["xih_par_F0.01"])
    ax.hist(ratios, bins=15, alpha=0.5, color=color, label=f"{label} (μ={np.mean(ratios):.2f})")
ax.axvline(1.0, color="gray", ls="--", alpha=0.5)
ax.set_xlabel("ξ̂_∥(0.1) / ξ̂_∥(0.01)"); ax.set_ylabel("count")
ax.set_title("Force effect ratio  (<1 = nonlinear softening, >1 = stiffening)")
ax.legend(fontsize=8)

# 1d: χ_excess group mean vs load
ax = axes[1, 1]
for group, color, label in [
    (rod, colors_map["rod"], f"rod η<0.6 (n={len(rod)})"),
    (medium, colors_map["medium"], f"medium (n={len(medium)})"),
    (flat, colors_map["flat"], f"flat η≥1 (n={len(flat)})"),
]:
    means = [np.mean([d[f"chi_F{F}"] for d in group.values()]) for F in FORCES]
    sems = [np.std([d[f"chi_F{F}"] for d in group.values()]) / np.sqrt(len(group))
            for F in FORCES]
    ax.errorbar(force_vals, means, yerr=sems, fmt="s-", color=color, lw=2, markersize=8,
                capsize=5, label=label)
ax.axhline(1.0, color="gray", ls="--", alpha=0.5, label="isotropic")
ax.set_xlabel("Fmax (pN/node)"); ax.set_ylabel("χ_excess (mean ± SEM)")
ax.set_title("Directional anisotropy vs load"); ax.set_xscale("log")
ax.legend(fontsize=8)

# 1e: xih_perp vs load
ax = axes[1, 2]
for group, color, label in [
    (rod, colors_map["rod"], f"rod η<0.6"),
    (medium, colors_map["medium"], f"medium"),
    (flat, colors_map["flat"], f"flat η≥1"),
]:
    means = [np.mean([d[f"xih_perp_F{F}"] for d in group.values()]) for F in FORCES]
    sems = [np.std([d[f"xih_perp_F{F}"] for d in group.values()]) / np.sqrt(len(group))
            for F in FORCES]
    ax.errorbar(force_vals, means, yerr=sems, fmt="d-", color=color, lw=2, markersize=8,
                capsize=5, label=label)
ax.set_xlabel("Fmax (pN/node)"); ax.set_ylabel("ξ̂_⊥ (mean ± SEM)")
ax.set_title("Perpendicular transmission vs load"); ax.set_xscale("log")
ax.legend(fontsize=8)

# 1f: top responders table
ax = axes[0, 2]
ax.axis("off")
# Rank by force sensitivity: (ξ̂_∥(0.1) - ξ̂_∥(0.01)) / ξ̂_∥(0.01)
sensitivities = []
for pid, data in complete.items():
    if data.get("xih_par_F0.01", 0) > 0.01 and data.get("xih_par_F0.1", 0) > 0:
        sens = (data["xih_par_F0.1"] - data["xih_par_F0.01"]) / data["xih_par_F0.01"]
        sensitivities.append((pid, sens, data["eta"], data["xih_par_F0.01"],
                              data["xih_par_F0.1"]))

text_lines = [
    "Top force-responders (ξ̂_∥ change F0.01→F0.1):",
    "",
    "STIFFENERS (most positive change):",
]
for pid, sens, eta, x1, x10 in sorted(sensitivities, key=lambda x: -x[1])[:5]:
    text_lines.append(f"  {pid:6s}  η={eta:.2f}  {x1:.2f}→{x10:.2f}  Δ={sens:+.0%}")

text_lines.append("")
text_lines.append("SOFTENERS (most negative change):")
for pid, sens, eta, x1, x10 in sorted(sensitivities, key=lambda x: x[1])[:5]:
    text_lines.append(f"  {pid:6s}  η={eta:.2f}  {x1:.2f}→{x10:.2f}  Δ={sens:+.0%}")

ax.text(0, 1, "\n".join(text_lines), fontfamily="monospace", fontsize=8.5,
        va="top", transform=ax.transAxes)

fig.suptitle(f"Force-dependence: F=0.01 → 0.05 → 0.1 pN/node  ({len(complete)} proteins)",
             fontsize=12, y=0.996)
fig.tight_layout(rect=(0, 0, 1, 0.99))
out1 = f"{FIGURES}/force_comparison.png"
fig.savefig(out1, dpi=160, bbox_inches="tight")
print(f"-> {out1}")

# ---- 2. Summary table ----
print(f"\n{'='*90}")
print(f"{'group':15s} {'n':>4s}  {'η̄':>6s}  "
      f"{'ξ̂_∥₀.₀₁':>8s} {'ξ̂_∥₀.₀₅':>8s} {'ξ̂_∥₀.₁':>8s}  "
      f"{'ξ̂_⊥₀.₀₁':>8s} {'ξ̂_⊥₀.₀₅':>8s} {'ξ̂_⊥₀.₁':>8s}  "
      f"{'χ₀.₀₁':>7s} {'χ₀.₀₅':>7s} {'χ₀.₁':>7s}")
print("-" * 90)
for name, group in [("rod η<0.6", rod), ("medium", medium), ("flat η≥1", flat)]:
    n = len(group)
    if n == 0: continue
    xp01 = np.mean([d.get("xih_par_F0.01", np.nan) for d in group.values()])
    xp05 = np.mean([d.get("xih_par_F0.05", np.nan) for d in group.values()])
    xp10 = np.mean([d.get("xih_par_F0.1", np.nan) for d in group.values()])
    xpe01 = np.mean([d.get("xih_perp_F0.01", np.nan) for d in group.values()])
    xpe05 = np.mean([d.get("xih_perp_F0.05", np.nan) for d in group.values()])
    xpe10 = np.mean([d.get("xih_perp_F0.1", np.nan) for d in group.values()])
    c01 = np.mean([d.get("chi_F0.01", np.nan) for d in group.values()])
    c05 = np.mean([d.get("chi_F0.05", np.nan) for d in group.values()])
    c10 = np.mean([d.get("chi_F0.1", np.nan) for d in group.values()])
    eta = np.mean([d.get("eta", np.nan) for d in group.values()])
    print(f"{name:15s} {n:>4d}  {eta:>5.2f}  "
          f"{xp01:>8.2f} {xp05:>8.2f} {xp10:>8.2f}  "
          f"{xpe01:>8.2f} {xpe05:>8.2f} {xpe10:>8.2f}  "
          f"{c01:>7.2f} {c05:>7.2f} {c10:>7.2f}")
print(f"\n{'all combined':15s} {len(complete):>4d}  "
      f"{np.nanmean([d.get('eta',np.nan) for d in complete.values()]):>5.2f}  "
      f"{np.nanmean([d.get('xih_par_F0.01',np.nan) for d in complete.values()]):>8.2f} "
      f"{np.nanmean([d.get('xih_par_F0.05',np.nan) for d in complete.values()]):>8.2f} "
      f"{np.nanmean([d.get('xih_par_F0.1',np.nan) for d in complete.values()]):>8.2f}")

# ---- 3. CSV export ----
csv_path = f"{RESULTS}/force_comparison.csv"
with open(csv_path, "w") as f:
    f.write("pid,eta,shape_group,"
            + ",".join(f"xih_par_{F},xih_perp_{F},chi_{F}" for F in FORCES)
            + "\n")
    for pid, data in sorted(complete.items()):
        eta = data["eta"]
        if eta < 0.6: grp = "rod"
        elif eta >= 1.0: grp = "flat"
        else: grp = "medium"
        vals = [f"{data.get(f'xih_par_F{F}',float('nan')):.4f},"
                f"{data.get(f'xih_perp_F{F}',float('nan')):.4f},"
                f"{data.get(f'chi_F{F}',float('nan')):.4f}" for F in FORCES]
        f.write(f"{pid},{eta:.3f},{grp}," + ",".join(vals) + "\n")
print(f"\n-> {csv_path}")
print("DONE")
