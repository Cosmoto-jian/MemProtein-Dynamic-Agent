# -*- coding: utf-8 -*-
"""Final 5-class comparison figure — whole-protein η fix."""

import numpy as np, os, glob
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy import stats

RESULTS = "data/results"; FIGURES = "data/figures"

# ---- load ----
by_class = defaultdict(list)
for p in glob.glob(f"{RESULTS}/scale_*_F0.01.npz"):
    pid = os.path.basename(p).replace("scale_","").replace("_F0.01.npz","")
    d = np.load(p)
    Lxy = float(d["L_parallel"]); Lz = float(d["L_perp"])
    eta = Lz / Lxy if Lxy > 0 else 0
    xih_p = float(d["xi_hat_parallel"][-1])
    xih_pe = float(d["xi_hat_perp"][-1])
    chi = float(d["chi_excess"][-1])
    if any(np.isnan(v) or np.isinf(v) for v in [xih_p, xih_pe, chi, eta]):
        continue
    if eta <= 0.25: cls = "extreme_flat"
    elif eta <= 0.50: cls = "very_flat"
    elif eta <= 0.75: cls = "moderate"
    elif eta <= 1.00: cls = "slight_flat"
    else: cls = "rod"
    by_class[cls].append({"pid":pid,"eta":eta,"xih_par":xih_p,"xih_perp":xih_pe,"chi":chi,"Lxy":Lxy,"Lz":Lz})

# ---- figure ----
fig, axes = plt.subplots(2, 2, figsize=(20, 14))

# 1a: ξ̂_∥ vs η, coloured by class
ax = axes[0, 0]
colors = {"extreme_flat":"#c0392b","very_flat":"#e67e22","moderate":"#f1c40f","slight_flat":"#2ecc71","rod":"#3498db"}
cls_order = ["extreme_flat","very_flat","moderate","slight_flat","rod"]
for cls in cls_order:
    g = by_class.get(cls, [])
    if not g: continue
    xs = [r["eta"] for r in g]; ys = [r["xih_par"] for r in g]
    ax.scatter(xs, ys, c=colors[cls], alpha=0.7, s=40, edgecolors="white", linewidth=0.3, label=f"{cls} (n={len(g)})")
ax.set_xlabel("η = L_z / L_xy  (>1=rod, <1=flat)"); ax.set_ylabel("ξ̂_∥")
ax.set_title("In-plane transmission ξ̂_∥ vs shape η")
ax.legend(fontsize=8); ax.axvline(1.0, color="gray", ls="--", alpha=0.4)

# 1b: χ vs η
ax = axes[0, 1]
for cls in cls_order:
    g = by_class.get(cls, [])
    if not g: continue
    ax.scatter([r["eta"] for r in g], [r["chi"] for r in g], c=colors[cls], alpha=0.7, s=40, edgecolors="white", linewidth=0.3)
ax.set_xlabel("η = L_z / L_xy"); ax.set_ylabel("χ_excess = ξ̂_∥ / ξ̂_⊥")
ax.set_title("Directional anisotropy χ vs shape η")
ax.axhline(1.0, color="gray", ls="--", alpha=0.4, label="χ=1 (isotropic)"); ax.legend(fontsize=8)

# 2a: merged bar chart
ax = axes[1, 0]
groups = [("extreme\nflat", by_class.get("extreme_flat",[])),
          ("very_flat", by_class.get("very_flat",[])),
          ("moderate", by_class.get("moderate",[])),
          ("slight\nflat", by_class.get("slight_flat",[])),
          ("rod", by_class.get("rod",[]))]
x_pos = np.arange(len(groups))
width = 0.35
for i, (key, color_i) in enumerate([("xih_par","#2980b9"),("chi","#c0392b")]):
    label = "ξ̂_∥" if key=="xih_par" else "χ_excess"
    means = [np.mean([r[key] for r in g]) if g else 0 for _, g in groups]
    sems = [np.std([r[key] for r in g])/np.sqrt(max(len(g),1)) if g else 0 for _, g in groups]
    ax.bar(x_pos + i*width, means, width, yerr=sems, color=color_i, alpha=0.8, capsize=3, label=label)
    for j, (name, g) in enumerate(groups):
        if g: ax.text(j + i*width, means[j] + 0.05, str(len(g)), ha="center", fontsize=7)
ax.set_xticks(x_pos + width/2); ax.set_xticklabels([n for n,_ in groups])
ax.set_title("ξ̂_∥ and χ by shape class (±SEM, n=count)"); ax.legend()

# 2b: summary table
ax = axes[1, 1]; ax.axis("off")
lines = ["Summary (F=0.01 pN, 123 proteins, whole-protein η)",
         "η = L_z / L_xy  (senior's definition)",
         "",
         f"{'class':15s} {'n':>4s} {'η̄':>6s} {'ξ̂_∥̄':>7s} {'ξ̂_⊥̄':>7s} {'χ̄':>6s}"]
for cls in cls_order:
    g = by_class.get(cls, [])
    if not g: continue
    lines.append(f"{cls:15s} {len(g):>4d} "
                 f"{np.mean([r['eta'] for r in g]):>6.2f} "
                 f"{np.mean([r['xih_par'] for r in g]):>7.2f} "
                 f"{np.mean([r['xih_perp'] for r in g]):>7.2f} "
                 f"{np.mean([r['chi'] for r in g]):>6.2f}")
# Merged
flat = by_class.get("extreme_flat",[])+by_class.get("very_flat",[])
slight = by_class.get("moderate",[])+by_class.get("slight_flat",[])
rod = by_class.get("rod",[])
for name, g in [("flat(η≤0.5)", flat), ("slight(0.5-1)", slight), ("rod(>1)", rod)]:
    lines.append(f"{name:15s} {len(g):>4d} "
                 f"{np.mean([r['eta'] for r in g]):>6.2f} "
                 f"{np.mean([r['xih_par'] for r in g]):>7.2f} "
                 f"{np.mean([r['xih_perp'] for r in g]):>7.2f} "
                 f"{np.mean([r['chi'] for r in g]):>6.2f}")
ax.text(0, 1, "\n".join(lines), fontfamily="monospace", fontsize=8.5, va="top", transform=ax.transAxes)

fig.suptitle("Membrane protein shape vs force transmission (whole-protein geometry, F=0.01 pN/node)", fontsize=12, y=0.996)
fig.tight_layout(rect=(0,0,1,0.99))
out = f"{FIGURES}/final_summary.png"
fig.savefig(out, dpi=160, bbox_inches="tight")
print(f"-> {out}")
print("DONE")
