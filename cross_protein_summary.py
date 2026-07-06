# -*- coding: utf-8 -*-
"""Cross-protein comparison summary: load all scale_*.npz, plot key metrics."""

import glob
import os
import re
import numpy as np
import matplotlib.pyplot as plt

RESULTS = "data/results"
FIGURES = "data/figures"

# ---- 1. load all scale npz ----
records = []
for npz_path in sorted(glob.glob(f"{RESULTS}/scale_*_F0.01.npz")):
    pid = os.path.basename(npz_path).replace("scale_", "").replace("_F0.01.npz", "")
    data = np.load(npz_path)
    rec = {"pid": pid, "L_par": float(data["L_parallel"]),
           "L_perp": float(data["L_perp"]), "eta": float(data["eta_shape"])}
    for k in ("xi_hat_parallel", "xi_hat_perp", "chi_excess"):
        vals = data[k]
        rec[k + "_mean"] = float(np.nanmean(vals))
        rec[k + "_final"] = float(vals[-1])
    records.append(rec)

# ---- 2. damage / skip / warn ----
damage = {}
with open("data/batch.log") as fh:
    for line in fh:
        m = re.match(r"\[(\w+)\s+F=[\d.]+\].*ratio=[\d.]+\s+(\S+)", line)
        if m:
            damage[m.group(1)] = m.group(2)

skipped = {"4xdl", "5unf"}
warned = set()
with open("data/batch.log") as fh:
    for line in fh:
        m = re.match(r"\[(\w+)\] WARN", line)
        if m:
            warned.add(m.group(1))

for r in records:
    r["flag"] = damage.get(r["pid"], "?")
    r["warned"] = r["pid"] in warned

records.sort(key=lambda r: (r["flag"] != "ok", r["pid"]))
ok_records = [r for r in records if r["flag"] == "ok"]
bad_records = [r for r in records if r["flag"] != "ok"]

n_total = len(records) + len(skipped)
print(f"Total selected: {n_total}")
print(f"  OK:       {len(ok_records)}")
print(f"  DAMAGED:  {len(bad_records)}")
print(f"  SKIPPED:  {len(skipped)}  ({', '.join(sorted(skipped))})")
print(f"  WARNED:   {len(warned)}  (partial chain mismatch, still processed)")

# ---- 3. figure ----
pids_ok = [r["pid"] for r in ok_records]
pids_bad = [r["pid"] for r in bad_records]

fig, axes = plt.subplots(2, 2, figsize=(24, 18))

# 3a: xi_hat_parallel bar chart
ax = axes[0, 0]
y_ok = [r["xi_hat_parallel_final"] for r in ok_records]
y_bad = [r["xi_hat_parallel_final"] for r in bad_records]
x_all = list(range(len(records)))
ax.bar(x_all[:len(ok_records)], y_ok, color="steelblue", label="OK")
ax.bar(x_all[len(ok_records):], y_bad, color="salmon", label="DAMAGED")
ax.axhline(1.0, color="gray", ls="--", alpha=0.5, label="ξ̂=1")
ax.set_xticks(x_all)
ax.set_xticklabels(pids_ok + pids_bad, rotation=90, fontsize=6)
ax.set_ylabel("ξ̂_∥  (in-plane final)")
ax.set_title("Normalized in-plane correlation length ξ̂_∥  (loading end, 50 ps)")
ax.legend()

# 3b: xi_hat_perp vs xi_hat_parallel scatter (OK only)
ax = axes[0, 1]
x_vals = [r["xi_hat_parallel_final"] for r in ok_records]
y_vals = [r["xi_hat_perp_final"] for r in ok_records]
sizes = [r["L_par"] * 30 for r in ok_records]
sc = ax.scatter(x_vals, y_vals, c=sizes, cmap="viridis", s=sizes, edgecolors="grey", linewidth=0.3)
ax.set_xlabel("ξ̂_∥ (in-plane)"); ax.set_ylabel("ξ̂_⊥ (perpendicular)")
ax.axhline(1.0, color="gray", ls="--", alpha=0.3); ax.axvline(1.0, color="gray", ls="--", alpha=0.3)
ax.set_title("ξ̂_⊥ vs ξ̂_∥  (OK only, size/colour = protein in-plane size L_∥)")
plt.colorbar(sc, ax=ax, label="L_∥ (nm)")

# 3c: chi_excess histogram
ax = axes[1, 0]
chi_ok = [r["chi_excess_final"] for r in ok_records]
chi_bad = [r["chi_excess_final"] for r in bad_records]
all_chi = chi_ok + chi_bad
bins = np.linspace(min(all_chi) - 0.1, max(all_chi) + 0.1, 35)
ax.hist(chi_ok, bins=bins, color="steelblue", alpha=0.7, label=f"OK (n={len(chi_ok)})")
ax.hist(chi_bad, bins=bins, color="salmon", alpha=0.7, label=f"DAMAGED (n={len(chi_bad)})")
ax.axvline(0, color="gray", ls="--", alpha=0.5)
ax.set_xlabel("χ_excess"); ax.set_ylabel("count")
ax.set_title("χ_excess distribution  (>0 = in-plane dominant, <0 = normal dominant)")
ax.legend()

# 3d: L_par vs L_perp geometry
ax = axes[1, 1]
lp_ok = [(r["L_par"], r["L_perp"], r["eta"]) for r in ok_records]
sc2 = ax.scatter([x[0] for x in lp_ok], [x[1] for x in lp_ok],
                 c=[x[2] for x in lp_ok], cmap="RdYlBu", s=60, edgecolors="grey", linewidth=0.3)
ax.set_xlabel("L_∥ (nm)"); ax.set_ylabel("L_⊥ (nm)")
ax.set_title("Protein geometry: in-plane size vs normal thickness  (colour = η)")
plt.colorbar(sc2, ax=ax, label="η (0=flat disk, 1≈sphere, >1=vertical rod)")

fig.suptitle(f"Cross-protein comparison: {len(ok_records)} OK + {len(bad_records)} DAMAGED  "
             f"(F=0.01 pN/node, loading phase 0–50 ps)\n"
             f"Skipped: {', '.join(sorted(skipped))}   —   "
             f"Warned: {len(warned)} (partial chain mismatch)",
             fontsize=13, y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.98))
out_png = f"{FIGURES}/cross_protein_summary.png"
fig.savefig(out_png, dpi=160, bbox_inches="tight")
print(f"\n-> {out_png}")

# ---- 4. top-N tables ----
print(f"\n{'='*80}")
print("Top 10 by ξ̂_∥ (strongest in-plane force transmission, OK only):")
for r in sorted(ok_records, key=lambda x: -x["xi_hat_parallel_final"])[:10]:
    print(f"  {r['pid']:8s}  ξ̂_∥={r['xi_hat_parallel_final']:.2f}  "
          f"ξ̂_⊥={r['xi_hat_perp_final']:.2f}  χ={r['chi_excess_final']:.2f}  "
          f"L_∥={r['L_par']:.1f}  η={r['eta']:.2f}")

print(f"\nTop 10 by χ_excess (strongest in-plane preference):")
for r in sorted(ok_records, key=lambda x: -x["chi_excess_final"])[:10]:
    print(f"  {r['pid']:8s}  χ={r['chi_excess_final']:.2f}  "
          f"ξ̂_∥={r['xi_hat_parallel_final']:.2f}  ξ̂_⊥={r['xi_hat_perp_final']:.2f}  "
          f"L_∥={r['L_par']:.1f}")

# ---- 5. CSV ----
csv_path = f"{RESULTS}/cross_protein_summary.csv"
with open(csv_path, "w") as f:
    f.write("pid,flag,warned,L_par_nm,L_perp_nm,eta,"
            "xi_hat_par_mean,xi_hat_perp_mean,chi_excess_mean,"
            "xi_hat_par_final,xi_hat_perp_final,chi_excess_final\n")
    for r in records:
        f.write(f"{r['pid']},{r['flag']},{r['warned']},"
                f"{r['L_par']:.3f},{r['L_perp']:.3f},{r['eta']:.3f},"
                f"{r.get('xi_hat_parallel_mean','nan'):.4f},"
                f"{r.get('xi_hat_perp_mean','nan'):.4f},"
                f"{r.get('chi_excess_mean','nan'):.4f},"
                f"{r.get('xi_hat_parallel_final','nan'):.4f},"
                f"{r.get('xi_hat_perp_final','nan'):.4f},"
                f"{r.get('chi_excess_final','nan'):.4f}\n")
print(f"\n-> {csv_path}")
print("DONE")
