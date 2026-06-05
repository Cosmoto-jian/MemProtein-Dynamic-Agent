#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Anchor-vs-all correlation at one time.

Fix one "anchor" residue and plot, against every other residue, the
instantaneous motion correlation (Z sign and XY cosine, relative to t=0) versus
that residue's distance from the anchor AT the chosen time. Every other node is
shown (no sampling): for Piezo1 that is 4553 points.

Run it directly (no command-line args needed):
    .venv/bin/python Analysis/anchor_correlation.py
Edit the CONFIG block below to change the anchor / time / file.
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# import the shared correlation helpers from the sibling module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from instant_correlation import load_trajectory, displacements, corr_z_pairs, corr_xy_pairs

# ============================================================
#  配置 —— 直接改这里(CONFIG: edit these)
# ============================================================
H5_FILE   = "simulation_data.h5"    # 仿真结果文件(相对项目根目录或绝对路径)
ANCHOR    = 1                       # 锚点粒子编号(1 号),1-based 节点 ID
TIME_PS   = 50.0                    # 看哪个时刻(ps)
REF_FRAME = 0                       # t=0 参考帧(算位移的基准)
DMAX      = 0.0                     # 只看多少 nm 以内(0 = 全部距离)
OUT       = "Analysis/anchor_corr"  # 输出前缀(会生成 .png 和 .npz)
# ============================================================


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    h5_path = H5_FILE if os.path.isabs(H5_FILE) else os.path.join(root, H5_FILE)

    coords, time, _ = load_trajectory(h5_path)
    n_nodes = coords.shape[1]
    frame = int(np.argmin(np.abs(time - TIME_PS)))
    print(f"{n_nodes} nodes, anchor = node {ANCHOR}, time = {time[frame]:.1f} ps")

    a = ANCHOR - 1                                   # 0-based anchor index
    others = np.array([j for j in range(n_nodes) if j != a])
    pairs = np.stack([np.full(len(others), a), others], axis=1)

    # correlations at the chosen frame
    dx, dy, dz = displacements(coords, ref=REF_FRAME)
    cz = corr_z_pairs(dz, pairs)[frame]              # (N-1,) in {-1,0,+1}
    cxy = corr_xy_pairs(dx, dy, pairs)[frame]        # (N-1,) in [-1,+1]

    # distance from the anchor to every other node, AT this time
    c = coords[frame]
    dist = np.linalg.norm(c[others] - c[a], axis=1)

    sel = dist <= DMAX if DMAX > 0 else slice(None)
    print(f"Plotting {len(others) if DMAX <= 0 else int(np.sum(dist <= DMAX))} points")

    out_png = (OUT if os.path.isabs(OUT) else os.path.join(root, OUT)) + ".png"
    out_npz = (OUT if os.path.isabs(OUT) else os.path.join(root, OUT)) + ".npz"
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    fig, (axz, axxy) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    # tiny jitter so the +-1/0 C^Z points don't overlap into solid lines
    jitter = (np.random.default_rng(0).random(len(others)) - 0.5) * 0.1
    axz.scatter(dist[sel], (cz + jitter)[sel], s=8, alpha=0.4, color="tab:blue")
    axxy.scatter(dist[sel], cxy[sel], s=8, alpha=0.4, color="tab:blue")

    axz.set_ylabel("C$^Z$ (sign, jittered)"); axz.set_ylim(-1.3, 1.3)
    axz.set_title("Z-axis correlation vs distance from anchor (up/down)")
    axz.axhline(0, color="gray", lw=0.5)
    axxy.set_ylabel("C$^{XY}$ (cosine)"); axxy.set_ylim(-1.1, 1.1)
    axxy.set_title("XY-plane correlation vs distance from anchor (in-plane)")
    axxy.set_xlabel("distance from anchor (nm, at that time)")
    axxy.axhline(0, color="gray", lw=0.5)
    fig.suptitle(f"Node {ANCHOR} vs all others — correlation vs distance @ {time[frame]:.0f} ps")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"Saved figure: {out_png}")

    np.savez(out_npz, anchor=ANCHOR, time_ps=time[frame],
             other_nodes=others + 1, distance=dist, C_Z=cz, C_XY=cxy)
    print(f"Saved data:   {out_npz}")


if __name__ == "__main__":
    main()
