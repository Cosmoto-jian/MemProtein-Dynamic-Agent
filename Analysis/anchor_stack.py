#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stacked anchor comparison (one row per anchor residue).

At a fixed time, draw one "distance vs correlation" panel per anchor residue,
stacked top-to-bottom so you can compare residue 1, residue 2, ... side by side.

Each panel: x = distance from that anchor to every other residue (nm, at the
chosen time); y = correlation of that residue's motion with the anchor. All
other residues are shown (no sampling).

Run directly (no command-line args):
    .venv/bin/python Analysis/anchor_heatmap.py
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from instant_correlation import load_trajectory, displacements, corr_z_pairs, corr_xy_pairs

# ============================================================
#  配置 —— 直接改这里(CONFIG: edit these)
# ============================================================
H5_FILE   = "simulation_data.h5"      # 仿真结果文件
ANCHORS   = list(range(1, 21))        # 要对照的锚点粒子(从上到下);改成 range(1,101) 看100个,或 [1,5,100,2000]
TIME_PS   = 50.0                      # 固定在哪个时刻(ps)
COMPONENT = "XY"                      # 画哪种相关:"XY"(平面) 或 "Z"(升降)
REF_FRAME = 40                        # t=0 参考帧
DMAX      = 0.0                       # 只看多少 nm 以内(0 = 全部)
OUT       = "Analysis/anchor_stack"   # 输出前缀(.png 和 .npz)
# ============================================================


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    h5_path = H5_FILE if os.path.isabs(H5_FILE) else os.path.join(root, H5_FILE)

    coords, time, _ = load_trajectory(h5_path)
    n_nodes = coords.shape[1]
    frame = int(np.argmin(np.abs(time - TIME_PS)))
    print(f"{n_nodes} nodes, {len(ANCHORS)} anchors, time = {time[frame]:.1f} ps, component = {COMPONENT}")

    dx, dy, dz = displacements(coords, ref=REF_FRAME)
    dxf, dyf, dzf = dx[frame:frame + 1], dy[frame:frame + 1], dz[frame:frame + 1]
    cframe = coords[frame]

    n = len(ANCHORS)
    fig, axes = plt.subplots(n, 1, sharex=True, figsize=(10, max(2.0, 1.3 * n)))
    if n == 1:
        axes = [axes]

    saved = {}
    for ax, anchor in zip(axes, ANCHORS):
        a = anchor - 1
        others = np.array([j for j in range(n_nodes) if j != a])
        pairs = np.stack([np.full(len(others), a), others], axis=1)
        dist = np.linalg.norm(cframe[others] - cframe[a], axis=1)

        if COMPONENT.upper() == "Z":
            corr = corr_z_pairs(dzf, pairs)[0]
            jitter = (np.random.default_rng(anchor).random(len(others)) - 0.5) * 0.1
            yvals = corr + jitter
        else:
            corr = corr_xy_pairs(dxf, dyf, pairs)[0]
            yvals = corr

        sel = dist <= DMAX if DMAX > 0 else slice(None)
        ax.scatter(dist[sel], yvals[sel], s=4, alpha=0.3, color="tab:blue")
        ax.axhline(0, color="gray", lw=0.4)
        ax.set_ylim(-1.2, 1.2)
        ax.set_ylabel(f"#{anchor}", rotation=0, labelpad=18, va="center")
        saved[f"anchor_{anchor}_dist"] = dist
        saved[f"anchor_{anchor}_corr"] = corr

    comp_name = "C$^Z$ (up/down)" if COMPONENT.upper() == "Z" else "C$^{XY}$ (in-plane)"
    axes[-1].set_xlabel("distance from anchor (nm, at that time)")
    fig.suptitle(f"{comp_name} vs distance, per anchor @ {time[frame]:.0f} ps  "
                 f"(y of each row = correlation; row label = anchor node)")
    fig.tight_layout()

    out_png = (OUT if os.path.isabs(OUT) else os.path.join(root, OUT)) + ".png"
    out_npz = (OUT if os.path.isabs(OUT) else os.path.join(root, OUT)) + ".npz"
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=120)
    print(f"Saved figure: {out_png}")
    np.savez(out_npz, anchors=np.array(ANCHORS), time_ps=time[frame], **saved)
    print(f"Saved data:   {out_npz}")


if __name__ == "__main__":
    main()
