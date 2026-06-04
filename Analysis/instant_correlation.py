#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Instantaneous correlation analysis of a deformation trajectory.

Implements the "single-step instantaneous correlation" methodology: for each
output frame t, measure how coordinated two residues' motions are, relative to
the t=0 reference structure.

    displacement   Dx_i(t) = x_i(t) - x_i(0)   (same for Dy, Dz)

    Z-axis correlation (1-D up/down coordination):
        C^Z_ij(t)  = sign( Dz_i(t) * Dz_j(t) )           in {-1, 0, +1}

    XY-plane correlation (2-D in-plane coordination):
        C^XY_ij(t) = (Dx_i*Dx_j + Dy_i*Dy_j)
                     / ( |D_xy,i| * |D_xy,j| )            in [-1, +1]
                   = cosine of the angle between the two in-plane
                     displacement vectors

Outputs per residue pair (i, j) two time series C^Z_ij(t) and C^XY_ij(t).
A sliding-window average is available to suppress thermal high-frequency noise.

This module is the post-processing stage of the pipeline; it consumes the
simulation_data.h5 produced by main_fast.py.
"""

import argparse
import os
from typing import Tuple

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")  # headless: save figures without a GUI window
import matplotlib.pyplot as plt


def load_trajectory(h5_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the node trajectory from an HDF5 result file.

    Returns:
        coords (T, N, 3) in nm
        time   (T,)      in ps
        target (M,)      1-based ids of the force-loaded (membrane) nodes
    """
    with h5py.File(h5_path, "r") as f:
        coords = f["timeseries/node_coords"][:] * 1e9  # m -> nm
        time = f["timeseries/time"][:] * 1e12          # s -> ps
        target = f["target_nodes"][:].astype(int)
    return coords, time, target


def displacements(coords: np.ndarray, ref: int = 0
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame displacement components relative to frame `ref`.

    Returns dx, dy, dz, each of shape (T, N).
    """
    d = coords - coords[ref]
    return d[:, :, 0], d[:, :, 1], d[:, :, 2]


def corr_z_pairs(dz: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """C^Z_ij(t) for each pair: sign of the product of the two z-displacements.

    pairs: (P, 2) array of 0-based node indices.
    Returns (T, P) array of values in {-1, 0, +1}; 0 when either displacement
    is ~0 (sign undefined).
    """
    a = dz[:, pairs[:, 0]]
    b = dz[:, pairs[:, 1]]
    out = np.sign(a * b)
    out[(np.abs(a) < 1e-9) | (np.abs(b) < 1e-9)] = 0.0
    return out


def corr_xy_pairs(dx: np.ndarray, dy: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """C^XY_ij(t) for each pair: cosine between in-plane displacement vectors.

    Returns (T, P) array in [-1, +1]; 0 when either in-plane displacement has
    ~0 magnitude (angle undefined).
    """
    ax, ay = dx[:, pairs[:, 0]], dy[:, pairs[:, 0]]
    bx, by = dx[:, pairs[:, 1]], dy[:, pairs[:, 1]]
    num = ax * bx + ay * by
    den = np.sqrt(ax**2 + ay**2) * np.sqrt(bx**2 + by**2)
    out = np.zeros_like(num)
    m = den > 1e-12
    out[m] = num[m] / den[m]
    return out


def sliding_window(series: np.ndarray, w: int) -> np.ndarray:
    """Centered moving average along the time axis (axis 0).

    w is the window size in frames; w <= 1 returns the series unchanged.
    This is the low-pass denoising of section 6 of the methodology.
    """
    if w <= 1:
        return series
    kernel = np.ones(w) / w
    return np.apply_along_axis(lambda s: np.convolve(s, kernel, mode="same"), 0, series)


def delta_corr(series: np.ndarray) -> np.ndarray:
    """Frame-to-frame change |C(t) - C(t-1)|; large values flag conformational
    transition points (section 5.2). Returns (T, P) with row 0 = 0."""
    out = np.zeros_like(series)
    out[1:] = np.abs(np.diff(series, axis=0))
    return out


def select_demo_pairs(coords0: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Pick representative pairs spanning a range of inter-residue distances.

    Anchors on the first force-loaded node and pairs it with partners at the
    10/30/50/70/90th distance percentiles, so the output shows how coordination
    changes with separation. Returns (P, 2) array of 0-based node indices.
    """
    tgt0 = target - 1  # to 0-based
    anchor = tgt0[0]
    others = tgt0[1:]
    dist = np.linalg.norm(coords0[others] - coords0[anchor], axis=1)
    order = np.argsort(dist)
    pcts = [10, 30, 50, 70, 90]
    picks = [others[order[int(p / 100 * (len(order) - 1))]] for p in pcts]
    return np.array([[anchor, p] for p in picks], dtype=int)


def main() -> None:
    ap = argparse.ArgumentParser(description="Instantaneous correlation analysis of a trajectory.")
    ap.add_argument("--h5", default="simulation_data.h5", help="Simulation HDF5 file")
    ap.add_argument("--pairs", default=None,
                    help="Comma-separated 1-based node pairs 'i-j,i-j'; default = auto demo pairs")
    ap.add_argument("--ref", type=int, default=0, help="Reference frame index (t=0)")
    ap.add_argument("--window", type=int, default=1, help="Sliding-window size (frames) for denoising")
    ap.add_argument("--out", default="Analysis/instant_corr", help="Output prefix (.png and .npz)")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    h5_path = args.h5 if os.path.isabs(args.h5) else os.path.join(root, args.h5)

    coords, time, target = load_trajectory(h5_path)
    print(f"Loaded trajectory: {coords.shape[0]} frames, {coords.shape[1]} nodes, "
          f"{time[0]:.1f}-{time[-1]:.1f} ps")

    dx, dy, dz = displacements(coords, ref=args.ref)
    coords0 = coords[args.ref]

    if args.pairs:
        pairs = np.array([[int(a) - 1, int(b) - 1]
                          for a, b in (p.split("-") for p in args.pairs.split(","))], dtype=int)
    else:
        pairs = select_demo_pairs(coords0, target)
        print("Auto-selected demo pairs (1-based node ids, by distance):")
        for i, j in pairs:
            d = np.linalg.norm(coords0[i] - coords0[j])
            print(f"  {i + 1:>5d} - {j + 1:<5d}  (distance {d:.1f} nm)")

    cz = corr_z_pairs(dz, pairs)
    cxy = corr_xy_pairs(dx, dy, pairs)
    if args.window > 1:
        cz = sliding_window(cz, args.window)
        cxy = sliding_window(cxy, args.window)

    # ---- plot ----
    out_png = (args.out if os.path.isabs(args.out) else os.path.join(root, args.out)) + ".png"
    out_npz = (args.out if os.path.isabs(args.out) else os.path.join(root, args.out)) + ".npz"
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    fig, (axz, axxy) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for k in range(pairs.shape[0]):
        d = np.linalg.norm(coords0[pairs[k, 0]] - coords0[pairs[k, 1]])
        label = f"{pairs[k,0]+1}-{pairs[k,1]+1} ({d:.1f} nm)"
        axz.plot(time, cz[:, k], lw=1.2, label=label)
        axxy.plot(time, cxy[:, k], lw=1.2, label=label)
    axz.set_ylabel("C$^Z$ (sign)"); axz.set_ylim(-1.2, 1.2)
    axz.set_title("Instantaneous Z-axis correlation (up/down coordination)")
    axz.axhline(0, color="gray", lw=0.5); axz.legend(fontsize=8, ncol=2)
    axxy.set_ylabel("C$^{XY}$ (cosine)"); axxy.set_ylim(-1.2, 1.2)
    axxy.set_title("Instantaneous XY-plane correlation (in-plane coordination)")
    axxy.set_xlabel("time (ps)"); axxy.axhline(0, color="gray", lw=0.5)
    win_note = f" (sliding window = {args.window} frames)" if args.window > 1 else ""
    fig.suptitle(f"Instantaneous correlation analysis{win_note}")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"Saved figure: {out_png}")

    np.savez(out_npz, time=time, pairs=pairs + 1, C_Z=cz, C_XY=cxy)
    print(f"Saved data:   {out_npz}")


if __name__ == "__main__":
    main()
