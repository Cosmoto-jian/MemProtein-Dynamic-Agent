#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Instantaneous correlation vs inter-residue distance.

For a chosen time (or several), measure how coordinated each residue pair's
motion is, relative to the t=0 reference structure, and plot it against the
pair's reference separation distance:

    displacement   Dx_i(t) = x_i(t) - x_i(0)   (same for Dy, Dz)

    Z-axis correlation (1-D up/down coordination):
        C^Z_ij(t)  = sign( Dz_i(t) * Dz_j(t) )           in {-1, 0, +1}

    XY-plane correlation (2-D in-plane coordination):
        C^XY_ij(t) = (Dx_i*Dx_j + Dy_i*Dy_j)
                     / ( |D_xy,i| * |D_xy,j| )            in [-1, +1]

The figure has two panels (C^Z and C^XY); in both the x-axis is the
inter-residue distance d_ij (nm at t=0) and each point is one residue pair,
coloured by the time at which it was evaluated. This shows how coordination
decays / switches with separation, and how that changes over time.

This module is the post-processing stage; it consumes simulation_data.h5.
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
    """Load the node trajectory. Returns coords (T,N,3) nm, time (T,) ps,
    and the 1-based ids of the force-loaded (membrane) nodes."""
    with h5py.File(h5_path, "r") as f:
        coords = f["timeseries/node_coords"][:] * 1e9  # m -> nm
        time = f["timeseries/time"][:] * 1e12          # s -> ps
        target = f["target_nodes"][:].astype(int)
    return coords, time, target


def superpose(coords: np.ndarray, ref: int = 0) -> np.ndarray:
    """Rigid-body align every frame onto frame `ref` (Kabsch / SVD): remove the
    overall translation and rotation. Returns aligned coords (T, N, 3).

    Inter-residue distances are rigid-body invariants, so they are unchanged;
    only the displacement field (and thus the correlations) is affected. This
    stops whole-protein drift/tumbling from leaking in as spurious coordination.
    """
    Q = coords[ref]
    qc = Q.mean(axis=0)
    Q0 = Q - qc
    out = np.empty_like(coords)
    for i in range(coords.shape[0]):
        P0 = coords[i] - coords[i].mean(axis=0)
        H = P0.T @ Q0
        U, S, Vt = np.linalg.svd(H)
        d = np.sign(np.linalg.det(Vt.T @ U.T))     # reflection correction
        R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T     # rotation mapping P0 -> Q0
        out[i] = P0 @ R.T + qc
    return out


def displacements(coords: np.ndarray, ref: int = 0, align: bool = True
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame displacement components relative to frame `ref`; (T, N) each.

    With align=True (default) each frame is first rigid-body superposed onto the
    reference, so overall translation/rotation do not contaminate the
    correlations. Set align=False to use raw coordinates.
    """
    if align:
        coords = superpose(coords, ref)
    d = coords - coords[ref]
    return d[:, :, 0], d[:, :, 1], d[:, :, 2]


def corr_z_pairs(dz: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """C^Z for each pair: sign of the product of the two z-displacements.
    Returns (T, P) in {-1, 0, +1}; 0 when either displacement is ~0."""
    a = dz[:, pairs[:, 0]]
    b = dz[:, pairs[:, 1]]
    out = np.sign(a * b)
    out[(np.abs(a) < 1e-9) | (np.abs(b) < 1e-9)] = 0.0
    return out


def corr_xy_pairs(dx: np.ndarray, dy: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """C^XY for each pair: cosine between in-plane displacement vectors.
    Returns (T, P) in [-1, +1]; 0 when either in-plane displacement is ~0."""
    ax, ay = dx[:, pairs[:, 0]], dy[:, pairs[:, 0]]
    bx, by = dx[:, pairs[:, 1]], dy[:, pairs[:, 1]]
    num = ax * bx + ay * by
    den = np.sqrt(ax**2 + ay**2) * np.sqrt(bx**2 + by**2)
    out = np.zeros_like(num)
    m = den > 1e-12
    out[m] = num[m] / den[m]
    return out


def sliding_window(series: np.ndarray, w: int) -> np.ndarray:
    """Centered moving average along the time axis (axis 0); w<=1 is a no-op."""
    if w <= 1:
        return series
    kernel = np.ones(w) / w
    return np.apply_along_axis(lambda s: np.convolve(s, kernel, mode="same"), 0, series)


def delta_corr(series: np.ndarray) -> np.ndarray:
    """Frame-to-frame change |C(t) - C(t-1)|; flags transition points."""
    out = np.zeros_like(series)
    out[1:] = np.abs(np.diff(series, axis=0))
    return out


def sample_pairs(n_nodes: int, n_sample: int, seed: int) -> np.ndarray:
    """Random distinct residue pairs (P, 2) as 0-based node indices."""
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n_nodes, n_sample)
    j = rng.integers(0, n_nodes, n_sample)
    keep = i != j
    return np.stack([i[keep], j[keep]], axis=1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Instantaneous correlation vs inter-residue distance.")
    ap.add_argument("--h5", default="simulation_data.h5", help="Simulation HDF5 file")
    ap.add_argument("--times", default="50",
                    help="Comma-separated times (ps) at which to evaluate, e.g. '10,50,100'")
    ap.add_argument("--ref", type=int, default=0, help="Reference frame index (t=0)")
    ap.add_argument("--n-sample", type=int, default=8000, help="Number of residue pairs to plot")
    ap.add_argument("--dmax", type=float, default=0.0, help="Max distance to show (nm); 0 = all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="Analysis/instant_corr", help="Output prefix (.png and .npz)")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    h5_path = args.h5 if os.path.isabs(args.h5) else os.path.join(root, args.h5)

    coords, time, _ = load_trajectory(h5_path)
    print(f"Loaded trajectory: {coords.shape[0]} frames, {coords.shape[1]} nodes, "
          f"{time[0]:.1f}-{time[-1]:.1f} ps")

    # requested times -> nearest frame indices
    want = [float(t) for t in args.times.split(",")]
    frames = [int(np.argmin(np.abs(time - t))) for t in want]
    print("Evaluating at times (ps): " + ", ".join(f"{time[f]:.1f}" for f in frames))

    pairs = sample_pairs(coords.shape[1], args.n_sample, args.seed)
    print(f"{len(pairs)} residue pairs sampled (of {coords.shape[1]} nodes; "
          f"plotting all of them would be ~{coords.shape[1]*(coords.shape[1]-1)//2//10**6}M pairs)")

    dx, dy, dz = displacements(coords, ref=args.ref)
    cz = corr_z_pairs(dz, pairs)        # (T, P)
    cxy = corr_xy_pairs(dx, dy, pairs)  # (T, P)

    def dist_at(frame: int) -> np.ndarray:
        """Actual inter-residue distance of each pair AT this frame (nm)."""
        c = coords[frame]
        return np.linalg.norm(c[pairs[:, 0]] - c[pairs[:, 1]], axis=1)

    # ---- plot: correlation vs inter-residue distance ----
    out_png = (args.out if os.path.isabs(args.out) else os.path.join(root, args.out)) + ".png"
    out_npz = (args.out if os.path.isabs(args.out) else os.path.join(root, args.out)) + ".npz"
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    cmap = plt.get_cmap("viridis")
    colors = [cmap(k / max(1, len(frames) - 1)) for k in range(len(frames))]

    dist_per_frame = []
    fig, (axz, axxy) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for k, fr in enumerate(frames):
        lab = f"{time[fr]:.0f} ps"
        d_fr = dist_at(fr)                 # distance AT this time
        dist_per_frame.append(d_fr)
        sel = d_fr <= args.dmax if args.dmax > 0 else slice(None)
        # small vertical jitter for C^Z (values are only -1/0/+1) so density shows
        jitter = (np.random.default_rng(args.seed + k).random(len(pairs)) - 0.5) * 0.12
        axz.scatter(d_fr[sel], (cz[fr] + jitter)[sel], s=4, alpha=0.25, color=colors[k], label=lab)
        axxy.scatter(d_fr[sel], cxy[fr][sel], s=4, alpha=0.25, color=colors[k], label=lab)
    axz.set_ylabel("C$^Z$ (sign, jittered)"); axz.set_ylim(-1.3, 1.3)
    axz.set_title("Z-axis correlation vs inter-residue distance (up/down)")
    axz.axhline(0, color="gray", lw=0.5)
    leg = axz.legend(title="time", markerscale=3, fontsize=9);
    for lh in leg.legend_handles: lh.set_alpha(1)
    axxy.set_ylabel("C$^{XY}$ (cosine)"); axxy.set_ylim(-1.1, 1.1)
    axxy.set_title("XY-plane correlation vs inter-residue distance (in-plane)")
    axxy.set_xlabel("inter-residue distance d (nm, at that time)")
    axxy.axhline(0, color="gray", lw=0.5)
    fig.suptitle("Instantaneous correlation vs distance")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"Saved figure: {out_png}")

    np.savez(out_npz, times_ps=np.array([time[f] for f in frames]),
             distance_per_time=np.array(dist_per_frame), pairs=pairs + 1,
             C_Z=cz[frames], C_XY=cxy[frames])
    print(f"Saved data:   {out_npz}")


if __name__ == "__main__":
    main()
