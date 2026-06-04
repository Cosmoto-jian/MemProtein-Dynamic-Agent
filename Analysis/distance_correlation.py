#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Distance-dependent instantaneous correlation (methodology section 7).

For each frame t and each inter-residue distance bin d, average the
instantaneous correlation over all residue pairs whose t=0 separation falls in
that bin:

    Cbar^XY(d, t) = mean over pairs(i,j) in bin d of  C^XY_ij(t)
    Cbar^Z (d, t) = mean over pairs(i,j) in bin d of  C^Z_ij(t)

This reveals how coordination at different spatial scales evolves in time, and
whether long-range correlations appear/switch (allosteric signal propagation).

Because the full pair set is ~N^2/2 (~10 million for 4554 nodes), you can either
sample pairs (fast, slight statistical error) or use --exact (all pairs, no
error) paired with a small number of time steps (--n-times). Pairs are processed
in chunks so memory stays bounded regardless of pair count.
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from instant_correlation import load_trajectory, displacements, corr_z_pairs, corr_xy_pairs


def select_frames(T: int, n_times: int) -> np.ndarray:
    """Pick `n_times` evenly spaced frame indices in [0, T-1]; 0 = all frames."""
    if n_times <= 0 or n_times >= T:
        return np.arange(T)
    return np.unique(np.linspace(0, T - 1, n_times).astype(int))


def distance_binned(coords: np.ndarray, ref: int, frames: np.ndarray, bin_width: float,
                    dmax: float, chunk: int, n_sample: int, seed: int, exact: bool):
    """Return (bin_edges, Cz_db, Cxy_db, counts).

    Cz_db / Cxy_db have shape (n_bins, F) where F = len(frames). Computation is
    chunked over residue pairs to bound memory.

    If `exact` is True, ALL N*(N-1)/2 pairs are used (no sampling error); the
    cost scales with the number of frames, which is why this pairs naturally
    with a small `frames` set. Otherwise `n_sample` random pairs are used.
    """
    N = coords.shape[1]
    coords0 = coords[ref]

    if exact:
        i, j = np.triu_indices(N, k=1)        # every unordered pair once
    else:
        rng = np.random.default_rng(seed)
        i = rng.integers(0, N, n_sample)
        j = rng.integers(0, N, n_sample)
        keep = i != j
        i, j = i[keep], j[keep]

    d = np.linalg.norm(coords0[i] - coords0[j], axis=1)
    within = d <= dmax
    i, j, d = i[within], j[within], d[within]

    # displacements relative to the reference frame, only for the chosen frames
    disp = coords[frames] - coords0           # (F, N, 3)
    dx, dy, dz = disp[:, :, 0], disp[:, :, 1], disp[:, :, 2]
    T = len(frames)

    edges = np.arange(0.0, dmax + bin_width, bin_width)
    n_bins = len(edges) - 1
    bin_idx = np.clip(np.digitize(d, edges) - 1, 0, n_bins - 1)

    sum_z = np.zeros((n_bins, T))
    sum_xy = np.zeros((n_bins, T))
    counts = np.zeros(n_bins, dtype=int)

    n_pairs = len(i)
    for s in range(0, n_pairs, chunk):
        e = min(s + chunk, n_pairs)
        pairs_c = np.stack([i[s:e], j[s:e]], axis=1)
        cz_c = corr_z_pairs(dz, pairs_c)          # (T, nc)
        cxy_c = corr_xy_pairs(dx, dy, pairs_c)    # (T, nc)
        bidx_c = bin_idx[s:e]
        for b in range(n_bins):
            m = bidx_c == b
            if m.any():
                sum_z[b] += cz_c[:, m].sum(axis=1)
                sum_xy[b] += cxy_c[:, m].sum(axis=1)
                counts[b] += int(m.sum())

    with np.errstate(invalid="ignore"):
        Cz_db = np.where(counts[:, None] > 0, sum_z / counts[:, None], np.nan)
        Cxy_db = np.where(counts[:, None] > 0, sum_xy / counts[:, None], np.nan)
    return edges, Cz_db, Cxy_db, counts


def main() -> None:
    ap = argparse.ArgumentParser(description="Distance-binned instantaneous correlation heatmaps.")
    ap.add_argument("--h5", default="simulation_data.h5")
    ap.add_argument("--ref", type=int, default=0, help="Reference frame (t=0)")
    ap.add_argument("--n-times", type=int, default=10,
                    help="Number of evenly spaced time steps to evaluate (0 = all frames)")
    ap.add_argument("--exact", action="store_true",
                    help="Use ALL residue pairs (no sampling); pair this with a small --n-times")
    ap.add_argument("--n-sample", type=int, default=300000,
                    help="Number of residue pairs to sample when not using --exact")
    ap.add_argument("--bin-width", type=float, default=1.0, help="Distance bin width (nm)")
    ap.add_argument("--dmax", type=float, default=0.0, help="Max distance (nm); 0 = auto (98th pct)")
    ap.add_argument("--min-count", type=int, default=50, help="Mask bins with fewer sampled pairs")
    ap.add_argument("--chunk", type=int, default=25000, help="Pairs per processing chunk")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="Analysis/distance_corr")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    h5_path = args.h5 if os.path.isabs(args.h5) else os.path.join(root, args.h5)

    coords, time, _ = load_trajectory(h5_path)
    print(f"Loaded trajectory: {coords.shape[0]} frames, {coords.shape[1]} nodes")

    dmax = args.dmax
    if dmax <= 0:
        rng = np.random.default_rng(args.seed)
        N = coords.shape[1]
        a = rng.integers(0, N, 50000)
        b = rng.integers(0, N, 50000)
        dd = np.linalg.norm(coords[args.ref][a] - coords[args.ref][b], axis=1)
        dmax = float(np.percentile(dd, 98))
        print(f"Auto dmax = {dmax:.1f} nm (98th percentile of pair distances)")

    frames = select_frames(coords.shape[0], args.n_times)
    sel_time = time[frames]
    mode = "exact (all pairs)" if args.exact else f"sampled ({args.n_sample} pairs)"
    print(f"Evaluating {len(frames)} time steps at {sel_time[0]:.1f}-{sel_time[-1]:.1f} ps; "
          f"mode = {mode}")

    edges, Cz, Cxy, counts = distance_binned(
        coords, args.ref, frames, args.bin_width, dmax, args.chunk,
        args.n_sample, args.seed, args.exact)
    centers = 0.5 * (edges[:-1] + edges[1:])
    print(f"{len(centers)} distance bins; pairs per bin: "
          f"min={counts.min()}, max={counts.max()}")

    # mask low-count bins
    low = counts < args.min_count
    Cz[low] = np.nan
    Cxy[low] = np.nan

    out_png = (args.out if os.path.isabs(args.out) else os.path.join(root, args.out)) + ".png"
    out_npz = (args.out if os.path.isabs(args.out) else os.path.join(root, args.out)) + ".npz"
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    fig, (az, axy) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    extent = [sel_time[0], sel_time[-1], edges[0], edges[-1]]
    for ax, data, title in ((az, Cz, "Cbar$^Z$(d,t)  up/down"),
                            (axy, Cxy, "Cbar$^{XY}$(d,t)  in-plane")):
        im = ax.imshow(data, aspect="auto", origin="lower", extent=extent,
                       cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
        ax.set_xlabel("time (ps)")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label="mean correlation", fraction=0.046, pad=0.04)
    az.set_ylabel("inter-residue distance d (nm)")
    fig.suptitle("Distance-dependent instantaneous correlation")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"Saved figure: {out_png}")

    np.savez(out_npz, time=sel_time, bin_edges=edges, bin_centers=centers,
             C_Z=Cz, C_XY=Cxy, counts=counts)
    print(f"Saved data:   {out_npz}")


if __name__ == "__main__":
    main()
