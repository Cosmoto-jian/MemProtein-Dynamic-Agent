#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Instantaneous correlation analysis of a deformation trajectory.

Core idea: at each frame, after rigid-body aligning to the t=0 reference, measure
how coordinated two residues' displacements are.
    C^Z  = sign(Dz_i * Dz_j)                          in {-1, 0, +1}
    C^XY = cosine between in-plane displacement vectors in [-1, +1]

Four ready-made figures (each saves <out>.png and <out>.npz):
    correlation_vs_distance  scatter of C vs distance, for sampled pairs / times
    distance_heatmap         mean C binned by distance, over time (heatmap)
    anchor_scatter           one anchor vs every other residue, at one time
    anchor_stack             one panel per anchor, stacked for comparison

All take align=True by default (Kabsch superposition) so whole-protein
translation/rotation does not leak in as spurious coordination.
"""

import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
#  core
# --------------------------------------------------------------------------- #
def load_trajectory(h5_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns coords (T,N,3) nm, time (T,) ps, target node ids (1-based)."""
    with h5py.File(h5_path, "r") as f:
        coords = f["timeseries/node_coords"][:] * 1e9
        time = f["timeseries/time"][:] * 1e12
        target = f["target_nodes"][:].astype(int)
    return coords, time, target


def superpose(coords: np.ndarray, ref: int = 0) -> np.ndarray:
    """Rigid-body align every frame onto frame `ref` (Kabsch/SVD). Distances are
    unchanged; only the displacement field is affected."""
    Q0 = coords[ref] - coords[ref].mean(axis=0)
    qc = coords[ref].mean(axis=0)
    out = np.empty_like(coords)
    for i in range(coords.shape[0]):
        P0 = coords[i] - coords[i].mean(axis=0)
        U, _, Vt = np.linalg.svd(P0.T @ Q0)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
        out[i] = P0 @ R.T + qc
    return out


def displacements(coords: np.ndarray, ref: int = 0, align: bool = True
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Displacement components relative to frame `ref`; (T,N) each. With
    align=True each frame is first rigid-body superposed onto the reference."""
    if align:
        coords = superpose(coords, ref)
    d = coords - coords[ref]
    return d[:, :, 0], d[:, :, 1], d[:, :, 2]


def corr_z_pairs(dz: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    a, b = dz[:, pairs[:, 0]], dz[:, pairs[:, 1]]
    out = np.sign(a * b)
    out[(np.abs(a) < 1e-9) | (np.abs(b) < 1e-9)] = 0.0
    return out


def corr_xy_pairs(dx: np.ndarray, dy: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    ax, ay = dx[:, pairs[:, 0]], dy[:, pairs[:, 0]]
    bx, by = dx[:, pairs[:, 1]], dy[:, pairs[:, 1]]
    num = ax * bx + ay * by
    den = np.sqrt(ax**2 + ay**2) * np.sqrt(bx**2 + by**2)
    out = np.zeros_like(num)
    m = den > 1e-12
    out[m] = num[m] / den[m]
    return out


def sample_pairs(n_nodes: int, n_sample: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n_nodes, n_sample)
    j = rng.integers(0, n_nodes, n_sample)
    keep = i != j
    return np.stack([i[keep], j[keep]], axis=1)


def _save(out: str, fig, **npz):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out + ".png", dpi=130)
    plt.close(fig)
    np.savez(out + ".npz", **npz)
    return out + ".png"


# --------------------------------------------------------------------------- #
#  figures
# --------------------------------------------------------------------------- #
def correlation_vs_distance(h5_path: str, times: Sequence[float] = (10, 50, 100),
                            n_sample: int = 8000, ref: int = 0, dmax: float = 0.0,
                            seed: int = 0, align: bool = True,
                            out: str = "data/results/instant_corr") -> str:
    """Scatter of C^Z / C^XY vs inter-residue distance (at that time), for
    several times overlaid."""
    coords, time, _ = load_trajectory(h5_path)
    frames = [int(np.argmin(np.abs(time - t))) for t in times]
    pairs = sample_pairs(coords.shape[1], n_sample, seed)
    dx, dy, dz = displacements(coords, ref, align)
    cz, cxy = corr_z_pairs(dz, pairs), corr_xy_pairs(dx, dy, pairs)

    cmap = plt.get_cmap("viridis")
    fig, (az, axy) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    dist_per = []
    for k, fr in enumerate(frames):
        c = coords[fr]
        d = np.linalg.norm(c[pairs[:, 0]] - c[pairs[:, 1]], axis=1)
        dist_per.append(d)
        sel = d <= dmax if dmax > 0 else slice(None)
        col = cmap(k / max(1, len(frames) - 1))
        jit = (np.random.default_rng(seed + k).random(len(pairs)) - 0.5) * 0.12
        az.scatter(d[sel], (cz[fr] + jit)[sel], s=4, alpha=0.25, color=col, label=f"{time[fr]:.0f} ps")
        axy.scatter(d[sel], cxy[fr][sel], s=4, alpha=0.25, color=col, label=f"{time[fr]:.0f} ps")
    az.set_ylabel("C$^Z$ (sign)"); az.set_ylim(-1.3, 1.3); az.axhline(0, color="gray", lw=0.5)
    az.set_title("Z-axis correlation vs inter-residue distance")
    leg = az.legend(title="time", markerscale=3, fontsize=9)
    for lh in leg.legend_handles:
        lh.set_alpha(1)
    axy.set_ylabel("C$^{XY}$ (cosine)"); axy.set_ylim(-1.1, 1.1); axy.axhline(0, color="gray", lw=0.5)
    axy.set_title("XY-plane correlation vs inter-residue distance")
    axy.set_xlabel("inter-residue distance d (nm, at that time)")
    fig.suptitle("Instantaneous correlation vs distance")
    fig.tight_layout()
    return _save(out, fig, times_ps=np.array([time[f] for f in frames]),
                 distance_per_time=np.array(dist_per), pairs=pairs + 1,
                 C_Z=cz[frames], C_XY=cxy[frames])


def distance_heatmap(h5_path: str, n_times: int = 10, exact: bool = True,
                     n_sample: int = 300000, bin_width: float = 1.0,
                     dmax: float = 0.0, min_count: int = 50, chunk: int = 25000,
                     ref: int = 0, seed: int = 0, align: bool = True,
                     out: str = "data/results/distance_corr") -> str:
    """Mean correlation binned by distance, as a time x distance heatmap."""
    coords, time, _ = load_trajectory(h5_path)
    T, N = coords.shape[0], coords.shape[1]
    if dmax <= 0:
        rng = np.random.default_rng(seed)
        a, b = rng.integers(0, N, 50000), rng.integers(0, N, 50000)
        dmax = float(np.percentile(np.linalg.norm(coords[ref][a] - coords[ref][b], axis=1), 98))

    frames = np.arange(T) if n_times <= 0 or n_times >= T else np.unique(
        np.linspace(0, T - 1, n_times).astype(int))
    if exact:
        i, j = np.triu_indices(N, k=1)
    else:
        rng = np.random.default_rng(seed)
        i, j = rng.integers(0, N, n_sample), rng.integers(0, N, n_sample)
        keep = i != j
        i, j = i[keep], j[keep]
    d = np.linalg.norm(coords[ref][i] - coords[ref][j], axis=1)
    within = d <= dmax
    i, j, d = i[within], j[within], d[within]

    dx, dy, dz = displacements(coords, ref, align)
    dx, dy, dz = dx[frames], dy[frames], dz[frames]
    edges = np.arange(0.0, dmax + bin_width, bin_width)
    nb = len(edges) - 1
    bidx = np.clip(np.digitize(d, edges) - 1, 0, nb - 1)
    sz = np.zeros((nb, len(frames))); sxy = np.zeros((nb, len(frames))); cnt = np.zeros(nb, int)
    for s in range(0, len(i), chunk):
        e = min(s + chunk, len(i))
        pc = np.stack([i[s:e], j[s:e]], axis=1)
        cz, cxy = corr_z_pairs(dz, pc), corr_xy_pairs(dx, dy, pc)
        bc = bidx[s:e]
        for b in range(nb):
            m = bc == b
            if m.any():
                sz[b] += cz[:, m].sum(1); sxy[b] += cxy[:, m].sum(1); cnt[b] += int(m.sum())
    with np.errstate(invalid="ignore"):
        Cz = np.where(cnt[:, None] > 0, sz / cnt[:, None], np.nan)
        Cxy = np.where(cnt[:, None] > 0, sxy / cnt[:, None], np.nan)
    Cz[cnt < min_count] = np.nan; Cxy[cnt < min_count] = np.nan

    t_sel = time[frames]
    fig, (az, axy) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    ext = [t_sel[0], t_sel[-1], edges[0], edges[-1]]
    for ax, data, title in ((az, Cz, "Cbar$^Z$(d,t)"), (axy, Cxy, "Cbar$^{XY}$(d,t)")):
        im = ax.imshow(data, aspect="auto", origin="lower", extent=ext, cmap="RdBu_r",
                       vmin=-1, vmax=1, interpolation="nearest")
        ax.set_xlabel("time (ps)"); ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    az.set_ylabel("inter-residue distance d (nm)")
    fig.suptitle("Distance-dependent instantaneous correlation")
    fig.tight_layout()
    return _save(out, fig, time=t_sel, bin_edges=edges, C_Z=Cz, C_XY=Cxy, counts=cnt)


def anchor_scatter(h5_path: str, anchor: int = 1, time_ps: float = 50.0,
                   ref: int = 0, dmax: float = 0.0, align: bool = True,
                   out: str = "data/results/anchor_corr") -> str:
    """One anchor residue vs every other residue (no sampling), at one time."""
    coords, time, _ = load_trajectory(h5_path)
    N = coords.shape[1]
    fr = int(np.argmin(np.abs(time - time_ps)))
    a = anchor - 1
    others = np.array([k for k in range(N) if k != a])
    pairs = np.stack([np.full(len(others), a), others], axis=1)
    dx, dy, dz = displacements(coords, ref, align)
    cz = corr_z_pairs(dz, pairs)[fr]
    cxy = corr_xy_pairs(dx, dy, pairs)[fr]
    c = coords[fr]
    dist = np.linalg.norm(c[others] - c[a], axis=1)
    sel = dist <= dmax if dmax > 0 else slice(None)

    fig, (az, axy) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    jit = (np.random.default_rng(0).random(len(others)) - 0.5) * 0.1
    az.scatter(dist[sel], (cz + jit)[sel], s=8, alpha=0.4, color="tab:blue")
    axy.scatter(dist[sel], cxy[sel], s=8, alpha=0.4, color="tab:blue")
    az.set_ylabel("C$^Z$ (sign)"); az.set_ylim(-1.3, 1.3); az.axhline(0, color="gray", lw=0.5)
    az.set_title("Z-axis correlation vs distance from anchor")
    axy.set_ylabel("C$^{XY}$ (cosine)"); axy.set_ylim(-1.1, 1.1); axy.axhline(0, color="gray", lw=0.5)
    axy.set_title("XY-plane correlation vs distance from anchor")
    axy.set_xlabel("distance from anchor (nm, at that time)")
    fig.suptitle(f"Node {anchor} vs all others @ {time[fr]:.0f} ps")
    fig.tight_layout()
    return _save(out, fig, anchor=anchor, time_ps=time[fr],
                 other_nodes=others + 1, distance=dist, C_Z=cz, C_XY=cxy)


def anchor_stack(h5_path: str, anchors: Sequence[int] = tuple(range(1, 21)),
                 time_ps: float = 50.0, component: str = "XY", ref: int = 0,
                 dmax: float = 0.0, align: bool = True,
                 out: str = "data/results/anchor_stack") -> str:
    """One distance-vs-correlation panel per anchor, stacked for comparison."""
    coords, time, _ = load_trajectory(h5_path)
    N = coords.shape[1]
    fr = int(np.argmin(np.abs(time - time_ps)))
    dx, dy, dz = displacements(coords, ref, align)
    dxf, dyf, dzf = dx[fr:fr + 1], dy[fr:fr + 1], dz[fr:fr + 1]
    cf = coords[fr]

    anchors = list(anchors)
    fig, axes = plt.subplots(len(anchors), 1, sharex=True, figsize=(10, max(2.0, 1.3 * len(anchors))))
    if len(anchors) == 1:
        axes = [axes]
    saved = {}
    for ax, anchor in zip(axes, anchors):
        a = anchor - 1
        others = np.array([k for k in range(N) if k != a])
        pairs = np.stack([np.full(len(others), a), others], axis=1)
        dist = np.linalg.norm(cf[others] - cf[a], axis=1)
        if component.upper() == "Z":
            corr = corr_z_pairs(dzf, pairs)[0]
            y = corr + (np.random.default_rng(anchor).random(len(others)) - 0.5) * 0.1
        else:
            corr = corr_xy_pairs(dxf, dyf, pairs)[0]
            y = corr
        sel = dist <= dmax if dmax > 0 else slice(None)
        ax.scatter(dist[sel], y[sel], s=4, alpha=0.3, color="tab:blue")
        ax.axhline(0, color="gray", lw=0.4); ax.set_ylim(-1.2, 1.2)
        ax.set_ylabel(f"#{anchor}", rotation=0, labelpad=18, va="center")
        saved[f"anchor_{anchor}_dist"] = dist
        saved[f"anchor_{anchor}_corr"] = corr
    name = "C^Z" if component.upper() == "Z" else "C^XY"
    axes[-1].set_xlabel("distance from anchor (nm, at that time)")
    fig.suptitle(f"{name} vs distance, per anchor @ {time[fr]:.0f} ps")
    fig.tight_layout()
    return _save(out, fig, anchors=np.array(anchors), time_ps=time[fr], **saved)
