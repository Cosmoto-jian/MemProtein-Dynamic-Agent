#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Vectorized vector-form finite-element (VFIFE) dynamics of a coarse-grained
elastic network under a triangular force load.

Physics (identical to the original main.py, just NumPy-vectorized):
  - central-difference explicit integration with viscous damping
  - per-element rate-form axial force; element fails (force -> 0) if it stretches
    beyond 2 nm
  - planar force load: force applied along the in-plane (XY) direction at the
    target nodes, ramped 0 -> Fmax -> 0 (triangle wave)

Units: pN / nm / ps. Output HDF5 stores coordinates in metres and time in
seconds (analysis converts back).
"""

import os
import time as _time
from typing import Optional

import numpy as np
import h5py

from .io import read_model_file, load_target_nodes, load_mass, load_evector


def triangle_load_factor(t: float, t0: float, t1: float, t2: float) -> float:
    """Pure triangular loading factor: 0 (t<=t0) -> 1 (t0..t1) -> 0 (t1..t2)."""
    if t <= t0:
        return 0.0
    if t <= t1:
        return 1.0 if t1 <= t0 else (t - t0) / (t1 - t0)
    if t2 <= t1:
        return 1.0
    if t <= t2:
        return 1.0 - (t - t1) / (t2 - t1)
    return 0.0


def run_simulation(model: str = "data/inputs/MODEL.txt",
                   target_nodes: str = "data/inputs/targetNode.txt",
                   mass: str = "data/inputs/mass.txt",
                   evector_mat: str = "data/inputs/evector.mat",
                   out: str = "data/results/simulation_data.h5",
                   ET: float = 100.0, h: float = 0.1, zeta: float = 0.01,
                   E: float = 1000.0, A: float = 0.01,
                   Fmax: float = 0.1, ramp_t0: float = 0.01,
                   ramp_t1: float = 50.0, unload_t1: float = 100.0,
                   save_internal: bool = False, force_save_stride: int = 1,
                   compress: int = 4, verbose: bool = True) -> dict:
    """Run the force-control simulation and write results to `out` (HDF5).

    Returns a summary dict (steps, final extension, final internal force, ...).
    """
    t_start = _time.time()
    Tpr = h

    ID = load_target_nodes(target_nodes)
    evector = load_evector(evector_mat)
    evector[:, 2] = 0.0  # planar stretching

    nodes, elements, constraints = read_model_file(model)
    node_number = len(nodes)
    element_number = len(elements)

    boundary = np.zeros((node_number, 3))
    for row in constraints:
        j = int(row[0]) - 1
        if 0 <= j < node_number:
            boundary[j] = row[1:4]

    nodes[:, 1:4] /= 10.0  # Angstrom -> nm
    c1 = 1.0 + zeta * h / 2.0
    c2 = 1.0 - zeta * h / 2.0
    h2 = h * h

    mas = load_mass(mass)
    if mas.shape[0] != node_number:
        raise ValueError(f"mass count {mas.shape[0]} != node count {node_number}")
    mas_col = mas[:, None]

    elem_n1 = elements[:, 1].astype(int) - 1
    elem_n2 = elements[:, 2].astype(int) - 1
    node_init = nodes[:, 1:4]
    boundary_mask = boundary == 1

    v = np.zeros((node_number, 3))
    xn = node_init.copy()
    f = np.zeros((node_number, 3))
    emf = np.zeros(element_number)
    F = np.zeros((node_number, 3))

    acc0 = xn - (h + 0.5 * zeta * h2) * v + 0.5 * h2 * (F + f) / mas_col / c1
    x_n = np.where(boundary_mask, xn, acc0)

    idx = ID - 1
    load_dirs = evector[idx, :].copy()
    ln = np.linalg.norm(load_dirs, axis=1, keepdims=True)
    load_unit = load_dirs / np.where(ln < 1e-12, 1.0, ln)

    pull = evector[idx, :]
    pn = np.linalg.norm(pull, axis=1)
    pull_valid = pn >= 1e-12
    pull_unit = np.zeros_like(pull)
    pull_unit[pull_valid] = pull[pull_valid] / pn[pull_valid, None]

    n = 0
    hc = -h
    time_dmp, ext_dmp, fapp_dmp, fint_dmp = [], [], [], []
    initial_pull_pos = xn[idx, :].copy()
    structure_data = []

    total_steps = int(ET / h)
    progress_interval = max(1, total_steps // 10)
    if verbose:
        print(f"Nodes {node_number}, elements {element_number}, target {len(ID)}; "
              f"{total_steps} steps")

    for step in range(total_steps + 1):
        time_var = step * h
        if verbose and (step % progress_interval == 0 or step == total_steps):
            print(f"  {100 * step / total_steps:.0f}%  (step {step}/{total_steps})")

        factor = triangle_load_factor(time_var, ramp_t0, ramp_t1, unload_t1)
        current_force = Fmax * factor
        F[:] = 0.0
        if len(ID) > 0:
            F[idx, :] = current_force * load_unit

        acc = (2.0 * xn - c2 * x_n + h2 * (F + f) / mas_col) / c1
        xn1 = np.where(boundary_mask, node_init, acc)

        d_old = xn[elem_n2] - xn[elem_n1]
        d_new = xn1[elem_n2] - xn1[elem_n1]
        la = np.linalg.norm(d_old, axis=1)
        lt = np.linalg.norm(d_new, axis=1)
        f_new = emf + E * A * (lt - la) / la
        f_new[lt >= 2.0] = 0.0          # element failure
        emf[:] = f_new
        ef2 = (f_new / lt)[:, None] * d_new
        f[:] = 0.0
        np.add.at(f, elem_n1, ef2)
        np.add.at(f, elem_n2, -ef2)

        F_snapshot = F.astype(np.float32).copy()
        f_snapshot = f.astype(np.float32).copy()
        x_n = xn.copy()
        xn = xn1.copy()

        if abs(hc - Tpr) < 0.01 * h:
            n += 1
            hc = 0.0
            disp = xn[idx, :] - initial_pull_pos
            ext = np.einsum("ij,ij->i", disp, pull_unit)
            ext[~pull_valid] = 0.0
            ext_dmp.append(float(np.mean(ext)))
            fint = np.einsum("ij,ij->i", f[idx, :], pull_unit)
            fint[~pull_valid] = 0.0
            total_internal = float(np.sum(fint))
            time_dmp.append(time_var)
            fapp_dmp.append(current_force)
            fint_dmp.append(-total_internal)
            payload = {"time": time_var, "applied_force": current_force,
                       "extension": float(np.mean(ext)),
                       "internal_force": -total_internal,
                       "xn": xn.copy(), "emf": emf.copy()}
            if (n - 1) % max(1, force_save_stride) == 0:
                payload["F_nodes"] = F_snapshot
                if save_internal:
                    payload["f_nodes"] = f_snapshot
            structure_data.append(payload)
        hc += h

    if verbose:
        print(f"Done in {_time.time() - t_start:.2f}s")

    os.makedirs(os.path.dirname(out), exist_ok=True)
    _write_hdf5(out, nodes, elements, ID, evector, structure_data,
                time_dmp, fapp_dmp, ext_dmp, fint_dmp, node_number, compress)

    return {"steps": len(structure_data),
            "final_extension": ext_dmp[-1] if ext_dmp else None,
            "final_internal_force": fint_dmp[-1] if fint_dmp else None,
            "out": out}


def _write_hdf5(out, nodes, elements, ID, evector, structure_data,
                time_dmp, fapp_dmp, ext_dmp, fint_dmp, node_number, compress):
    with h5py.File(out, "w") as f5:
        f5.create_dataset("time_steps", data=np.array(time_dmp))
        f5.create_dataset("applied_forces", data=np.array(fapp_dmp))
        f5.create_dataset("extensions", data=np.array(ext_dmp))
        f5.create_dataset("internal_forces", data=np.array(fint_dmp))
        f5.create_dataset("applied_forces_semantics", data=np.bytes_("per-node-magnitude"))
        f5.create_dataset("applied_forces_total", data=np.array(fapp_dmp) * len(ID))
        f5.create_dataset("initial_nodes", data=nodes[:, 1:4])
        f5.create_dataset("elements", data=elements)
        f5.create_dataset("target_nodes", data=ID)
        f5.create_dataset("evector", data=evector)
        f5.create_dataset("final_nodes", data=structure_data[-1]["xn"])

        ts = f5.create_group("timeseries")
        ns = len(structure_data)
        coords_arr = np.zeros((ns, node_number, 3))
        time_arr = np.zeros(ns)
        for i, sd in enumerate(structure_data):
            coords_arr[i] = sd["xn"] * 1e-9      # nm -> m
            time_arr[i] = sd["time"] * 1e-12     # ps -> s
        ts.create_dataset("node_coords", data=coords_arr, compression="gzip",
                          compression_opts=compress)
        ts.create_dataset("time", data=time_arr)
        conn = elements[:, 1:3].astype(int) - 1
        ts.create_dataset("element_connectivity",
                          data=np.tile(conn[np.newaxis], (ns, 1, 1)),
                          compression="gzip", compression_opts=compress)

        sg = f5.create_group("step_data")
        for i, sd in enumerate(structure_data):
            g = sg.create_group(f"step_{i:04d}")
            g.attrs["time"] = sd["time"]
            g.attrs["applied_force_scalar"] = sd["applied_force"]
            g.attrs["extension_mean"] = sd["extension"]
            g.attrs["internal_force_sum_proj"] = sd["internal_force"]
            g.create_dataset("node_coordinates", data=sd["xn"])
            g.create_dataset("element_forces", data=sd["emf"])
            if "F_nodes" in sd:
                g.create_dataset("node_forces", data=sd["F_nodes"],
                                 compression="gzip", compression_opts=compress)
            if "f_nodes" in sd:
                g.create_dataset("node_forces_internal", data=sd["f_nodes"],
                                 compression="gzip", compression_opts=compress)
