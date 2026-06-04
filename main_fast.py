#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Piezo1 force-control simulation (triangle load) — VECTORIZED / FAST version.

This is a drop-in faster equivalent of main.py. The physics, numerical scheme
(central-difference explicit integration), unit handling, and HDF5 output format
are IDENTICAL to main.py. The only change is that the per-step Python loops over
nodes and over the ~36844 elements are replaced by NumPy vector operations
(np.add.at scatter assembly). Results should match main.py to floating-point
precision.

Keep main.py unchanged as the reference; use this for production / batch runs.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import time as python_time
import sys
import os
import argparse
import h5py

# Add Processed directory to path to import utils module
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'Processed'))
from utils import load_target_nodes, load_mass_data, read_model_file


def parse_args():
    ap = argparse.ArgumentParser(description="Piezo1 force-control simulator (triangle load) — vectorized fast version.")
    # Simulation parameters
    ap.add_argument("--ET", type=float, default=100.0, help="Total simulation time (ps)")
    ap.add_argument("--h", type=float, default=1.0e-1, help="Time step (ps)")
    ap.add_argument("--zeta", type=float, default=0.01, help="Viscous damping factor")
    ap.add_argument("--E", type=float, default=1000.0, help="Young's modulus (pN/nm^2)")
    ap.add_argument("--A", type=float, default=0.01, help="Cross-section area (nm^2)")

    # Force loading (pure triangle wave: up 50 ps -> down to 100 ps)
    ap.add_argument("--Fmax", type=float, default=0.1, help="Per-node max force magnitude (pN)")
    ap.add_argument("--ramp_t0", type=float, default=0.01, help="Ramp start time (ps)")
    ap.add_argument("--ramp_t1", type=float, default=50.0, help="Ramp-to-peak end time (ps)")
    ap.add_argument("--unload_t1", type=float, default=100.0, help="Unloading end time (ps)")

    # I/O
    ap.add_argument("--model", type=str, default="Raw/MODEL.txt", help="Model file")
    ap.add_argument("--target_nodes", type=str, default="Raw/targetNode.txt", help="Target nodes file")
    ap.add_argument("--evector_mat", type=str, default="Processed/evector.mat", help="MAT file for evector")
    ap.add_argument("--mass", type=str, default="Raw/mass.txt", help="Mass file (per node)")
    ap.add_argument("--out", type=str, default="simulation_data.h5", help="Output HDF5 filename")

    # Visualization and saving
    ap.add_argument("--no-gui", action="store_true", help="Disable 3D preview & blocking input()")
    ap.add_argument("--save-internal", action="store_true", help="Also save per-node internal forces")
    ap.add_argument("--force-save-stride", type=int, default=1, help="Save forces every N output steps")
    ap.add_argument("--compress", type=int, default=4, help="Gzip compression level for big arrays (0-9)")
    return ap.parse_args()


def load_unload_factor(t0: float, t1: float, t2: float, t: float) -> float:
    """
    Pure triangular loading factor (0->1->0):
    0..t0: 0
    t0..t1: Linear ramp up 0->1
    t1..t2: Linear ramp down 1->0
    t>=t2: 0
    If t2 <= t1, it is treated as ramp-up only (maintains 1).
    """
    if t <= t0:
        return 0.0
    if t <= t1:
        if t1 <= t0:
            return 1.0
        return (t - t0) / (t1 - t0)
    if t2 <= t1:
        return 1.0
    if t <= t2:
        return 1.0 - (t - t1) / (t2 - t1)
    return 0.0


def main():
    args = parse_args()
    start_time = python_time.time()

    # ========== Path handling ==========
    def resolve_path(path):
        if not os.path.isabs(path):
            return os.path.join(SCRIPT_DIR, path)
        return path

    args.model = resolve_path(args.model)
    args.target_nodes = resolve_path(args.target_nodes)
    args.evector_mat = resolve_path(args.evector_mat)
    args.mass = resolve_path(args.mass)
    args.out = resolve_path(args.out)

    # ========== Simulation parameters ==========
    ET   = args.ET
    h    = args.h
    Tpr  = h                 # Data output period (consistent with original program)
    zeta = args.zeta
    A    = args.A
    E    = args.E

    # ========== Force loading parameters (triangular) ==========
    max_force       = args.Fmax
    ramp_start_time = args.ramp_t0
    ramp_end_time   = args.ramp_t1
    unload_end_time = args.unload_t1 if args.unload_t1 is not None else ET

    # ========== Stretching setup ==========
    ID = load_target_nodes(args.target_nodes).astype(int)
    print(f"Loaded {len(ID)} target nodes for force loading")

    evector_data = loadmat(args.evector_mat)
    evector = evector_data['evector'].copy()
    # Planar stretching: only load in XY plane
    evector[:, 2] = 0.0

    # ========== Model import ==========
    print("Reading model file...")
    nodes, elements, constraints = read_model_file(args.model)

    node_number = len(nodes)
    element_number = len(elements)
    print(f"Number of nodes: {node_number}")
    print(f"Number of elements: {element_number}")
    print(f"Number of constraints: {len(constraints)}")

    # Boundary conditions: constraints format [NodeID, fixX, fixY, fixZ] (1=fixed, 0=free)
    boundary = np.zeros((node_number, 3))
    for i in range(len(constraints)):
        j = int(constraints[i, 0]) - 1
        if 0 <= j < node_number:
            boundary[j, 0] = constraints[i, 1]
            boundary[j, 1] = constraints[i, 2]
            boundary[j, 2] = constraints[i, 3]

    # ========== Units and initialization ==========
    nodes[:, 1:4] = nodes[:, 1:4] / 10.0  # A -> nm

    c1 = 1.0 + zeta * h / 2.0
    c2 = 1.0 - zeta * h / 2.0
    h2 = h * h

    mas = load_mass_data(args.mass)  # Expected unit: pN.ps^2/nm
    if mas.shape[0] != node_number:
        raise ValueError(f"Number of masses ({mas.shape[0]}) inconsistent with number of nodes ({node_number})")
    mas_col = mas[:, None]  # (n,1) for broadcasting

    # Precompute element connectivity (0-based) and reference coordinates
    elem_n1 = elements[:, 1].astype(int) - 1   # (E,)
    elem_n2 = elements[:, 2].astype(int) - 1   # (E,)

    node_init = nodes[:, 1:4]                   # (n,3) initial coords in nm
    boundary_mask = (boundary == 1)            # (n,3) bool

    v  = np.zeros((node_number, 3))  # Initial velocity
    d  = np.zeros((node_number, 3))  # Initial displacement
    xn = node_init + d               # Current position (t)

    # Initialization
    time_var = 0.0
    hc = -h
    f = np.zeros((node_number, 3))   # Internal forces (net nodal internal force from assembly)
    emf = np.zeros(element_number)   # Element axial force (scalar)
    F = np.zeros((node_number, 3))   # External forces (nodal)

    # Initial x_{t-h}  (vectorized; equivalent to per-component loop in main.py)
    acc0 = xn - (h + 0.5 * zeta * h2) * v + 0.5 * h2 * (F + f) / mas_col / c1
    x_n = np.where(boundary_mask, xn, acc0)

    # Precompute normalized loading directions for the target nodes
    idx = ID - 1
    load_dirs = evector[idx, :].copy()
    load_norms = np.linalg.norm(load_dirs, axis=1, keepdims=True)
    load_norms = np.where(load_norms < 1e-12, 1.0, load_norms)
    load_unit = load_dirs / load_norms                      # (m,3) unit loading direction

    # Pulling directions (unit) for diagnostics; zero-direction rows masked out
    pull_dirs = evector[idx, :]                             # (m,3)
    pull_norm = np.linalg.norm(pull_dirs, axis=1)           # (m,)
    pull_valid = pull_norm >= 1e-12
    pull_unit = np.zeros_like(pull_dirs)
    pull_unit[pull_valid] = pull_dirs[pull_valid] / pull_norm[pull_valid, None]

    # ========== Storage ==========
    n = 0
    time_dmp = []
    extension_dmp = []
    applied_force_dmp = []
    internal_force_dmp = []

    initial_positions = xn.copy()
    initial_pull_pos = initial_positions[idx, :]

    # ========== Preview ==========
    if not args.no_gui:
        print("Generating force loading setup visualization...")
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(nodes[:, 1], nodes[:, 2], nodes[:, 3],
                   c='cyan', s=15, alpha=0.6, label='All nodes')
        force_nodes_coords = nodes[ID-1, 1:4]
        ax.scatter(force_nodes_coords[:, 0], force_nodes_coords[:, 1], force_nodes_coords[:, 2],
                   c='magenta', s=100, label='Force loading nodes')
        ax.quiver(force_nodes_coords[:, 0], force_nodes_coords[:, 1], force_nodes_coords[:, 2],
                  load_unit[:, 0], load_unit[:, 1], load_unit[:, 2],
                  color='red', linewidth=2, label='Force direction')
        ax.set_title('Force loading points and direction visualization')
        ax.set_xlabel('X (nm)'); ax.set_ylabel('Y (nm)'); ax.set_zlabel('Z (nm)')
        ax.legend(); ax.set_box_aspect([1, 1, 1])
        plt.show()
        input("Force loading points and directions displayed. Press Enter to continue calculation...")

    # ========== Time stepping ==========
    print("Starting time integration calculation...")
    total_steps = int(ET / h)
    print(f"Total steps: {total_steps}")
    progress_interval = max(1, total_steps // 10)

    structure_data = []

    for step in range(total_steps + 1):
        time_var = step * h
        if step % progress_interval == 0 or step == total_steps:
            progress = (step / total_steps) * 100
            print(f"Calculating... {progress:.1f}% (Step {step}/{total_steps})")

        # Pure triangular loading factor
        force_factor = load_unload_factor(
            t0=ramp_start_time, t1=ramp_end_time, t2=unload_end_time, t=time_var
        )
        current_applied_force = max_force * force_factor

        # External force: apply along precomputed unit directions at target nodes
        F[:] = 0.0
        if len(ID) > 0:
            F[idx, :] = current_applied_force * load_unit

        # Explicit central-difference integration (vectorized)
        acc = (2.0 * xn - c2 * x_n + h2 * (F + f) / mas_col) / c1
        xn1 = np.where(boundary_mask, node_init, acc)

        # ===== Vectorized element internal-force assembly =====
        d_old = xn[elem_n2] - xn[elem_n1]      # (E,3) vectors at time t
        d_new = xn1[elem_n2] - xn1[elem_n1]    # (E,3) vectors at time t+h
        la = np.linalg.norm(d_old, axis=1)     # (E,) previous length
        lt = np.linalg.norm(d_new, axis=1)     # (E,) current length

        # Rate-form axial force increment: f_new = emf + E*A*(lt-la)/la
        f_new = emf + E * A * (lt - la) / la
        # Element failure: lt >= 2 nm -> force reset to 0 (matches truss3d_force)
        failed = lt >= 2.0
        f_new[failed] = 0.0
        emf[:] = f_new

        # Nodal force vectors: ef2 = f_new * (d_new / lt)
        e_vec = d_new / lt[:, None]
        ef2 = f_new[:, None] * e_vec           # (E,3)

        # Assembly: in main.py f[n1] -= ef1(=-ef2) -> f[n1]+=ef2 ; f[n2] -= ef2
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

            time_dmp.append(time_var)
            applied_force_dmp.append(current_applied_force)

            # Average elongation projected along loading direction (vectorized)
            displacements = xn[idx, :] - initial_pull_pos       # (m,3)
            extension_values = np.einsum('ij,ij->i', displacements, pull_unit)
            extension_values[~pull_valid] = 0.0
            extension_dmp.append(float(np.mean(extension_values)))

            # Internal force projection on pulled nodes (vectorized)
            internal_proj = np.einsum('ij,ij->i', f[idx, :], pull_unit)
            internal_proj[~pull_valid] = 0.0
            total_internal_force = float(np.sum(internal_proj))
            internal_force_dmp.append(-total_internal_force)

            payload = {
                'time': time_var,
                'applied_force': current_applied_force,
                'extension': float(np.mean(extension_values)),
                'internal_force': -total_internal_force,
                'xn': xn.copy(),
                'emf': emf.copy()
            }
            if ((n - 1) % max(1, args.force_save_stride) == 0):
                payload['F_nodes'] = F_snapshot
                if args.save_internal:
                    payload['f_nodes'] = f_snapshot
            structure_data.append(payload)

        hc += h

    end_time = python_time.time()
    print(f"Calculation completed! Time taken: {end_time - start_time:.2f} 秒")

    # ========== Write HDF5 ==========
    hdf5_filename = args.out
    with h5py.File(hdf5_filename, 'w') as f5:
        f5.create_dataset('time_steps', data=np.array(time_dmp))
        f5.create_dataset('applied_forces', data=np.array(applied_force_dmp))
        f5.create_dataset('extensions', data=np.array(extension_dmp))
        f5.create_dataset('internal_forces', data=np.array(internal_force_dmp))

        f5.create_dataset('applied_forces_semantics', data=np.bytes_('per-node-magnitude'))
        f5.create_dataset('applied_forces_total', data=np.array(applied_force_dmp) * len(ID))

        f5.create_dataset('initial_nodes', data=nodes[:, 1:4])
        f5.create_dataset('elements', data=elements)
        f5.create_dataset('target_nodes', data=ID)
        f5.create_dataset('evector', data=evector)
        f5.create_dataset('final_nodes', data=structure_data[-1]['xn'])

        # timeseries
        ts = f5.create_group('timeseries')
        num_steps = len(structure_data)
        node_coords_array = np.zeros((num_steps, node_number, 3), dtype=np.float64)
        time_array = np.zeros(num_steps, dtype=np.float64)
        for si, sd in enumerate(structure_data):
            node_coords_array[si] = sd['xn'] * 1e-9   # nm -> m
            time_array[si] = sd['time'] * 1e-12       # ps -> s
        ts.create_dataset('node_coords', data=node_coords_array,
                          compression='gzip', compression_opts=args.compress)
        ts.create_dataset('time', data=time_array)
        element_connectivity = elements[:, 1:3].astype(int) - 1
        elem_arr = np.tile(element_connectivity[np.newaxis, :, :], (num_steps, 1, 1))
        ts.create_dataset('element_connectivity', data=elem_arr,
                          compression='gzip', compression_opts=args.compress)

        # step_data
        step_group = f5.create_group('step_data')
        for step_idx, sd in enumerate(structure_data):
            step_name = f'step_{step_idx:04d}'
            g = step_group.create_group(step_name)
            g.attrs['time'] = sd['time']
            g.attrs['applied_force_scalar'] = sd['applied_force']
            g.attrs['extension_mean'] = sd['extension']
            g.attrs['internal_force_sum_proj'] = sd['internal_force']
            g.create_dataset('node_coordinates', data=sd['xn'])
            g.create_dataset('element_forces', data=sd['emf'])
            if 'F_nodes' in sd:
                g.create_dataset('node_forces', data=sd['F_nodes'],
                                 compression='gzip', compression_opts=args.compress)
            if 'f_nodes' in sd:
                g.create_dataset('node_forces_internal', data=sd['f_nodes'],
                                 compression='gzip', compression_opts=args.compress)

    print(f"Successfully saved simulation data to: {hdf5_filename}")
    print(f"Output steps: {len(structure_data)} (with every {max(1, args.force_save_stride)} steps saving node-wise force fields)")

    # ========== Summary ==========
    print("\n=== Force-loading simulation results summary (pure triangular loading) ===")
    print(f"Max applied force (per node): {max_force:.3f} pN")
    print(f"Peak time: {ramp_end_time:.2f} ps, unloading end: {unload_end_time:.2f} ps")
    print(f"Final extension (average): {extension_dmp[-1]:.4f} nm")
    print(f"Final internal force (sum of projections): {internal_force_dmp[-1]:.2f} pN")
    print(f"Total simulation time: {ET:.2f} ps, time step: {h:.6f} ps, damping: {zeta:.3f}")


if __name__ == "__main__":
    main()
