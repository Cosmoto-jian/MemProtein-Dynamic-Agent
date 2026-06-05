#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reading and writing of the simulator's text/MAT input files.

File formats (all produced by memprotein.preprocess):
    MODEL.txt       n_nodes / node rows (id x y z) / n_elements / element rows
                    (id n1 n2) / n_constraints / constraint rows (id fx fy fz)
    targetNode.txt  one line of space-separated 1-based node ids
    mass.txt        one mass value per line
    evector.mat     MATLAB file with key 'evector', shape (n_nodes, 3)
"""

from typing import Tuple

import numpy as np
from scipy.io import loadmat, savemat


def read_model_file(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read MODEL.txt.

    Returns:
        nodes        (N, 4) float [id, x, y, z]
        elements     (M, 3) int   [id, node1, node2]
        constraints  (C, 4) int   [node_id, fixX, fixY, fixZ]
    """
    with open(path) as f:
        n_nodes = int(f.readline())
        nodes = np.array([[float(x) for x in f.readline().split()] for _ in range(n_nodes)])

        n_elements = int(f.readline())
        elements = np.array([[int(x) for x in f.readline().split()] for _ in range(n_elements)],
                            dtype=int)

        n_constraints = int(f.readline())
        constraints = np.array(
            [[int(x) for x in f.readline().split()] for _ in range(n_constraints)],
            dtype=int).reshape(-1, 4)
    return nodes, elements, constraints


def write_model_file(path: str, coords: np.ndarray, elements: np.ndarray,
                     constraints: np.ndarray) -> None:
    """Write MODEL.txt from node coordinates (N,3), elements (M,3) and
    constraints (C,4). Format matches read_model_file."""
    with open(path, "w") as f:
        f.write(f"{len(coords)}\n")
        for i, (x, y, z) in enumerate(coords, start=1):
            f.write(f"{i}\t{x:.3f}\t{y:.3f}\t{z:.3f}\n")
        f.write(f"{len(elements)}\n")
        for eid, n1, n2 in elements:
            f.write(f"{eid}\t{n1}\t{n2}\n")
        f.write(f"{len(constraints)}\n")
        for row in constraints:
            f.write("\t".join(str(int(v)) for v in row) + "\n")


def load_target_nodes(path: str) -> np.ndarray:
    """Load 1-based force-loaded node ids from targetNode.txt."""
    return np.loadtxt(path).astype(int)


def load_mass(path: str) -> np.ndarray:
    """Load per-node mass values from mass.txt."""
    return np.loadtxt(path)


def save_evector(path: str, evector: np.ndarray) -> None:
    """Save per-node force-direction vectors (N,3) to a .mat file."""
    savemat(path, {"evector": evector})


def load_evector(path: str) -> np.ndarray:
    """Load per-node force-direction vectors (N,3) from a .mat file."""
    return loadmat(path)["evector"].copy()
