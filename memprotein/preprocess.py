#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build all simulator inputs from a single OPM-aligned PDB plus the OPM
transmembrane-segment annotation.

Pipeline of one call to build_inputs():
    PDB C-alpha atoms            -> nodes (one bead per residue)
    distance cutoff (10 A)       -> elements (elastic network)
    OPM TM segments per chain    -> targetNode (membrane residues to load)
    in-plane radial direction    -> evector (force directions)
    constant 1.0                 -> mass

Design decisions are validated against the original Piezo1 6b3r setup; see the
project README.
"""

import os
import re
from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial import cKDTree

from .io import write_model_file, save_evector

Segment = Tuple[int, int]


def parse_pdb_ca(pdb_path: str, include_hetatm: bool = False
                 ) -> Tuple[np.ndarray, List[str], List[int]]:
    """Read C-alpha atoms in file order. Returns coords (N,3) Angstrom, chain
    ids, residue numbers. DUM (membrane dummy) atoms are skipped; HETATM C-alpha
    (lipids, ligands, UNK fragments) are excluded by default."""
    coords: List[List[float]] = []
    chains: List[str] = []
    resids: List[int] = []
    with open(pdb_path) as fh:
        for line in fh:
            rec = line[:6].strip()
            if rec not in ("ATOM", "HETATM"):
                continue
            if rec == "HETATM" and not include_hetatm:
                continue
            if line[12:16].strip() != "CA" or line[17:20].strip() == "DUM":
                continue
            coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            chains.append(line[21])
            resids.append(int(line[22:26]))
    return np.asarray(coords, dtype=np.float64), chains, resids


def parse_opm_tm_text(tm_path: str) -> Dict[str, List[Segment]]:
    """Parse OPM TM-segment text into {chain: [(start, end), ...]}.
    Accepts lines copied from an OPM protein page, e.g.
    'A - Tilt: 22 - TM segments: 1( 581- 599), 2( 610- 629)'."""
    segments: Dict[str, List[Segment]] = {}
    header_re = re.compile(r"^\s*([A-Za-z0-9])\s*-.*?TM segments\s*:(.*)$")
    pair_re = re.compile(r"\(\s*(\d+)\s*-\s*(\d+)\s*\)")
    with open(tm_path) as fh:
        for line in fh:
            m = header_re.match(line)
            if not m:
                continue
            pairs = [(int(a), int(b)) for a, b in pair_re.findall(m.group(2))]
            if pairs:
                segments.setdefault(m.group(1), []).extend(pairs)
    return segments


def select_membrane_nodes(chains: List[str], resids: List[int],
                          tm: Dict[str, List[Segment]]) -> np.ndarray:
    """1-based node ids whose (chain, resid) falls in a TM segment."""
    selected = [i + 1 for i, (ch, r) in enumerate(zip(chains, resids))
                if any(a <= r <= b for a, b in tm.get(ch, ()))]
    return np.asarray(selected, dtype=int)


def build_elements(coords: np.ndarray, cutoff: float = 10.0) -> np.ndarray:
    """Elastic-network elements: residue pairs within `cutoff` Angstrom.
    Returns (M, 3) int [element_id, node1, node2] (1-based)."""
    pairs = cKDTree(coords).query_pairs(cutoff, output_type="ndarray")
    pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
    elements = np.empty((len(pairs), 3), dtype=int)
    elements[:, 0] = np.arange(1, len(pairs) + 1)
    elements[:, 1:] = pairs + 1
    return elements


def radial_evector(coords: np.ndarray) -> np.ndarray:
    """In-plane radial unit vectors (outward from the XY centroid), Z = 0."""
    vec = coords - coords.mean(axis=0)
    vec[:, 2] = 0.0
    norms = np.linalg.norm(vec, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return vec / norms


def build_inputs(pdb_path: str, tm_path: str, out_dir: str = "data/inputs",
                 cutoff: float = 10.0, mass: float = 1.0,
                 include_hetatm: bool = False) -> dict:
    """Generate MODEL.txt, targetNode.txt, mass.txt and evector.mat in out_dir.

    Returns a summary dict (n_nodes, n_elements, n_membrane, chains, paths).
    """
    os.makedirs(out_dir, exist_ok=True)
    coords, chains, resids = parse_pdb_ca(pdb_path, include_hetatm=include_hetatm)
    tm = parse_opm_tm_text(tm_path)
    target = select_membrane_nodes(chains, resids, tm)
    elements = build_elements(coords, cutoff)
    evector = radial_evector(coords)
    constraints = np.empty((0, 4), dtype=int)  # free-floating (matches original)

    paths = {
        "model": os.path.join(out_dir, "MODEL.txt"),
        "target": os.path.join(out_dir, "targetNode.txt"),
        "mass": os.path.join(out_dir, "mass.txt"),
        "evector": os.path.join(out_dir, "evector.mat"),
    }
    write_model_file(paths["model"], coords, elements, constraints)
    np.savetxt(paths["target"], target[np.newaxis, :], fmt="%d")
    np.savetxt(paths["mass"], np.full(len(coords), mass), fmt="%g")
    save_evector(paths["evector"], evector)

    return {
        "n_nodes": len(coords),
        "n_elements": len(elements),
        "n_membrane": len(target),
        "chains": sorted(tm),
        "paths": paths,
    }
