#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generic input builder for the MemProtein dynamics simulator.

Given a single OPM-aligned PDB and the per-chain transmembrane (TM) segment
annotation copied straight from the OPM protein page, this script produces ALL
the input files the simulator (main.py / main_fast.py) needs:

    MODEL.txt     node coords + element connectivity (+ constraints)
    targetNode.txt  membrane residues to be force-loaded
    mass.txt      per-node mass
    evector.mat   per-node force directions

Design notes / decisions (validated against the original Piezo1 6b3r setup):
  * Nodes      = every C-alpha in the PDB, in file order (node id = order).
  * Elements   = residue pairs within `--cutoff` Angstrom (default 10, which
                 exactly reproduces the original 36844-element Piezo1 model).
  * Membrane residues (targetNode) = C-alpha whose (chain, resid) falls inside
                 that chain's OPM TM-segment ranges. This is the "manual"
                 selection 学长 asked for: paste OPM's TM-segment text, the
                 program ingests it. No automatic geometric guessing.
  * evector    = in-membrane-plane radial direction (outward from the XY
                 centroid), i.e. membrane tension pulling the protein open.
                 The Z component is 0 (the simulator zeroes Z anyway).
  * mass       = 1 per node (matches the original mass.txt).
  * constraints= none (the original model is free-floating; its 3 "constraints"
                 had fix flags 0 0 0, i.e. no-ops).

Usage:
    python build_inputs.py --pdb Raw/6b3r.pdb --tm Raw/6b3r_tm.txt --outdir Raw
"""

import argparse
import os
import re
from typing import Dict, List, Tuple

import numpy as np
from scipy.io import savemat
from scipy.spatial import cKDTree

Segment = Tuple[int, int]


def parse_pdb_ca(pdb_path: str, include_hetatm: bool = True
                 ) -> Tuple[np.ndarray, List[str], List[int]]:
    """Read C-alpha atoms from a PDB in file order.

    Returns:
        coords  (N, 3) float array in Angstrom
        chains  list of chain ids (length N)
        resids  list of residue sequence numbers (length N)

    DUM (OPM membrane dummy) atoms are always skipped. HETATM C-alpha atoms
    (e.g. OPM's "UNK" unknown-density fragments) are included by default to
    reproduce the original model; pass include_hetatm=False to drop them.
    """
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
            if line[12:16].strip() != "CA":
                continue
            if line[17:20].strip() == "DUM":
                continue
            coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            chains.append(line[21])
            resids.append(int(line[22:26]))
    return np.asarray(coords, dtype=np.float64), chains, resids


def parse_opm_tm_text(tm_path: str) -> Dict[str, List[Segment]]:
    """Parse OPM TM-segment text into {chain: [(start, end), ...]}.

    Accepts lines copied directly from an OPM protein page, e.g.:
        A - Tilt: 22 - TM segments: 1( 581- 599), 2( 610- 629), 3( 631- 652)
    Whitespace inside the ranges is tolerated. Lines without a recognizable
    "<chain> - ... TM segments:" header are ignored.
    """
    segments: Dict[str, List[Segment]] = {}
    header_re = re.compile(r"^\s*([A-Za-z0-9])\s*-.*?TM segments\s*:(.*)$")
    pair_re = re.compile(r"\(\s*(\d+)\s*-\s*(\d+)\s*\)")
    with open(tm_path) as fh:
        for line in fh:
            m = header_re.match(line)
            if not m:
                continue
            chain = m.group(1)
            pairs = [(int(a), int(b)) for a, b in pair_re.findall(m.group(2))]
            if pairs:
                segments.setdefault(chain, []).extend(pairs)
    return segments


def select_membrane_nodes(chains: List[str], resids: List[int],
                          tm: Dict[str, List[Segment]]) -> np.ndarray:
    """Return 1-based node ids whose (chain, resid) lies in a TM segment."""
    selected: List[int] = []
    for i, (ch, r) in enumerate(zip(chains, resids)):
        for start, end in tm.get(ch, ()):
            if start <= r <= end:
                selected.append(i + 1)
                break
    return np.asarray(selected, dtype=int)


def build_elements(coords: np.ndarray, cutoff: float) -> np.ndarray:
    """Build elastic-network elements: residue pairs within `cutoff` Angstrom.

    Returns an (M, 3) int array: [element_id, node1_id, node2_id] (1-based).
    """
    pairs = cKDTree(coords).query_pairs(cutoff, output_type="ndarray")
    pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]  # deterministic order
    elements = np.empty((len(pairs), 3), dtype=int)
    elements[:, 0] = np.arange(1, len(pairs) + 1)
    elements[:, 1:] = pairs + 1
    return elements


def radial_evector(coords: np.ndarray) -> np.ndarray:
    """In-plane radial unit vectors (outward from XY centroid), Z = 0."""
    vec = coords - coords.mean(axis=0)
    vec[:, 2] = 0.0
    norms = np.linalg.norm(vec, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return vec / norms


def write_model_file(path: str, coords: np.ndarray, elements: np.ndarray,
                     constraints: np.ndarray) -> None:
    """Write MODEL.txt in the format read by utils.read_model_file."""
    with open(path, "w") as fh:
        fh.write(f"{len(coords)}\n")
        for i, (x, y, z) in enumerate(coords, start=1):
            fh.write(f"{i}\t{x:.3f}\t{y:.3f}\t{z:.3f}\n")
        fh.write(f"{len(elements)}\n")
        for eid, n1, n2 in elements:
            fh.write(f"{eid}\t{n1}\t{n2}\n")
        fh.write(f"{len(constraints)}\n")
        for row in constraints:
            fh.write("\t".join(str(int(v)) for v in row) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build simulator inputs from an OPM PDB + TM-segment text.")
    ap.add_argument("--pdb", required=True, help="OPM-aligned PDB file")
    ap.add_argument("--tm", required=True, help="Text file with OPM TM-segment lines (per chain)")
    ap.add_argument("--outdir", default="Raw", help="Output directory")
    ap.add_argument("--cutoff", type=float, default=10.0, help="Elastic-network distance cutoff (Angstrom)")
    ap.add_argument("--mass", type=float, default=1.0, help="Per-node mass value")
    ap.add_argument("--drop-hetatm", action="store_true",
                    help="Exclude HETATM C-alpha (e.g. OPM UNK fragments); default keeps them")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    coords, chains, resids = parse_pdb_ca(args.pdb, include_hetatm=not args.drop_hetatm)
    n_nodes = len(coords)
    print(f"Parsed {n_nodes} C-alpha nodes from {args.pdb}")

    tm = parse_opm_tm_text(args.tm)
    print(f"Parsed TM segments for chains: {', '.join(sorted(tm))} "
          f"({sum(len(v) for v in tm.values())} segments total)")

    target = select_membrane_nodes(chains, resids, tm)
    print(f"Selected {len(target)} membrane (force-loaded) nodes")

    elements = build_elements(coords, args.cutoff)
    print(f"Built {len(elements)} elements (cutoff {args.cutoff} A)")

    evector = radial_evector(coords)
    constraints = np.empty((0, 4), dtype=int)  # free-floating, like the original

    model_path = os.path.join(args.outdir, "MODEL.txt")
    target_path = os.path.join(args.outdir, "targetNode.txt")
    mass_path = os.path.join(args.outdir, "mass.txt")
    evec_path = os.path.join(args.outdir, "evector.mat")

    write_model_file(model_path, coords, elements, constraints)
    np.savetxt(target_path, target[np.newaxis, :], fmt="%d")
    np.savetxt(mass_path, np.full(n_nodes, args.mass), fmt="%g")
    savemat(evec_path, {"evector": evector})

    print("Wrote:")
    for p in (model_path, target_path, mass_path, evec_path):
        print(f"  {p}")


if __name__ == "__main__":
    main()
