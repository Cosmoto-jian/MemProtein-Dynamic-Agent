# -*- coding: utf-8 -*-
"""Run each protein at several peak forces and tag the result file with the force.

Output: data/results/sim_<id>_F<force>.h5  (existing files are skipped).

Also reports a crude structural-integrity check: the largest node displacement
over the whole trajectory, relative to the protein's in-plane size L_parallel.
ratio >> 1 (or any NaN) means the structure was likely blown apart by too large
a force.

Run:  .venv/bin/python run_forces.py
"""
import glob
import os

import numpy as np

from memprotein.preprocess import build_inputs
from memprotein.simulate import run_simulation
from memprotein import analysis as an

FORCES = [0.01, 0.05, 0.1]           # pN per node
RESULTS = "data/results"


def damage_metric(h5: str) -> tuple:
    """(max_disp_nm, ratio_to_Lpar, has_nan) over the whole trajectory."""
    coords, _, _ = an.load_trajectory(h5)
    disp = coords - coords[0]
    maxd = float(np.nanmax(np.linalg.norm(disp, axis=2)))
    geom = an.geometry_scales(h5)
    return maxd, maxd / geom.L_parallel, bool(np.isnan(coords).any())


def main() -> None:
    pdbs = sorted(glob.glob("data/raw/*.pdb"))
    print(f"{len(pdbs)} proteins x {len(FORCES)} forces\n")
    rows = []
    for pdb in pdbs:
        pid = os.path.basename(pdb)[:-4]
        tm = f"data/raw/{pid}_tm.txt"
        if not os.path.exists(tm):
            print(f"[{pid}] no tm file, skip"); continue
        pre = None
        for F in FORCES:
            out = f"{RESULTS}/sim_{pid}_F{F}.h5"
            if os.path.exists(out):
                print(f"[{pid} F={F}] exists, skip")
            else:
                if pre is None:                          # preprocess once per protein
                    pre = build_inputs(pdb, tm)
                run_simulation(model=pre["paths"]["model"],
                               target_nodes=pre["paths"]["target"],
                               mass=pre["paths"]["mass"],
                               evector_mat=pre["paths"]["evector"],
                               meta=pre["paths"]["meta"],
                               out=out, Fmax=F, verbose=False)
            maxd, ratio, nan = damage_metric(out)
            tag = "NaN!" if nan else ("DAMAGED?" if ratio > 1.0 else "ok")
            rows.append((pid, F, maxd, ratio, tag))
            print(f"[{pid} F={F}] maxdisp={maxd:.2f}nm  ratio={ratio:.2f}  {tag}", flush=True)

    print("\n=== structural-integrity summary (ratio = maxdisp / L_parallel) ===")
    print(f"{'protein':8}{'F':>6}{'maxdisp':>10}{'ratio':>8}  flag")
    for pid, F, maxd, ratio, tag in rows:
        print(f"{pid:8}{F:>6}{maxd:>10.2f}{ratio:>8.2f}  {tag}")


if __name__ == "__main__":
    main()
