# -*- coding: utf-8 -*-
"""Run each protein at several peak forces and tag the result file with the force.

Output: data/results/sim_<id>_F<force>.h5  (existing files are skipped).

Also reports a crude structural-integrity check: the largest node displacement
over the whole trajectory, relative to the protein's in-plane size L_parallel.
ratio >> 1 (or any NaN) means the structure was likely blown apart by too large
a force.

Typical use::

    # Auto-download 12 proteins from OPM, run all at default forces
    .venv/bin/python run_forces.py --pdb-ids 1bl8,1c3w,1j4n,1u19,2oar,2oau,2rh1,4bw5,6b3r,6mgv,6w7b,7aa5 --fetch

    # Batch from existing data/raw/
    .venv/bin/python run_forces.py

    # Specific forces
    .venv/bin/python run_forces.py --forces 0.01 0.05 0.1
"""
import argparse
import glob
import os

import numpy as np

from memprotein.preprocess import build_inputs
from memprotein.simulate import run_simulation
from memprotein import analysis as an
from memprotein.opm import ensure_raw_inputs, FetchResult, _validate

DEFAULT_FORCES = [0.01, 0.05, 0.1]           # pN per node
RESULTS = "data/results"
RAW_DIR = "data/raw"


def damage_metric(h5: str) -> tuple:
    """(max_disp_nm, ratio_to_Lpar, has_nan) over the whole trajectory."""
    coords, _, _ = an.load_trajectory(h5)
    disp = coords - coords[0]
    maxd = float(np.nanmax(np.linalg.norm(disp, axis=2)))
    geom = an.geometry_scales(h5)
    return maxd, maxd / geom.L_parallel, bool(np.isnan(coords).any())


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch force-control simulation")
    ap.add_argument("--pdb-ids", default=None,
                    help="Comma-separated PDB IDs (e.g. 6b3r,1bl8). "
                         "Without this, defaults to glob(data/raw/*.pdb).")
    ap.add_argument("--fetch", action="store_true",
                    help="Auto-download missing PDBs and TM annotations from OPM "
                         "(requires --pdb-ids).")
    ap.add_argument("--forces", nargs="+", type=float, default=DEFAULT_FORCES,
                    help=f"Per-node peak force values (pN). Default: {DEFAULT_FORCES}")
    ap.add_argument("--raw-dir", default=RAW_DIR,
                    help=f"Directory for raw PDB + tm files. Default: {RAW_DIR}")
    args = ap.parse_args()

    forces = args.forces
    raw_dir = os.path.abspath(args.raw_dir)

    # --- Step 1: resolve the protein list ---
    skipped: list[FetchResult] = []
    warned: list[FetchResult] = []

    if args.pdb_ids:
        pdb_ids = [s.strip() for s in args.pdb_ids.split(",") if s.strip()]
        if args.fetch:
            print(f"Fetching {len(pdb_ids)} proteins from OPM …\n")
            results = ensure_raw_inputs(pdb_ids, raw_dir=raw_dir)
            # Split by status
            ok_ids = []
            for r in results:
                if r.status == "skip":
                    skipped.append(r)
                elif r.status == "warn":
                    warned.append(r)
                    ok_ids.append(r.pdb_id)
                else:
                    ok_ids.append(r.pdb_id)
            pdb_ids = ok_ids
        else:
            # Validate that files exist (friendly error for missing ones)
            missing = []
            for pid in pdb_ids:
                p = os.path.join(raw_dir, f"{pid.lower()}.pdb")
                t = os.path.join(raw_dir, f"{pid.lower()}_tm.txt")
                if not os.path.exists(p) or not os.path.exists(t):
                    missing.append(pid)
            if missing:
                print(f"Missing PDB/TM for: {','.join(missing)}")
                print("Hint: re-run with --fetch to auto-download from OPM.")
                return
    else:
        # Backward-compatible: discover from data/raw/
        pdb_files = sorted(glob.glob(f"{raw_dir}/*.pdb"))
        pdb_ids = [os.path.basename(p)[:-4] for p in pdb_files]

        # Validate existing files
        for pid in list(pdb_ids):
            pdb = os.path.join(raw_dir, f"{pid.lower()}.pdb")
            tm = os.path.join(raw_dir, f"{pid.lower()}_tm.txt")
            if not os.path.exists(tm):
                continue  # iter_jobs will skip it later
            status, reason = _validate(pdb, tm)
            if status == "skip":
                skipped.append(FetchResult(pid, status, reason, pdb, tm))
                pdb_ids.remove(pid)
            elif status == "warn":
                warned.append(FetchResult(pid, status, reason, pdb, tm))

    if not pdb_ids:
        print("No proteins to process. Use --pdb-ids with --fetch, or place "
              "PDB + _tm.txt files in data/raw/.")
        if skipped:
            _print_skip_summary(skipped)
        return

    # --- Step 2: run ---
    print(f"\nProcessing {len(pdb_ids)} proteins x {len(forces)} forces")
    if warned:
        print(f"({len(warned)} with warnings — see summary below)\n")

    rows = []
    for pid in pdb_ids:
        pdb = os.path.join(raw_dir, f"{pid.lower()}.pdb")
        tm = os.path.join(raw_dir, f"{pid.lower()}_tm.txt")
        if not os.path.exists(pdb):
            print(f"[{pid}] no PDB, skip")
            continue
        if not os.path.exists(tm):
            print(f"[{pid}] no tm file, skip")
            continue

        pre = None
        for F in forces:
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

    # --- Step 3: summary ---
    print("\n=== structural-integrity summary (ratio = maxdisp / L_parallel) ===")
    print(f"{'protein':8}{'F':>6}{'maxdisp':>10}{'ratio':>8}  flag")
    for pid, F, maxd, ratio, tag in rows:
        print(f"{pid:8}{F:>6}{maxd:>10.2f}{ratio:>8.2f}  {tag}")

    if warned:
        _print_warn_summary(warned)
    if skipped:
        _print_skip_summary(skipped)


def _print_warn_summary(warned: list[FetchResult]) -> None:
    print("\n=== warnings (proteins processed but may need attention) ===")
    for r in warned:
        print(f"  {r.pdb_id}: {r.reason}")


def _print_skip_summary(skipped: list[FetchResult]) -> None:
    print("\n=== skipped (excluded from processing) ===")
    for r in skipped:
        print(f"  {r.pdb_id}: {r.reason}")


if __name__ == "__main__":
    main()
