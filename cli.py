#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Command-line entry point for the MemProtein pipeline.

Examples:
    # end-to-end: OPM PDB + TM text -> inputs -> simulation -> figures
    python cli.py run --pdb data/raw/6lod.pdb --tm data/raw/6lod_tm.txt

    # individual stages
    python cli.py preprocess --pdb data/raw/6lod.pdb --tm data/raw/6lod_tm.txt
    python cli.py simulate --ET 100 --Fmax 0.1
    python cli.py analyze --kind heatmap --n-times 10

Run `python cli.py <command> -h` for the options of each stage.
"""

import argparse

from memprotein.preprocess import build_inputs
from memprotein.simulate import run_simulation
from memprotein.pipeline import run
from memprotein import analysis


def _add_sim_args(p):
    p.add_argument("--ET", type=float, default=100.0, help="total time (ps)")
    p.add_argument("--h", type=float, default=0.1, help="time step (ps)")
    p.add_argument("--Fmax", type=float, default=0.1, help="peak per-node force (pN)")
    p.add_argument("--E", type=float, default=1000.0, help="Young's modulus (pN/nm^2)")
    p.add_argument("--A", type=float, default=0.01, help="cross-section area (nm^2)")
    p.add_argument("--zeta", type=float, default=0.01, help="damping factor")
    p.add_argument("--ramp-t0", type=float, default=0.01)
    p.add_argument("--ramp-t1", type=float, default=50.0)
    p.add_argument("--unload-t1", type=float, default=100.0)


def main():
    ap = argparse.ArgumentParser(description="MemProtein force-deformation pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # run (end-to-end)
    r = sub.add_parser("run", help="end-to-end: preprocess + simulate + analyze")
    r.add_argument("--pdb", required=True)
    r.add_argument("--tm", required=True)
    r.add_argument("--cutoff", type=float, default=10.0)
    r.add_argument("--include-hetatm", action="store_true")
    r.add_argument("--no-analyze", action="store_true", help="skip the analysis figures")
    _add_sim_args(r)

    # preprocess
    pp = sub.add_parser("preprocess", help="OPM PDB + TM -> model inputs")
    pp.add_argument("--pdb", required=True)
    pp.add_argument("--tm", required=True)
    pp.add_argument("--out-dir", default="data/inputs")
    pp.add_argument("--cutoff", type=float, default=10.0)
    pp.add_argument("--include-hetatm", action="store_true")

    # simulate
    sm = sub.add_parser("simulate", help="run the dynamics on existing inputs")
    sm.add_argument("--inputs-dir", default="data/inputs")
    sm.add_argument("--out", default="data/results/simulation_data.h5")
    _add_sim_args(sm)

    # analyze
    an = sub.add_parser("analyze", help="make correlation figures from a result file")
    an.add_argument("--h5", default="data/results/simulation_data.h5")
    an.add_argument("--kind", choices=["all", "instant", "heatmap", "anchor", "stack"],
                    default="all")
    an.add_argument("--anchor", type=int, default=1)
    an.add_argument("--time", type=float, default=50.0)
    an.add_argument("--n-times", type=int, default=10)
    an.add_argument("--no-align", action="store_true", help="disable rigid-body alignment")

    args = ap.parse_args()

    if args.cmd == "run":
        sim_kwargs = dict(ET=args.ET, h=args.h, Fmax=args.Fmax, E=args.E, A=args.A,
                          zeta=args.zeta, ramp_t0=args.ramp_t0, ramp_t1=args.ramp_t1,
                          unload_t1=args.unload_t1)
        run(args.pdb, args.tm, cutoff=args.cutoff, include_hetatm=args.include_hetatm,
            analyze=not args.no_analyze, sim_kwargs=sim_kwargs)

    elif args.cmd == "preprocess":
        s = build_inputs(args.pdb, args.tm, out_dir=args.out_dir, cutoff=args.cutoff,
                         include_hetatm=args.include_hetatm)
        print(f"nodes {s['n_nodes']}, elements {s['n_elements']}, "
              f"membrane {s['n_membrane']}, chains {s['chains']}")

    elif args.cmd == "simulate":
        d = args.inputs_dir
        run_simulation(model=f"{d}/MODEL.txt", target_nodes=f"{d}/targetNode.txt",
                       mass=f"{d}/mass.txt", evector_mat=f"{d}/evector.mat",
                       meta=f"{d}/nodes.npz",
                       out=args.out, ET=args.ET, h=args.h, Fmax=args.Fmax, E=args.E,
                       A=args.A, zeta=args.zeta, ramp_t0=args.ramp_t0,
                       ramp_t1=args.ramp_t1, unload_t1=args.unload_t1)

    elif args.cmd == "analyze":
        align = not args.no_align
        k = args.kind
        if k in ("all", "instant"):
            print(analysis.correlation_vs_distance(args.h5, align=align))
        if k in ("all", "heatmap"):
            print(analysis.distance_heatmap(args.h5, n_times=args.n_times, align=align))
        if k in ("all", "anchor"):
            print(analysis.anchor_scatter(args.h5, anchor=args.anchor, time_ps=args.time, align=align))
        if k in ("all", "stack"):
            print(analysis.anchor_stack(args.h5, time_ps=args.time, align=align))


if __name__ == "__main__":
    main()
