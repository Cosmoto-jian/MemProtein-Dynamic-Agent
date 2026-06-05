#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""End-to-end orchestration: OPM PDB + TM text -> inputs -> simulation -> figures.

    from memprotein.pipeline import run
    run("data/raw/6lod.pdb", "data/raw/6lod_tm.txt")
"""

import os
from typing import Optional

from .preprocess import build_inputs
from .simulate import run_simulation
from . import analysis


def run(pdb: str, tm: str, inputs_dir: str = "data/inputs",
        results_dir: str = "data/results", cutoff: float = 10.0,
        include_hetatm: bool = False, analyze: bool = True,
        sim_kwargs: Optional[dict] = None) -> dict:
    """Run the whole pipeline. Returns a summary dict.

    sim_kwargs is forwarded to run_simulation (ET, h, Fmax, E, A, ...).
    """
    sim_kwargs = dict(sim_kwargs or {})
    os.makedirs(results_dir, exist_ok=True)

    print(f"[1/3] preprocess  {os.path.basename(pdb)}")
    pre = build_inputs(pdb, tm, out_dir=inputs_dir, cutoff=cutoff,
                       include_hetatm=include_hetatm)
    print(f"      nodes {pre['n_nodes']}, elements {pre['n_elements']}, "
          f"membrane {pre['n_membrane']}, chains {pre['chains']}")

    print("[2/3] simulate")
    h5 = os.path.join(results_dir, "simulation_data.h5")
    sim = run_simulation(model=pre["paths"]["model"],
                         target_nodes=pre["paths"]["target"],
                         mass=pre["paths"]["mass"],
                         evector_mat=pre["paths"]["evector"],
                         out=h5, verbose=True, **sim_kwargs)

    figures = []
    if analyze:
        print("[3/3] analyze")
        figures += [
            analysis.correlation_vs_distance(h5, out=os.path.join(results_dir, "instant_corr")),
            analysis.distance_heatmap(h5, out=os.path.join(results_dir, "distance_corr")),
            analysis.anchor_scatter(h5, out=os.path.join(results_dir, "anchor_corr")),
            analysis.anchor_stack(h5, out=os.path.join(results_dir, "anchor_stack")),
        ]
        for p in figures:
            print(f"      {p}")

    return {"preprocess": pre, "simulate": sim, "figures": figures, "h5": h5}
