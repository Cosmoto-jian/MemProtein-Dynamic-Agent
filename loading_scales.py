# -*- coding: utf-8 -*-
"""Loading-phase, geometry-normalized correlation-scale analysis per PDB.

For each structure: only the loading half (0-50 ps) is used, split into 10 load
levels (every 5 ps). It extracts the in-plane (parallel) and vertical (perp)
correlation lengths, normalizes them by the geometric scales L_parallel / L_perp,
and saves the xi_hat(load) evolution plus the chi_excess curve.

Run:  .venv/bin/python loading_scales.py
"""
from collections import Counter

import numpy as np

from memprotein import analysis as an

JOBS = [("6b3r", "data/results/sim_6b3r.h5"),
        ("6w7b", "data/results/sim_6w7b.h5"),
        ("7aa5", "data/results/sim_7aa5.h5")]

for name, h5 in JOBS:
    chains, _ = an.load_node_meta(h5)
    chain = Counter(chains).most_common(1)[0][0]      # representative subunit
    nodes = an.chain_nodes(h5, chain)
    ti, tj = np.triu_indices(len(nodes), k=1)
    pairs = np.stack([nodes[ti], nodes[tj]], axis=1)
    geom = an.geometry_scales(h5, node_subset=nodes)
    print(f"[{name}] chain {chain}: {len(nodes)} nodes, {len(pairs)} pairs | "
          f"L_par={geom.L_parallel:.2f} L_perp={geom.L_perp:.2f} "
          f"eta={geom.eta_shape:.2f}", flush=True)
    out = an.correlation_scales_loading(h5, t_load=50.0, n_frames=10,
                                        pairs=pairs, node_subset=nodes,
                                        out=f"data/results/scale_{name}")
    print(f"[{name}] -> {out}", flush=True)
print("ALL DONE", flush=True)
