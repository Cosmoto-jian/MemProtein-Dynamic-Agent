# -*- coding: utf-8 -*-
"""Loading-phase, geometry-normalized correlation-scale analysis per PDB.

For each structure: only the loading half (0-50 ps) is used, split into 10 load
levels (every 5 ps). It extracts the in-plane (parallel) and vertical (perp)
correlation lengths, normalizes them by the geometric scales L_parallel / L_perp,
and saves the xi_hat(load) evolution plus the chi_excess curve.

Run:  .venv/bin/python loading_scales.py
"""
import glob
import os
import re
import sys
from collections import Counter

import numpy as np

from memprotein import analysis as an

# Pick the peak force to analyze (matches sim_<id>_F<force>.h5). Default 0.01.
FORCE = sys.argv[1] if len(sys.argv) > 1 else "0.01"
_pat = re.compile(rf"sim_(.+)_F{re.escape(FORCE)}\.h5$")
JOBS = sorted((m.group(1), p) for p in glob.glob(f"data/results/sim_*_F{FORCE}.h5")
              if (m := _pat.match(os.path.basename(p))))
print(f"analyzing force F={FORCE}: {len(JOBS)} proteins\n")

for name, h5 in JOBS:
    geom = an.geometry_scales(h5)           # whole-protein η (all nodes)
    chains, _ = an.load_node_meta(h5)
    chain = Counter(chains).most_common(1)[0][0]   # largest chain for pairs
    nodes = an.chain_nodes(h5, chain)
    ti, tj = np.triu_indices(len(nodes), k=1)
    pairs = np.stack([nodes[ti], nodes[tj]], axis=1)
    print(f"[{name}] {len(nodes)} nodes ({len(chains)} chains, representative={chain}) "
          f"{len(pairs)} pairs | "
          f"L_par={geom.L_parallel:.2f} L_perp={geom.L_perp:.2f} "
          f"eta={geom.eta_shape:.2f}", flush=True)
    out = an.correlation_scales_loading(h5, t_load=50.0, n_frames=10,
                                        pairs=pairs, node_subset=None,
                                        out=f"data/figures/scale_{name}_F{FORCE}")
    print(f"[{name}] -> {out}", flush=True)
print("ALL DONE", flush=True)
