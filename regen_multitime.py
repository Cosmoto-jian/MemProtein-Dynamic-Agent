# -*- coding: utf-8 -*-
"""Regenerate multitime correlation figures with realtime=True so the x-axis is
the CURRENT inter-residue distance (at each time), not the t=0 distance."""
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
    print(f"[{name}] chain {chain}: {len(nodes)} nodes, {len(pairs)} pairs", flush=True)
    out = an.binned_multitime(h5, times=(0, 20, 40, 60, 80, 100),
                              pairs=pairs, realtime=True,
                              out=f"data/results/multitime_{name}_rt")
    print(f"[{name}] -> {out}", flush=True)
print("ALL DONE", flush=True)
