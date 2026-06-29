# -*- coding: utf-8 -*-
"""Loading-phase multitime correlation-vs-distance figures (the HTML-deck style:
binned-mean correlation vs distance, one line per time frame).

Only the loading half (0-50 ps) is used; the 60/80/100 ps frames fell in the
unloading half and made the correlation length jump back, so they are excluded.
realtime=True -> x-axis is each pair's actual distance at that time.

Reads sim_<id>_F<force>.h5 and tags the output with the same force.
Run:  .venv/bin/python regen_multitime.py [force]      # default force 0.01
"""
import glob
import os
import re
import sys
from collections import Counter

import numpy as np

from memprotein import analysis as an

# Pick the peak force to plot (matches sim_<id>_F<force>.h5). Default 0.01.
FORCE = sys.argv[1] if len(sys.argv) > 1 else "0.01"
_pat = re.compile(rf"sim_(.+)_F{re.escape(FORCE)}\.h5$")
JOBS = sorted((m.group(1), p) for p in glob.glob(f"data/results/sim_*_F{FORCE}.h5")
              if (m := _pat.match(os.path.basename(p))))
print(f"force F={FORCE}: {len(JOBS)} proteins\n")

for name, h5 in JOBS:
    chains, _ = an.load_node_meta(h5)
    chain = Counter(chains).most_common(1)[0][0]      # representative subunit
    nodes = an.chain_nodes(h5, chain)
    ti, tj = np.triu_indices(len(nodes), k=1)
    pairs = np.stack([nodes[ti], nodes[tj]], axis=1)
    print(f"[{name}] chain {chain}: {len(nodes)} nodes, {len(pairs)} pairs", flush=True)
    out = an.binned_multitime(h5, times=(0, 10, 20, 30, 40, 50),
                              pairs=pairs, realtime=True,
                              out=f"data/results/multitime_{name}_F{FORCE}")
    print(f"[{name}] -> {out}", flush=True)
print("ALL DONE", flush=True)
