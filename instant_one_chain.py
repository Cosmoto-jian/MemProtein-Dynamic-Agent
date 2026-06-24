#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single-chain instantaneous correlation: restrict the instant_corr figures to
residue pairs within ONE chain.

Useful for symmetric multimers (e.g. the 6b3r trimer A/C/E): look at the
coordination inside a single subunit, without mixing chains.

Chain membership is read straight from the result file (which now stores each
node's chain + residue number), so node indices can never get out of sync.

Edit the CONFIG block below, then:
    .venv/bin/python instant_one_chain.py
"""

import numpy as np

from memprotein import analysis as an

# ============================================================
#  配置 —— 直接改这里(CONFIG)
# ============================================================
H5       = "data/results/simulation_data.h5"   # 仿真结果文件
CHAIN    = "R"                                  # 只分析这一条链
TIMES    = (50,)                                # 看哪些时刻(ps)
N_SAMPLE = 0                                    # 采样多少对;0 = 全部对(单链内全画)
DMAX     = 0.0                                  # 只看多少 nm 以内(0 = 全部)
OUT      = "data/results/instant_corr_chain"
# ============================================================


def main() -> None:
    nodes = an.chain_nodes(H5, CHAIN)   # 0-based 节点序号,直接从 h5 读取
    print(f"链 {CHAIN}: {len(nodes)} 个节点")

    if N_SAMPLE <= 0:
        ti, tj = np.triu_indices(len(nodes), k=1)
        pairs = np.stack([nodes[ti], nodes[tj]], axis=1)
        print(f"全部对: {len(pairs)} 对")
        kw = dict(pairs=pairs)
    else:
        kw = dict(n_sample=N_SAMPLE, node_subset=nodes)

    print("散点:", an.correlation_vs_distance(H5, times=TIMES, dmax=DMAX,
                                            out=f"{OUT}{CHAIN}", **kw))
    print("密度:", an.correlation_hexbin(H5, time_ps=TIMES[0],
                                        out=f"{OUT}{CHAIN}_hexbin", **kw))
    print("均线:", an.correlation_binned(H5, time_ps=TIMES[0],
                                        out=f"{OUT}{CHAIN}_binned", **kw))


if __name__ == "__main__":
    main()
