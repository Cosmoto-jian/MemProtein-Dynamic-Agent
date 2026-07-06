# -*- coding: utf-8 -*-
"""Tile the 36 binned-mean multitime figures into one overview image.

Layout: each protein is a vertical strip of 3 (force 0.01 / 0.05 / 0.1 top->bottom).
The 12 strips are arranged in 2 banks of 6 proteins (columns) so the result is not
absurdly wide. Output: data/figures/montage_multitime.png
"""
import glob
import os
import re

import matplotlib.pyplot as plt

FIG = "data/figures"
FORCES = ["0.01", "0.05", "0.1"]

pat = re.compile(r"multitime_(.+)_F0\.01\.png$")
proteins = sorted(pat.match(os.path.basename(p)).group(1)
                  for p in glob.glob(f"{FIG}/multitime_*_F0.01.png"))
print(f"{len(proteins)} proteins: {proteins}")

PER_BANK = 6
banks = [proteins[i:i + PER_BANK] for i in range(0, len(proteins), PER_BANK)]
nrows = len(banks) * len(FORCES)            # 2 banks * 3 forces = 6
ncols = PER_BANK                            # 6

fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.2, nrows * 3.2))
for ax in axes.ravel():
    ax.axis("off")

for b, bank in enumerate(banks):
    for fi, F in enumerate(FORCES):
        r = b * len(FORCES) + fi
        for c, pid in enumerate(bank):
            ax = axes[r, c]
            img = plt.imread(f"{FIG}/multitime_{pid}_F{F}.png")
            ax.imshow(img)
            ax.set_title(f"{pid}   F={F} pN", fontsize=11, fontweight="bold")

fig.suptitle("Binned mean correlation vs inter-residue distance  "
             "(per protein: rows = peak force 0.01/0.05/0.1 pN; "
             "lines = loading time 0-50 ps)",
             fontsize=14, y=0.997)
fig.tight_layout(rect=(0, 0, 1, 0.99))
out = f"{FIG}/montage_multitime.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"-> {out}")
