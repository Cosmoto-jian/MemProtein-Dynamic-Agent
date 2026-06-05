# MemProtein-Dynamic-Agent

Membrane-protein force-deformation simulation + residue co-motion analysis.

Given a membrane-protein structure from the [OPM database](https://opm.phar.umich.edu),
simulate its dynamic deformation under membrane tension (radial stretching), then
analyse how residues move in a coordinated way (instantaneous correlation
analysis). Verified on Piezo1 (6b3r), rhodopsin (1u19), 6lod, and others.

> 中文文档见 [README.md](README.md).

## Layout

```
memprotein/          core package (importable by code / an agent)
  io.py              read/write input files (MODEL/targetNode/mass/evector)
  preprocess.py      OPM PDB + TM-segment text -> model inputs
  simulate.py        vector-form finite-element (VFIFE) dynamics
  analysis.py        instantaneous correlation analysis + four figures
  pipeline.py        end-to-end orchestration
cli.py               command-line entry (run / preprocess / simulate / analyze)
viz/animate.py       PyVista 3-D deformation animation (needs a display)
data/
  raw/               original inputs you provide (.pdb and _tm.txt)
  inputs/            generated model files (auto, git-ignored)
  results/           simulation .h5 + analysis figures (auto, git-ignored)
clean.sh             remove all generated products
```

## Install

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```
(Python 3.10+, tested on 3.14. `pyvista`/`vtk` are for visualization only.)

## Prepare inputs (two files per protein)

1. **Download the OPM PDB** into `data/raw/` (replace `6lod` with your 4-letter PDB id):
   ```bash
   curl -L -o data/raw/6lod.pdb "https://storage.googleapis.com/opm-assets/pdb/6lod.pdb"
   ```
2. **Copy the TM-segment text**: on the protein's OPM page find "Transmembrane
   Secondary structure segments" and copy the per-chain lines
   (`A - TM segments: 1(33-64), 2(...)`) into `data/raw/6lod_tm.txt`.

## Run

End-to-end (preprocess -> simulate -> 4 figures):
```bash
.venv/bin/python cli.py run --pdb data/raw/6lod.pdb --tm data/raw/6lod_tm.txt
```

Individual stages:
```bash
.venv/bin/python cli.py preprocess --pdb data/raw/6lod.pdb --tm data/raw/6lod_tm.txt
.venv/bin/python cli.py simulate --ET 100 --Fmax 0.1
.venv/bin/python cli.py analyze --kind all
```

As a library:
```python
from memprotein.pipeline import run
run("data/raw/6lod.pdb", "data/raw/6lod_tm.txt")
```

Outputs land in `data/results/`: `simulation_data.h5` plus
`instant_corr.png` / `distance_corr.png` / `anchor_corr.png` / `anchor_stack.png`.

3-D animation (needs a display): `.venv/bin/python viz/animate.py`
Clean generated products: `bash clean.sh`

## Method, in brief

- **Coarse-graining**: one bead (the Cα) per amino-acid residue.
- **Elastic network**: residue pairs within 10 Å are connected by springs.
- **Loading**: membrane residues (from the OPM TM segments) are pulled radially
  outward in the membrane plane, mimicking membrane tension.
- **Integration**: explicit central-difference (vectorized, ~26× faster than the
  original element-by-element loop).
- **Analysis**: per frame and per residue pair, measure instantaneous motion-
  direction correlation — C^Z (out-of-plane, sign) and C^XY (in-plane, cosine) —
  after per-frame Kabsch rigid-body alignment to remove spurious whole-protein
  translation/rotation. See how coordination changes with distance and time.
```
