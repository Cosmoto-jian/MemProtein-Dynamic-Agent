
## Installation & Configuration

### Environment Requirements

- Python 3.7+
- Operating systems: Windows / Linux / macOS

### Install Dependencies

```bash
# Clone or download the project
cd VEND

# Install dependencies
pip install -r requirements.txt
```

**Core dependencies:**
- `numpy` — numerical computation
- `scipy` — scientific computing
- `matplotlib` — data visualization
- `h5py` — HDF5 file I/O
- `pyvista` — 3D visualization

## Usage Instructions

The complete workflow consists of four stages: **preprocessing** generates the input → the **main simulation program** solves the dynamics → **visualization** inspects structural deformation → **post-processing analysis** extracts domain-level motion patterns and functional clusters.

### 1. Data Preprocessing (Pre_process)

The three scripts under `Pre_process/` generate the input files required in `Raw/` from raw protein structure data. They have data dependencies on each other and should be run in the following order:

1. **`CaCG.py`** — coarse-grains an all-atom protein PDB file from the OPM database into a Cα-residue representation and outputs `Raw/piezo-cg.pdb`. This forms the geometric basis of the whole coarse-grained model.
2. **`Connector.py`** — reads `Raw/piezo-cg.pdb`, builds the Kirchhoff matrix based on a distance cutoff (default `cutoff = 32 Å`) to obtain the element connectivity of the elastic network model, and writes out `Raw/MODEL.txt` (node coordinates + element connectivity).
3. **`MemRNum.py`** — extracts, from the OPM-aligned all-atom PDB, the residue numbers (after coarse-graining) of chains A/C/E that lie within the membrane according to predefined transmembrane segment ranges. These residue numbers are eventually aggregated into `Raw/targetNode.txt`, the target node list for external force loading.

> All three scripts currently use hard-coded local PDB paths (the `pdb_file` variable inside each script). Update them to your local paths before running.

### 2. Preparing Input Files

Make sure the `Raw/` directory contains the following files:

| File | Description | Format |
|------|-------------|--------|
| `MODEL.txt` | Node coordinates, element connectivity, boundary constraints | Text |
| `targetNode.txt` | List of loaded node IDs | Text |
| `mass.txt` | Mass of each node | Text |
| `evector.mat` | Force direction vectors | MATLAB |

**Example `MODEL.txt` format:**
```
4554                    # number of nodes
1  -87.749  -33.498  -98.366   # NodeID  X  Y  Z (Å)
2  -84.749  -35.791  -97.785
...
8123                    # number of elements
1  1  2                 # ElementID  Node1  Node2
...
456                     # number of constraints
1  1  1  1              # NodeID  fixX  fixY  fixZ (1 = fixed)
...
```

### 3. Running the Simulation

#### Basic usage

```bash
python main.py
```

#### Advanced parameter configuration

```bash
python main.py \
  --ET 100.0 \           # total simulation time (ps)
  --h 0.1 \              # time step (ps)
  --zeta 0.01 \          # damping coefficient
  --E 1000.0 \           # Young's modulus (pN/nm²)
  --A 0.01 \             # cross-sectional area (nm²)
  --Fmax 0.1 \           # peak force (pN/node)
  --ramp_t0 0.01 \       # loading start time (ps)
  --ramp_t1 50.0 \       # peak time (ps)
  --unload_t1 100.0 \    # unloading end time (ps)
  --out results.h5 \     # output filename
  --no-gui \             # disable interactive preview
  --save-internal \      # save internal force data
  --compress 4           # compression level (0-9)
```

#### Headless mode (suitable for batch computation)

```bash
python main.py --no-gui --ET 200 --Fmax 0.2
```

### 4. Result Visualization

After the simulation finishes, view the results with the built-in player:

```bash
# Automatically find the latest h5 file
python Viz/animate.py

# Specify a file
python Viz/animate.py simulation_data.h5

# Set the display interval (one frame every 50 steps)
python Viz/animate.py --interval 50

# View only a specific frame
python Viz/animate.py --frame 100
```

**Visualization features:**
- Frame-by-frame playback of structural deformation
- Display of time and step information
- Interactive 3D rotation and zoom
- Clean rendering of nodes and elements

## Output Data

Simulation results are saved in HDF5 format (`simulation_data.h5`) and include the following datasets:

### Global data

| Dataset | Description | Unit |
|---------|-------------|------|
| `time_steps` | Time series | ps |
| `applied_forces` | Per-node external force magnitude | pN |
| `applied_forces_total` | Total external force | pN |
| `extensions` | Mean extension | nm |
| `internal_forces` | Total internal force (projection) | pN |
| `initial_nodes` | Initial node coordinates | nm |
| `final_nodes` | Final node coordinates | nm |
| `elements` | Element connectivity table | - |
| `target_nodes` | Loaded node IDs | - |
| `evector` | Force direction vectors | - |

### Time-series data (`timeseries/`)

| Dataset | Shape | Unit |
|---------|-------|------|
| `node_coords` | (n_steps, n_nodes, 3) | m |
| `element_connectivity` | (n_steps, n_elements, 2) | - |
| `time` | (n_steps,) | s |

### Step-by-step detailed data (`step_data/step_XXXX/`)

| Dataset | Description |
|---------|-------------|
| `node_coordinates` | Node coordinates |
| `element_forces` | Element axial forces |
| `node_forces` | Per-node external force vectors |
| `node_forces_internal` | Per-node internal force vectors |

### Reading example

```python
import h5py
import numpy as np

with h5py.File('simulation_data.h5', 'r') as f:
    # Read time series
    time = f['time_steps'][:]
    force = f['applied_forces'][:]
    extension = f['extensions'][:]

    # Read node trajectories
    coords = f['timeseries/node_coords'][:]  # (n_steps, n_nodes, 3)

    # Read a specific step
    step = f['step_data/step_0100']
    node_coords = step['node_coordinates'][:]
    element_forces = step['element_forces'][:]
```

## Physical Unit System

The program uses the piconewton–nanometer–picosecond unit system, well-suited for biomolecular scales:

| Quantity | Unit | Notes |
|----------|------|-------|
| Length | nm (nanometer) | 1 nm = 10 Å |
| Force | pN (piconewton) | 1 pN = 10⁻¹² N |
| Time | ps (picosecond) | 1 ps = 10⁻¹² s |
| Mass | Da (Dalton) | 1 Da ≈ 1.66 × 10⁻²⁷ kg |
| Young's modulus | pN/nm² | - |
| Cross-sectional area | nm² | - |

**Unit-conversion notes:**
- 1 pN·ps²/nm = 1 Da
- Input coordinate unit: Å → automatically converted to nm by the program
- Output coordinate unit: stored in meters in HDF5 (for SI consistency); convert when reading

## Typical Application Case

### Mechanical response of the Piezo1 ion channel

```bash
python main.py \
  --ET 100.0 \
  --h 0.1 \
  --Fmax 0.1 \
  --E 1000.0 \
  --A 0.01 \
  --ramp_t0 0.01 \
  --ramp_t1 50.0 \
  --unload_t1 100.0
```

**Model parameters:**
- Nodes: 4554
- Elements: 8123
- Loaded nodes: ~1520
- Total mass: 881.1 kDa
- Average nodal mass: 193.6 Da

## Notes

1. **Time-step stability**: explicit integration must satisfy the CFL condition; `h ≤ 0.1 ps` is recommended.
2. **Memory usage**: for large-scale models (> 5000 nodes), enabling compression options is recommended.
3. **Convergence check**: inspect the extension–force curve for smoothness.
4. **Element failure**: an element automatically fails (Young's modulus set to zero) when its length exceeds twice its initial length.
5. **Boundary conditions**: fixed constraints are enforced strictly via boundary conditions.

## FAQ

**Q: The simulation is slow — what can I do?**
A: Increase the time step, reduce output frequency, or run in headless (no-GUI) mode.

**Q: How do I change material parameters?**
A: Use the command-line flags `--E` and `--A` to adjust Young's modulus and cross-sectional area.

**Q: The visualization window does not show up.**
A: Pass `--no-gui` to skip the interactive preview, or verify your PyVista installation.

**Q: How do I extract the trajectory of a specific node?**
A: Read `timeseries/node_coords` through the HDF5 interface.


## Theoretical Background

For detailed theoretical derivations, see:
- Ting, E.C., et al. (2004). "Fundamentals of a Vector Form Intrinsic Finite Element"
- Vector-form finite element theory: principal-axis vectors track element spatial motion and remove rigid-body effects
- Elastic network model: coarse-grained representation (cutoff = 10 Å)
- Central difference method: a conditionally stable explicit time-integration scheme

## License

This project is released under the MIT License — see the LICENSE file for details.

---

**Last updated:** 20260603