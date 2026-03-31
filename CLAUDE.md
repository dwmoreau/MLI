# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

MLINDEX is a powder diffraction indexing program. Given a list of observed diffraction peaks, it returns candidate unit cells ranked by the de Wolff M20 figure of merit. ML models initialize candidate unit cells for each Bravais lattice; candidates are then refined by least-squares optimization. The paper describing the methods was submitted to the Journal of Applied Crystallography.

## Installation

```bash
pip install .
```

ML model files (~1 GB) are stored in `mlindex/models/` via git-lfs. After cloning, retrieve them with:
```bash
git lfs fetch --all && git lfs checkout
```

## Running the indexer

### ML-based indexer (primary)
Accepts either a GSAS-II `.pkslst` file (requires wavelength) or a numpy `.npy` array of q² values (1/Å²):

```bash
# Serial (1 rank)
python -m mlindex.command_line.run --peak-file peaks.npy

# Parallel (exactly 6 ranks — required for full Bravais lattice coverage)
mpiexec -n 6 mlindex.run --peak-file peaks.pkslst --wavelength 0.413128
```

Outputs `indexing_results.json` and prints top-20 candidates ranked by M20.

### Analytical indexer (lightweight alternative, no ML models needed)
```bash
python -m mlindex.command_line.run_analytical --peak-file peaks.npy
```
Covers high-symmetry lattices (cF, cI, cP, hP, hR, tI, tP, oC, oF, oI, oP) serially.

## Key architectural concepts

### Internal representations
- **q²**: The input unit. q² = (2 sin θ / λ)² = 1/d². Peak lists are always in q² (Å⁻²).
- **xnn**: The primary internal metric tensor representation (6-component vector). All optimization happens in xnn space. Converted to/from conventional unit cell parameters by `mlindex/utilities/UnitCellTools.py`.
- **Bravais lattice split groups**: Each Bravais lattice is subdivided into "split groups" (e.g., `cF_0`, `mP_0_01`) based on unit-cell axis ordering and extinction groups. These are the granularity at which random forest regressors and integral filters are trained.

### Execution flow (`mlindex/command_line/run.py`)
1. Load peak list → `q2_obs` array (up to 20 peaks used)
2. For each of 14 Bravais lattices, an `OptimizerManager` generates unit cell candidates using three ML components, then refines them with Gauss-Newton optimization and scores by M20/Minfo
3. Rank 0 collects all results, sorts by M20, writes JSON

### MPI parallelism
With 6 ranks: cubic (cF/cI/cP) and triclinic/monoclinic lattices run serially on dedicated ranks; orthorhombic lattices (oC/oF/oI/oP) run in parallel across all ranks. The mapping is hardcoded in `run.py`. With 1 rank: everything runs serially.

### Model training pipeline (`mlindex/model_training/Wrapper.py`)
`Wrapper` is the central training orchestrator. The three ML components trained per split group:

1. **RandomGenerator** (`RandomGenerator.py`): Random forest that predicts unit cell volume from the observed peak list. Used to generate random candidate unit cells.
2. **MITemplates** (`MITemplates.py`): Miller index template library + HistGradientBoosting calibrator. Templates are sets of hkl assignments sampled from training data; the calibrator scores how likely a template is to converge to the correct unit cell.
3. **IntegralFilter** (`IntegralFilter.py`): Neural network (PyTorch/ONNX) that filters and ranks candidates. Quantized for inference and stored as `.onnx` files.

All trained models are saved under `mlindex/models/{tag}/` (e.g., `mlindex/models/cubic_1/`).

### Dataset generation (`mlindex/dataset_generation/GenerateDataset.py`)
- Source databases: CSD and COD parquet files at `/global/cfs/cdirs/m4064/dwmoreau/`
  - `unique_csd_entries_cnrs_removed.parquet`
  - `unique_cod_entries_not_in_csd_cnrs_removed.parquet`
- Uses cctbx to simulate powder patterns from CIF structures; requires cctbx in the environment
- Runs MPI-parallel; each rank processes a subset of entries and saves `chunk_NN.parquet`; rank 0 combines into `dataset_{BL}.parquet`
- Output goes to `../data/generated_datasets/` relative to where the script is run

### Optimization (`mlindex/optimization/`)
- `Optimizer.py`: `OptimizerManager` (rank-0) + `OptimizerWorker` (other ranks). Loads the three ML models via `Wrapper.setup_from_tag()`, generates candidates, runs refinement loops.
- `CandidateOptLoss.py`: Gauss-Newton least-squares update for xnn given hkl assignments.
- `AnalyticOptimizer.py`: Geometry-based candidate generation (no ML), used by the analytical CLI.
- `CandidateValidation.py`: Post-optimization spacegroup assignment based on systematic absences.

### Key utilities (`mlindex/utilities/`)
- `UnitCellTools.py`: Converts between xnn, unit cell (a,b,c,α,β,γ), and reciprocal-space forms. `fix_unphysical()` enforces physical constraints on xnn.
- `Q2Calculator.py`: Computes q² for given hkl and metric tensor.
- `FigureOfMerits.py`: M20 (de Wolff) and Minfo (Taupin) implementations.
- `SpaceGroups.py`: Systematic absence rules and hkl reference sets per spacegroup.
- `Reindexing.py`: Selling reduction, monoclinic standardization, and axis permutation for canonical unit-cell orientation.

## Optional dependencies

Dataset generation requires: `pyarrow`, `openpyxl`, `cctbx`, `tqdm`

Model training additionally requires: `skl2onnx`, `keras`, `torch`, `torchvision`
