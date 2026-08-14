# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

MLINDEX is a powder diffraction indexing program. Given a list of observed diffraction peaks, it returns candidate unit cells ranked by the de Wolff M20 figure of merit. ML models initialize candidate unit cells for each Bravais lattice; candidates are then refined by least-squares optimization.

## Installation

```bash
pip install .
```

ML model files (~545 MB) are published on the Hugging Face Hub at
[`dwmoreau/mlindex-models`](https://huggingface.co/dwmoreau/mlindex-models), and `mlindex.download_models`
fetches them from there with `huggingface_hub` (no git or git-lfs needed). Each release pins a model
revision via `model_revision` in `mlindex/model_metadata.json`.

The same files are also still tracked in `mlindex/models/` via git-lfs for development. After cloning:
```bash
git lfs pull
```

Models are located at runtime by `_resolve_models_dir()` in `mlindex/optimization/UtilitiesOptimizer.py`,
which searches, in order: `MLINDEX_MODELS_DIR` (used as-is), `$XDG_DATA_HOME/mlindex/models`, then the
package's own `mlindex/models/`. A models directory is the one that *directly* contains `cubic_1/`,
`hexagonal_1/`, ... — the helpers live in `mlindex/paths.py` and are shared with the downloader.

Note that the wheel always ships a *partial* `mlindex/models/` tree: the `models/*/data/hkl_ref_*.npy`
package-data glob, which `AnalyticOptimizer` and `CreatePeakList` read via `importlib.resources`. That is
why the "do we have models?" probe checks for `cubic_1/integral_filter/` rather than for `models/`.

## Running the indexer

### ML-based indexer (primary)
Accepts either a GSAS-II `.pkslst` file (requires wavelength) or a numpy `.npy` array of q² values (1/Å²):

```bash
# Serial (1 process, no workers spawned)
python -m mlindex.command_line.run --peak-file peaks.npy

# Multiprocessing with N processes (recommended over MPI for local runs)
python -m mlindex.command_line.run --peak-file peaks.npy --nproc 4

# MPI mode (exactly 6 ranks — requires mpiexec, legacy parallel mode)
mpiexec -n 6 python -m mlindex.command_line.run --peak-file peaks.pkslst --wavelength 0.413128 --mpi

# Zero-point error correction (requires wavelength)
python -m mlindex.command_line.run --peak-file peaks.pkslst --wavelength 0.413128 --zero-error
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

### Parallelism

**Multiprocessing mode** (`--nproc N`): Worker processes are spawned once at startup and kept alive across all 14 Bravais lattices to avoid re-import overhead. The main process acts as manager for every lattice. Uses `multiprocessing.Queue` for communication; `mpi4py` is never imported. Supports arbitrary N; `--nproc 1` runs fully serially with no workers spawned.

**MPI mode** (`--mpi`): With 6 ranks, cubic (cF/cI/cP) and triclinic/monoclinic lattices run serially on dedicated ranks; orthorhombic lattices (oC/oF/oI/oP) run in parallel across all ranks. The mapping is hardcoded in `run.py`. With 1 rank, everything runs serially. Requires `--mpi` flag; `mpi4py` is only imported when this flag is passed.

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
- `Candidates.py`: `Candidates` class — runs the refinement loop, computes M20/Minfo, handles reindexing and spacegroup assignment for a single rank's candidate set.
- `MPIOptimizer.py`: `OptimizerBase`, `OptimizerWorker`, `OptimizerManager` — MPI-based parallel optimizer. Contains extracted helpers `_run_loop`, `_generate_candidates_xnn`, `_downsample_computation` that are reused by the MP subclasses.
- `MPOptimizer.py`: `MPOptimizerManager`, `MPOptimizerWorker` — multiprocessing subclasses that override only the communication methods, replacing MPI calls with `multiprocessing.Queue`. Also contains `setup_mp_optimizers`, `run_mp_bl`, `shutdown_mp_workers`.
- `Optimizer.py`: Backward-compatibility re-export shim — imports from `Candidates` and `MPIOptimizer`. Existing notebooks and scripts that import from `Optimizer` continue to work.
- `UtilitiesOptimizer.py`: Factory functions (`get_cubic_optimizer`, etc.) and `get_optimizers`. All factory functions accept an `optimizer_class=None` parameter — pass `MPOptimizerManager` to use multiprocessing instead of MPI.
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
