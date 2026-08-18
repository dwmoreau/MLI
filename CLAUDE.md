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

## Cross-platform compatibility (required for all edits)

Users run this on Windows, macOS, and Linux. Development happens on macOS, so Windows
breakage is not caught by simply running the code. **All edits must be cross-platform.**
The end-user path — `pip install mlindex` → `mlindex.download_models` → `mlindex.run` —
is the one that matters most; it must never regress on Windows.

Rules, each of which corresponds to a bug this project has actually shipped or narrowly avoided:

- **Never build paths by string concatenation or with `/` separators.** Use `os.path.join`,
  `pathlib`, or `importlib.resources.files(...).joinpath(...)`. Never `'/'.join(parts)`.
- **Never assume a path shape.** Do not reconstruct a directory by appending known
  components to a path a user supplied, and never walk up with `.parent.parent` to undo
  that. `MLINDEX_MODELS_DIR` was broken this way for every path not ending in
  `mlindex/models`, and was 100% broken on Windows, where `D:\models` resolved to `D:\`.
  Pass the real directory through instead — see `mlindex/paths.py`.
- **Always pass `encoding=` when opening or writing text.** Windows defaults to the locale
  codepage, not UTF-8. Anything non-ASCII (`Å`, `°`, `θ`, `²`) fails to write on cp1251 and
  cp932, and superscripts fail even on cp1252. `_write_results` in `run.py` passes
  `encoding='utf-8'` for exactly this reason.
- **Keep everything argparse prints pure ASCII** — help strings, descriptions, metavars.
  Redirecting or piping `--help` on Windows encodes through the locale codepage and raises
  `UnicodeEncodeError`. Write `Angstrom`, `2-theta`, `1/Angstrom^2`; not `Å`, `2θ`, `Å⁻²`.
  Non-ASCII is fine in file output as long as the encoding is explicit.
- **Pass `newline=''` to `open()` when using the `csv` module** (`IOManagers.write_params`),
  or Windows produces `\r\r\n` and blank rows.
- **Keep multiprocessing spawn-safe.** Windows and macOS both use the `spawn` start method:
  worker targets must be module-level functions and every argument must be picklable. See
  `_mp_worker_fn` in `MPOptimizer.py`. Never rely on `fork` semantics or inherited state.
- **Do not use Unix-only modules** (`fcntl`, `pwd`, `grp`, `resource`, `os.fork`) or shell
  out to Unix tools in shipped code.
- **Avoid symlinks.** `snapshot_download` is called with `local_dir=`, which copies files;
  the cache mode would create symlinks and require Developer Mode or admin rights on Windows.
- **`shutil.rmtree` over a git clone fails on Windows** — git marks `.git/objects` read-only,
  so cleanup raises `PermissionError`. This affects the legacy `--source github` path in
  `download_models.py`, which is still unfixed and slated for removal in 0.2.0.

Not currently supported on Windows: MPI mode (`--mpi`), model training, and dataset
generation. `cctbx-base` ships a `win_amd64` wheel but no `win_arm64`, so Windows-on-ARM
cannot install at all.

## Optional dependencies

Dataset generation requires: `pyarrow`, `openpyxl`, `cctbx`, `tqdm`

Model training additionally requires: `skl2onnx`, `keras`, `torch`, `torchvision`, `lightgbm`

`lightgbm` is only needed to fit the ranking-objective variant of the FOM combiner
(`mlindex/model_training/FomCombiner.py`). It is imported inside the fit path, so inference and
model loading work without it.

## The ML-FOM project — its record is untracked, and this is the only pointer to it

There is a long-running research project on top of this codebase — **ML-FOM**, replacing the de
Wolff M20 figure of merit with a learned one. Its entire working record lives under `docs/`, which
`.gitignore` excludes on purpose: those are working notes and they are not for GitHub.

**So `docs/` is not in this repository.** A fresh clone has none of it, `git pull` on another
machine does not deliver it, and `git clean -xdf` deletes it. This section exists because it is
the only trace of the project that a clone does carry. If `docs/fom/` is missing, the record has
to be restored from a backup or from NERSC before doing any FOM work:

```bash
rsync -a ~/mli-record-backup/<YYYY-MM-DD>/docs/ docs/       # local dated snapshot
docs/sync_record.sh snapshot                                # take a new one
```

When `docs/` is present, read `docs/fom/PROTOCOL.md` first — it defines how a session is
conducted and its standing rules override any other instruction in that project — then
`docs/fom/README.md`, `docs/fom/STATUS.md`, and the relevant handoff under `docs/fom/handoffs/`.

Three things about that work that affect ordinary edits here:

- **FOM work goes on branch `fom`; general correctness fixes go on `main`** and are merged in. The
  test is whether the change would matter to someone who never touches figures of merit.
- **Perlmutter cannot push to `origin`** (`Permission denied (publickey)`), so a fix recorded as
  done may exist on one machine only. Check with `git log origin/main --oneline -- <path>`.
- **Never `git add -A`.** `mlindex/models/` is git-lfs and often mid-retrain, and
  `mlindex/characterization/` and `mlindex/data/generated_datasets/` are regenerable run output.
  Stage named paths.
