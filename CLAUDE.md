# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

MLINDEX is a powder diffraction indexing program. Given a list of observed diffraction peaks, it returns candidate unit cells ranked by a learned ranker. ML models initialize candidate unit cells for each Bravais lattice; candidates are refined by least-squares optimization, cut at M20 3.5, and the survivors are scored by a gradient-boosted classifier (`mlindex/models/ranker_1`, ONNX) whose calibrated probability orders the printed list. Without the ranker the list is ordered by `M_sym`, and the output says so; the analytical indexer always ranks by `M_sym`.

## How to make changes (required for all edits)

These rules outrank convenience. Each was written down because this repository has paid for
ignoring it.

**Fix it upstream, not downstream.** When a value is wrong where it is used, change where it comes
from. Reaching in from outside to correct another object's state is a band-aid: it leaves the
original defect in place for the next caller, and the correction has to be remembered at every
future call site. Prefer the smaller, higher change to the larger, lower one.

**Reuse what exists; no redundancy.** One behaviour has one implementation and one call site.
Before adding a function, look for the one that already does it. If a change appears to need the
same call repeated in several subclass overrides, it belongs at the point those overrides share --
a call added to three of four overrides is a bug that has not happened yet, and this repository
has shipped exactly that twice.

**A function that draws random numbers takes its generator as a required argument.** No `rng=None`
default: a default has to invent an unseeded generator, so a caller who forgets one gets a result
that cannot be reproduced, and no error.

**Reusable code lives in `mlindex/utilities/`, and that is the first place to look.** Search there
before writing anything; if what you are about to write would be useful to a second caller, it goes
there rather than on the class or module that happens to need it first. `validate_candidate_known_bl`
(`optimization/CandidateValidation.py`) is the standing example of the cost: it re-implements
`utilities/UnitCellTools.get_partial_unit_cell` inline as an if/elif chain, and hardcodes reindexing
matrices that `utilities/Reindexing.py` is the home for. A second implementation does not announce
itself when it drifts from the first.

## Installation

```bash
pip install .
```

ML model files (~472 MB, 741 files) are published on the Hugging Face Hub at
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
why the "do we have models?" probe checks for `cubic_1/abnn/` rather than for `models/`.

## Running the indexer

### ML-based indexer (primary)
Accepts either a GSAS-II `.pkslst` file (requires wavelength) or a numpy `.npy` array of peak positions. **`--peak-units` defaults to d-spacing (Å)**; pass `--peak-units q2` for an array of q² values (1/Å²), or `q`, or `2theta` with a wavelength. A q² array read as d-spacings still runs, and prints cells of a few Å or less:

```bash
# Serial (1 process, no workers spawned)
python -m mlindex.command_line.run --peak-file peaks.npy --peak-units q2

# Multiprocessing with N processes (recommended over MPI for local runs)
python -m mlindex.command_line.run --peak-file peaks.npy --peak-units q2 --nproc 4

# MPI mode (exactly 6 ranks — requires mpiexec, legacy parallel mode)
mpiexec -n 6 python -m mlindex.command_line.run --peak-file peaks.pkslst --wavelength 0.413128 --mpi

# Zero-point error correction (requires wavelength)
python -m mlindex.command_line.run --peak-file peaks.pkslst --wavelength 0.413128 --zero-error
```

Outputs `indexing_results.json` and `.txt`, and prints the top 20. Each lattice hands the ranker its 20 best candidates (by the ranker's score; by M20 under the `M_sym` fallback); a row that repeats a higher-ranked cell of the same lattice in another basis is dropped; each cell is printed in one standard setting (`Reindexing.printed_setting`: mP and mC reduced with beta >= 90, mC in the C setting, aP as its Niggli cell).

### Analytical indexer (lightweight alternative, no ML models needed)
```bash
python -m mlindex.command_line.run_analytical --peak-file peaks.npy --peak-units q2
```
Covers high-symmetry lattices (cF, cI, cP, hP, hR, tI, tP, oC, oF, oI, oP) serially.

## Key architectural concepts

### Internal representations
- **q²**: The internal unit. q² = (2 sin θ / λ)² = 1/d². Peak lists are converted to q² (Å⁻²) on input (`run._convert_to_q2`).
- **xnn**: The primary internal metric tensor representation (6-component vector). All optimization happens in xnn space. Converted to/from conventional unit cell parameters by `mlindex/utilities/UnitCellTools.py`.
- **Bravais lattice split groups**: Each Bravais lattice is subdivided into "split groups" (e.g., `cF_0`, `mP_0_01`) based on unit-cell axis ordering and extinction groups. These are the granularity at which random forest regressors and ABNNs are trained.

### Execution flow (`mlindex/command_line/run.py`)
1. Load peak list → `q2_obs` array (up to 20 peaks used)
2. For each of 14 Bravais lattices, an `OptimizerManager` generates unit cell candidates using three ML components, refines them with Gauss-Newton optimization, cuts them at M20 3.5 (`Candidates.PRUNE_M20_THRESHOLD`), reduces monoclinic and triclinic cells, assigns an extinction group by M20, and computes the ranker's inputs
3. The collected pools are scored by the ranker (`utilities/Ranker.py`), shortlisted 20 per lattice, de-duplicated, put in their printed setting, and written

### Parallelism

**Multiprocessing mode** (`--nproc N`): `run.allocate_lattice_groups` deals the lattices into groups by a measured cost table, and each group runs at the same time with its own manager and refinement workers (`MPOptimizer.setup_lattice_groups`, `run_lattice_groups`). The first group runs in the calling process. Uses `multiprocessing.Queue` for communication; `mpi4py` is never imported. Supports arbitrary N; `--nproc 1` runs fully serially with no workers spawned. The same `--nproc` always reproduces; a different `--nproc` may give a different list.

**MPI mode** (`--mpi`): With 6 ranks, cubic (cF/cI/cP) and triclinic/monoclinic lattices run serially on dedicated ranks; orthorhombic lattices (oC/oF/oI/oP) run in parallel across all ranks. The mapping is hardcoded in `run.py`. With 1 rank, everything runs serially. Requires `--mpi` flag; `mpi4py` is only imported when this flag is passed.

### Model training pipeline (`mlindex/model_training/Wrapper.py`)
`Wrapper` is the central training orchestrator. The three ML components trained per split group:

1. **RandomGenerator** (`RandomGenerator.py`): Random forest that predicts unit cell volume from the observed peak list. Used to generate random candidate unit cells.
2. **MITemplates** (`MITemplates.py`): Miller index template library + HistGradientBoosting calibrator. Templates are sets of hkl assignments sampled from training data; the calibrator scores how likely a template is to converge to the correct unit cell.
3. **ABNN** (`ABNN.py`): Attention-Based Neural Network -- `IntraVolume_MultiHeadAttention` over
   per-volume extraction branches. Predicts a unit cell from the peak list, then resamples
   Miller-index labellings of that cell to make several candidates from each prediction. The
   labellings come from `FigureOfMerits.get_assignment_distribution`, a closed-form posterior;
   the second network that used to assign them was retired in favour of it. Quantized for
   inference and stored as `.onnx` files.

All trained models are saved under `mlindex/models/{tag}/` (e.g., `mlindex/models/cubic_1/`).

### Dataset generation (`mlindex/dataset_generation/GenerateDataset.py`)
- Source databases: CSD and COD parquet files at `/global/cfs/cdirs/m4064/dwmoreau/`
  - `unique_csd_entries_cnrs_removed.parquet`
  - `unique_cod_entries_not_in_csd_cnrs_removed.parquet`
- Uses cctbx to simulate powder patterns from CIF structures; requires cctbx in the environment
- Runs MPI-parallel; each rank processes a subset of entries and saves `chunk_NN.parquet`; rank 0 combines into `dataset_{BL}.parquet`
- Output goes to `../data/generated_datasets/` relative to where the script is run

### Optimization (`mlindex/optimization/`)
- `Candidates.py`: `Candidates` class — runs the refinement loop, computes M20, handles reindexing and spacegroup assignment for a single rank's candidate set.
- `MPIOptimizer.py`: `OptimizerBase`, `OptimizerWorker`, `OptimizerManager` — MPI-based parallel optimizer. Contains extracted helpers `_run_loop`, `_generate_candidates_xnn`, `_downsample_computation` that are reused by the MP subclasses.
- `MPOptimizer.py`: `MPOptimizerManager`, `MPOptimizerWorker` — multiprocessing subclasses that override only the communication methods, replacing MPI calls with `multiprocessing.Queue`. Also contains the lattice-group runner (`setup_lattice_groups`, `run_lattice_groups`, `shutdown_lattice_groups`) and `run_mp_bl`.
- `GeneratorPools.py`, `GroupedStarts.py`: one pattern's candidates generated per generator, and starting clouds in groups -- the instruments behind the ensemble fractions; research, not the run path.
- `Optimizer.py`: Backward-compatibility re-export shim — imports from `Candidates` and `MPIOptimizer`. Existing notebooks and scripts that import from `Optimizer` continue to work.
- `UtilitiesOptimizer.py`: Factory functions (`get_cubic_optimizer`, etc.) and `get_optimizers`. All factory functions accept an `optimizer_class=None` parameter — pass `MPOptimizerManager` to use multiprocessing instead of MPI.
- `CandidateOptLoss.py`: Gauss-Newton least-squares update for xnn given hkl assignments.
- `AnalyticOptimizer.py`: Geometry-based candidate generation (no ML), used by the analytical CLI.
- `CandidateValidation.py`: Post-optimization spacegroup assignment based on systematic absences.

### Key utilities (`mlindex/utilities/`)
- `UnitCellTools.py`: Converts between xnn, unit cell (a,b,c,α,β,γ), and reciprocal-space forms. `fix_unphysical()` enforces physical constraints on xnn.
- `Q2Calculator.py`: Computes q² for given hkl and metric tensor.
- `FigureOfMerits.py`: M20 (de Wolff), `M_sym` and the per-peak assignment posterior (`get_assignment_distribution`, `get_assignment_posterior`).
- `Ranker.py`: the learned ranker's inputs, design matrix and calibration; `LearnedRanker` (ONNX) and the `MeritRanker` (M_sym) fallback; `load_ranker`, `score_pool`.
- `MillerIndexAssignment.py`: Miller-index assignment of a cell's lines to the observed peaks.
- `SpaceGroups.py`: Systematic absence rules and hkl reference sets per spacegroup.
- `Reindexing.py`: Selling and Niggli reduction, the monoclinic a-c reduction (`reduce_monoclinic_mesh`, keeping P, I or C centring), the printed setting, axis permutation, and `same_lattice` (cctbx) for comparing cells in any basis.
- `Allocation.py`, `Digests.py`: integer shares of a candidate budget; stable digests and seeds across processes.
- `EnsembleObjective.py`, `ClumpDiscount.py`, `ConvergenceCurve.py`, `ErrorAdder.py`: research instruments for the generator ensemble and peak-error simulation; not on the run path.

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
  so cleanup raises `PermissionError`. The legacy `--source github` download did this; it was
  removed in 0.2.0.

Not currently supported on Windows: MPI mode (`--mpi`), model training, and dataset
generation. `cctbx-base` ships a `win_amd64` wheel but no `win_arm64`, so Windows-on-ARM
cannot install at all.

## Optional dependencies

Dataset generation requires: `gemmi`, `pyarrow`, `openpyxl`, `cctbx`, `tqdm`

Model training additionally requires: `skl2onnx`, `onnx`, `tf2onnx`, `keras`, `torch`, `torchvision`

The indexer itself needs `onnxruntime>=1.21`: 1.20 has no implementation of the ranker's
`TreeEnsemble` (ai.onnx.ml opset 5) and fails to load it; 1.21.0 to 1.24.1 agree with the fit to
3e-15 (P18, macOS arm64).

## The ML-FOM project — its record is untracked, and this is the only pointer to it

There is a long-running research project on top of this codebase — **ML-FOM**, replacing the de
Wolff M20 figure of merit with a learned one. Its entire working record lives under `docs/`, which
`.gitignore` excludes on purpose: those are working notes and they are not for GitHub.

**So `docs/` is not in this repository.** A fresh clone has none of it, `git pull` on another
machine does not deliver it, and `git clean -xdf` deletes it. This section exists because it is
the only trace of the project that a clone does carry. If `docs/` is missing, the record has to be
restored from a backup or from NERSC before doing any FOM work:

```bash
rsync -a ~/mli-record-backup/<YYYY-MM-DD>/docs/ docs/       # local dated snapshot
docs/sync_record.sh snapshot                                # take a new one
```

**There have been two campaigns, and they are kept separate.**

- `docs/fom_campaign1/` — the first campaign, **closed 2026-08-23**. Read-only. See its
  `CLOSED.md`. Do not edit anything in it.
- `docs/fom_campaign2/` — the live campaign, started 2026-08-24.

When `docs/` is present, read `docs/fom_campaign2/PROTOCOL.md` first — it defines how a session is
conducted and its standing rules override any other instruction in that project — then
`INHERITED.md` (what campaign 1 established, and what it left bounded), `README.md`, `STATUS.md`,
and the relevant handoff under `docs/fom_campaign2/handoffs/`.

**Do not read campaign 1's `STATUS.md` to find out what carries forward.** It is 7 524 lines and
much of it is superseded by its own later entries. `INHERITED.md` is the distillation; go to the
archive only for the evidence behind one specific number.

Four things about that work that affect ordinary edits here:

- **FOM work goes on branch `fom_campaign2`; general correctness fixes go on `main`** and are
  merged in. The test is whether the change would matter to someone who never touches figures of
  merit.
- **`fom_campaign2` branched from `main`, not from `fom`.** The old `fom` branch is kept for its
  code, not extended: campaign-2 sessions **cherry-pick** what they need from it and record what
  they took in `docs/fom_campaign2/CHERRY_PICK.md`. Seventy-one files differ between `main` and
  `fom`, most of them belonging to work campaign 2 has dropped, so do not merge `fom`.
- **Pushing to `origin` needs an unlocked SSH key**, and Perlmutter cannot push at all
  (`Permission denied (publickey)`), so a fix recorded as done may exist on one machine only.
  Check with `git log origin/main --oneline -- <path>` before relying on it. Two correctness
  fixes sat on a local `main` unpushed for days because of exactly this.
- **Never `git add -A`.** `mlindex/models/` is git-lfs and often mid-retrain, and
  `mlindex/characterization/` and `mlindex/data/generated_datasets/` are regenerable run output.
  Stage named paths.
