# tools/

Development tools. **Nothing here ships.** `pyproject.toml` finds only
`mlindex` and `mlindex.*` packages, so this directory is absent from the wheel,
and no module under `mlindex/` imports anything here. Run these from a source
checkout, from the repository root.

## What is on this branch

`fom_campaign2` carries **five** of the fourteen files this directory holds on
`fom`, because campaign 2 takes only what a step uses and records what it left
(`docs/fom_campaign2/CHERRY_PICK.md`). Ported by S02:

| file | what it is |
| --- | --- |
| `microbench.py` | the shared timing methodology every reproducer runs through |
| `capture_hook.py` | **new here.** How a capture records the optimiser's real arguments |
| `repro_fom_zoo.py` | capture / cost / profile / bench / verify / fuzz for the merit zoo |
| `repro_msym.py` | the same for `get_M_rev_sym`, and the fuzz for two memory-order traps |
| `s02_cost_sweep.py` | **new here.** Regenerates `artifacts/S02_zoo_cost.csv` in one command |

The other nine — `profile_run.py`, `mlindex_profiler.py`, `ab_fast_assign.py`,
`bench_assign_hkls.py`, `bench_fast_assign.py`, `repro_downsample.py`,
`repro_hessian.py`, `repro_offbytwo.py`, `repro_subsampling.py`,
`validate_true_cell.py` — are on `fom` at `7c137c3` and are **not** ported. The
sections below that describe them are kept because they record measurements
worth not repeating, but the files they name are not here.

**Two changes from the `fom` originals**, both in the reproducers. The capture
point is `Candidates.get_M20` rather than `Candidates._retention_fom_values`,
which does not exist on this branch — see `capture_hook.py`, and note the
re-pointed hook was checked to produce a **bit-identical capture** to the old
one. And `merit_calls`, `_named_outputs` and `PROFILE_TARGETS` no longer name
the cross-validated and in-sample families, which campaign 2 does not port.
`cost` additionally gained `--csv` and its provenance flags.

**The measured tables below are campaign 1's**, on campaign 1's hardware. S02's
own measurements are in `docs/fom_campaign2/artifacts/S02_zoo_cost.csv`, and the
prices there supersede any number in this file.

## `profile_run.py` — where does `mlindex.run` spend its time?

Drives `run._run_mp`, the same entry point `python -m mlindex.command_line.run`
uses, under two independent measurement layers:

| layer | what it answers | artifact |
| --- | --- | --- |
| phase timers | which *pipeline stage* is slow | `<tag>.phases.txt`, `.json` |
| cProfile | which *function* inside that stage is slow | `<tag>.prof`, `<tag>.cprofile.txt` |

Each layer runs as its own pass. They are not combined, because cProfile
charges the phase-timer wrapper frames to the wrappers (they dominated the
table at `--tier detail`) and the wrappers in turn slow down what cProfile is
measuring.

```bash
# stage-level breakdown of a serial run (default, lowest overhead)
python tools/profile_run.py --peak-file peaks.npy --peak-units q2

# function-level detail, with a measured baseline and an output diff
python tools/profile_run.py --peak-file peaks.npy --peak-units q2 \
    --tier detail --baseline

# one lattice, to iterate quickly
python tools/profile_run.py --peak-file peaks.npy --peak-units q2 \
    --bravais-lattices aP --tier detail --baseline

# multiprocessing: workers write their own <tag>.rank<N>.* artifacts
python tools/profile_run.py --peak-file peaks.npy --peak-units q2 --nproc 4

# explore the call graph
python -m pstats indexing_profile/profile.prof
snakeviz indexing_profile/profile.prof
```

### Reading the phase report

Stages nest, so the report gives both inclusive and **self** time; self time
excludes instrumented children, which makes the self column sum to the total
exactly once. Rows are broken down per Bravais lattice, and array sizes
(candidates per rank, `hkl_ref` length, peaks used, downsample chunk sizes)
are recorded alongside so a hot stage can be read against the work it did.

### `--baseline`

Prepends a discarded warm-up pass and then an uninstrumented pass. The warm-up
matters: numba's JIT and onnxruntime's first session cost roughly 2 s in a cold
interpreter, enough to make the instrumented passes look *faster* than an
un-warmed baseline. Overhead percentages are computed against the warm
uninstrumented pass, and the baseline's `indexing_results.json` is diffed
against the profiled one so any change in results shows up immediately.

Measured on this machine (single `aP` lattice, 20 peaks): phase timers at
`--tier detail` cost +1.2%, and results were bit-identical.

`--cold-baseline` additionally times a fresh `python -m mlindex.command_line.run`
subprocess. That is what a user actually waits for, but it includes interpreter
startup and imports, so it is reported separately and never used for overhead.

### Tiers

- `stage` (default) — one timer per pipeline step. Negligible overhead.
- `detail` — also per-array-op helpers (`fast_assign`, `get_M20`,
  `gauss_newton_step`, `Q2Calculator.get_q2`, ...).
- `full` — also model loading in `Wrapper.setup_*`.

## Hotspot reproducers

One isolated, runnable reproducer per candidate optimization the profile turned
up. Each captures the **real** arguments the hot function is called with during a
run, then benchmarks candidate implementations against them and checks the output.

| file | what it tests | measured share |
| --- | --- | --- |
| `repro_downsample.py` | `_downsample_chunk` rebuilding an O(n^2) distance matrix per removal | 2.9% |
| `repro_hessian.py` | `matrix_rank`/`eigvalsh`, `inv`+`matmul` vs `solve`, and the 4-D Hessian intermediate | 2.8% + 2.9% + part of 6.7% |
| `repro_subsampling.py` | `vectorized_subsampling` compacting its arrays every pick | 3.0% |
| `repro_offbytwo.py` | `correct_off_by_two` materialising all 64 factor slices (~103 MB) | memory |
| `repro_msym.py` | `get_M_rev_sym` materialising `(n_candidates, n_ref, n_peaks)` | 88% of the merit |
| `repro_fom_zoo.py` | the rest of the FOM zoo: `n_over`/`max_gap`, `M_wu`, `M_1`, the assignment family | 90% of `compute_all` |

All of them go through `microbench.py`, which enforces one timing methodology:

- **interleaved** — each round times every variant once, so drift hits all alike;
- **order-randomised** within a round, so position can't bias a variant;
- **paired** — ratios formed within a round, reported as median with [p10, p90];
- **null-calibrated** — every reproducer registers a `control` variant that is a
  byte-identical copy of the reference. Its true ratio is 1.000x by
  construction, so its interval is the noise floor. A variant whose interval
  overlaps the control's has not been shown to differ from the reference,
  however good its median looks.

The control is the load-bearing part. An earlier "best of N, run sequentially"
harness could not distinguish a 3% effect from a 3% artefact, and it reported an
end-to-end result that was pure background load.

Correctness is checked *before* timing, and a variant whose output differs is
reported as DISQUALIFIED regardless of speed. Beyond the captured inputs, the
order-sensitive rewrites are cross-checked against the shipped implementation on
randomised small inputs — which is how the `masked` subsampling variant was
caught re-picking already-chosen indices on 203 of 300 cases.

### `repro_msym.py` — the M_sym figure of merit

`get_M_rev_sym` returns `M_tilde`, `M_rev` and `M_sym`, and is on the ML-FOM
project's shortlist for the inner loop rather than on the shipped indexing path
(nothing calls it unless `retention_foms` asks for one of the three). It scored
every reference line against its nearest observed peak by materialising the whole
`|reference - peak|` stack and minimising over its last axis — 1.6 GB and 88% of
the call at 10 000 candidates against a 1 000-line reference list.

The rewrite does the same work per row in a numba kernel: only the in-range lines
are scored, and the nearest peak is found by a branchless binary search over the
sorted peaks instead of a scan over all of them.

```bash
python tools/repro_msym.py capture --peak-file 11bmb_3844_peak_list.npy \
    --peak-units q2 --bravais-lattice mP
python tools/repro_msym.py profile --capture-file captures/msym_mP.npz
python tools/repro_msym.py bench   --capture-file captures/msym_mP.npz
python tools/repro_msym.py cost    --capture-file captures/msym_mP.npz
python tools/repro_msym.py fuzz
```

Measured on the mP capture (1 000 reference lines, 20 peaks), bit-identical at
every size and on 600 randomised cases:

| candidates | before | after | speedup | peak allocation |
| --- | --- | --- | --- | --- |
| 1 000 | 68.8 ms | 5.1 ms | 13.4x | 321 MB -> 9 MB |
| 6 000 (production) | 411 ms | 28.4 ms | 14.5x | 1.9 GB -> 55 MB |
| 10 000 | 678 ms | 48.2 ms | 14.0x | 3.2 GB -> 91 MB |

Against `get_M20` in the same harness, which is the unit `S06_zoo_cost.csv` and
`FomCombiner.affordable_features` use: **36.4x -> 2.5x** at the mP inner-loop size,
and 9.5x -> 1.6x on the cubic capture (100 lines, 10 peaks).

Two things the variants in that file exist to pin down, both of which cost
bit-identity when they are got wrong, and both of which the fuzz catches:

* `scored` and `in_range` are allocated with `empty_like`, so they inherit the
  reference array's memory order. `.sum(axis=1)` does not group its additions the
  same way over an F-ordered array as over a C-ordered one.
* `scored` keeps its zeros instead of being compacted to the in-range entries.
  numpy's pairwise summation groups a full row differently from a short one even
  though every dropped term is an exact zero.

Search forms measured on the 10 000-candidate capture, kernel time only: textbook
`while left < right` binary search 65.8 ms, branchless `lower_bound` 41.2 ms,
branchless linear count over all peaks 38.6 ms. The linear count wins at
`n_peaks = 20` and loses at any larger peak list, so the branchless binary search
is what ships. Writing the nearer-of-two distance as an if/else rather than a
conditional expression costs 2x on its own: which of the pair is nearer is a coin
flip, so the branch mispredicts half the time.

### `repro_fom_zoo.py` — the rest of the zoo

Same job as `repro_msym.py` for every other merit `compute_all` evaluates, plus the
cross-validated and assignment families S10 and S11 added. It prices merits
rather than assuming which is slow, and it verifies against **the module as it
was at a git revision** rather than against hand-copied variants, so a rewrite
that perturbs a neighbouring merit cannot pass.

```bash
# the candidates the *analysis* scores: converged, pruned, from the frozen pool
python tools/repro_fom_zoo.py capture-pool --bravais-lattice mP --lattice-system monoclinic
# the candidates the *inner loop* holds: mid-refinement, far denser
python tools/repro_fom_zoo.py capture --peak-file 11bmb_3844_peak_list.npy \
    --peak-units q2 --bravais-lattice mP

python tools/repro_fom_zoo.py cost    --capture-file captures/pool_mP.npz            # now
python tools/repro_fom_zoo.py cost    --capture-file captures/pool_mP.npz --revision HEAD
python tools/repro_fom_zoo.py profile --capture-file captures/pool_mP.npz --merits n_over/max_gap
python tools/repro_fom_zoo.py bench   --capture-file captures/pool_mP.npz
python tools/repro_fom_zoo.py verify  --capture-file captures/pool_mP.npz
python tools/repro_fom_zoo.py fuzz
```

**Capture the pool, not the loop.** The fraction of reference lines below the
cut-off is what decides what most of these merits cost, and it is **5.5%** on the
frozen benchmark pool against **44%** in the inner loop, because the loop's
candidates have not converged yet. Both captures are supported; the pool one is
what the S06/S08 cost column means.

Measured on `pool_mP` (1 000 reference lines, 20 peaks, 1 000 candidates), against
the same module at `HEAD`. `get_M20` is 2.2 ms in the same run:

| merit | before | after | speedup | vs `get_M20` |
| --- | --- | --- | --- | --- |
| `n_over` / `max_gap` | 66.3 ms | 5.4 ms | **13.5x** | 29.8x -> 2.8x |
| `compute_all` (24 columns) | 155.0 ms | 19.6 ms | **8.2x** | 69.7x -> 10.1x |
| assignment posterior | 166.9 ms | 29.3 ms | **5.7x** | 75.1x -> 15.1x |
| assignment sigma | 33.6 ms | 9.5 ms | **3.1x** | 15.1x -> 4.9x |
| `M_wu` | 12.9 ms | 6.7 ms | 1.70x | 5.8x -> 3.5x |
| `M_1` | 7.2 ms | 6.0 ms | 1.21x | 3.2x -> 3.1x |

The four rewrites, and what each one was:

* **`get_n_over`** built the same `(n_candidates, n_ref, n_peaks)` stack `M_sym`
  did, and then found the longest run of unaccounted-for lines in a **Python
  double loop** — 1.3 M interpreted iterations per 1 000 candidates. Both are now
  one forward pass per row. Its two outputs are counts, so unlike the merits that
  sum floats there is no summation grouping to preserve here.
* **`get_assignment_sigma`** scanned the reference array once per peak, twenty
  passes each building a `(chunk, n_ref)` temporary. One pass carrying twenty
  running minima reads the row once.
* **`get_assignment_posterior`** spent its time building the log-sum-exp
  argument, not on the exponential: fusing the four array passes is worth 5.7x,
  of which skipping the 98.8% of exponentials that underflow to exactly zero is
  only 3.7 ms. `np.exp(..., out=, where=)` is bit-identical to `np.exp` on a
  boolean-indexed subset, checked over 20 M values.
* **`_sorted_lines_in_range` is now computed once in `compute_all`** and handed
  to `M_wu`, `M_1` and `n_over`, which each used to sort the same array from the
  same cut-off. They still sort for themselves when called directly.

Everything is checked by `verify`, which compares **every column of every family**
— 49 of them — against the module at `HEAD`: bit-identical on monoclinic,
tetragonal and triclinic pool captures at 1 000 and 5 000 candidates, and on the
inner-loop capture at 1 000 and 10 000. Plus 800 randomised cases through `fuzz`.

Two things `fuzz` caught that the captures could not, both now fixed and both
worth knowing before writing another kernel here:

* **A running minimum does not propagate NaN.** `NaN < running` is false, so a
  naive scan returns the minimum of the rest where `.min()` returns NaN.
* **Memory order changes a row sum.** `np.sum(axis=1)` groups its additions
  differently over an F-ordered block, and which order a numpy expression
  produces when its operands disagree is not something to reproduce by
  construction — so the posterior takes its fast path only for a C-contiguous
  reference array.

**Tried and rejected, with numbers, so they are not retried without new
evidence.** Compacting the in-range lines to the front before sorting, so the
sort is over `max(count)` columns instead of `n_ref`: **1.5x at 4% density but
0.75x at the captured 44%**, and numpy's own sort already speeds up on the
mostly-`+inf` input, so it is conditional on a density the caller does not know.
Sorting inside numba instead: **0.75x** — numba's quicksort loses to numpy's.

**Left alone, and why.** `cv_fom` is the most expensive merit in the zoo (21x
`get_M20`) and **96% of it is already compiled**: 72% in `gauss_newton_solve` and
24% in `fast_assign`, both tuned numba kernels. Its cost is five folds x two
refits plus five assignment scans — real work, not implementation waste. Same for
`holdout_fom` (3.6x) and `insample_fom` (1.9x), which are `fast_assign` plus
arithmetic on `(n_candidates, n_peaks)`.

An independent check on the harness: measured against `HEAD` it puts `n_over` at
**29.8x** `get_M20` and `max_gap` with it, where `S06_zoo_cost.csv` recorded 29.4x
and 29.5x on different hardware in a different script.

### Results (11bmb_3844 peak list, 14 lattices, nproc=1)

Measured with the harness above; every ratio cleared its own noise floor.

| change | speedup on the function | output |
| --- | --- | --- |
| `fast_assign` four interleaved accumulators | 1.13-1.22x | bit-identical |
| `_downsample_chunk` maintained distance matrix | 9.2-22.1x | bit-identical |
| `vectorized_subsampling` zero instead of compact | 1.13x | bit-identical |
| `correct_off_by_two` running best (103 MB -> 1.6 MB) | 1.41x | bit-identical |

Verified end to end: the four together give `IDENTICAL: max relative
difference 0.000e+00` against the pre-change `indexing_results.json`.

**Not applied.** `gauss_newton_solve` in `numba_functions.py` is a
per-candidate Gauss-Newton kernel: 9.3-9.6x, peak working memory 224 MB -> 1.3
MB on a 39,753-candidate chunk, and it cannot raise. It is not wired into
`CandidateOptLoss.gauss_newton_step` because it perturbs the step by ~1e-8
relative to the cell, and this optimizer amplifies that: the top 5 candidates
are unchanged but 4 of the top 20 differ. `repro_hessian.py` has the
measurements and the robustness scenarios. Enabling it is a judgement call about
whether a changed tail ranking is acceptable, not a technical blocker.

Things that were tried and lost, so they are not worth retrying without new
evidence: thread-level `prange` inside `fast_assign` (5.5-6.7x on an idle
machine, but the candidates are already split across processes, so it only
nests threads under saturated cores); sorting each reference row and binary
searching (0.15x); swapping the loop order (0.85x); block min/max pruning
(0.82x); a vectorizable min pass plus an index lookup (0.55x); holding the
argmin index in a local rather than storing in the branch (0.65x); eight
accumulator lanes rather than four (0.85x).

### Caveats

- `_downsample_computation` fans chunks over a `ThreadPoolExecutor`, which
  cProfile cannot see. The profiler runs those chunks inline so they appear;
  `list(ex.map(...))` yields the same values in the same order either way, so
  results are unchanged. Pass `--keep-threads` to profile the real pool
  instead, accepting that chunk time goes missing from the cProfile table.
- With `--nproc N > 1` each worker writes its own `<tag>.rank<N>.*` artifacts.
  Worker wall time includes idle time spent blocked on the task queue.
