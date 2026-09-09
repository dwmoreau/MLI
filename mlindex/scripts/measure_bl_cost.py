"""Re-measure the per-Bravais-lattice cost table that drives --nproc planning.

`allocate_lattice_groups` in `mlindex/command_line/run.py` decides which lattices
share a process and which are split across several. It decides that from `_BL_COST`,
a table of (generation, refinement) seconds per Bravais lattice. Those numbers were
measured once, on one machine; this script measures them again so the table can be
checked, or retuned for a machine whose balance differs.

The two halves are measured separately because they behave differently under
parallelism. Generation runs on a group's manager alone -- only it holds the models --
so it does not divide by group size. Refinement stripes across the group and does.

    gen  = time inside OptimizerManager._generate_candidates_xnn
    par  = time inside OptimizerBase._run_loop, minus gen

Nothing is patched permanently and no model file is touched: each lattice's optimizer
is built on a one-process communicator and its two methods are wrapped on the instance
for the duration of the run.

Worked command -- all fourteen lattices, three patterns, about ten minutes:

    python -m mlindex.scripts.measure_bl_cost --output bl_cost.md

A quicker check of two lattices:

    python -m mlindex.scripts.measure_bl_cost --bravais-lattices mP,cI --repeats 2

The script prints a ready-to-paste `_BL_COST` and says whether the new numbers would
change any allocation plan. If they do not, the existing table is still good and
nothing needs editing.
"""
import argparse
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

# Patterns shipped with the package, spanning several lattice systems. Every Bravais
# lattice is tried against every pattern, so these only have to be representative
# peak lists, not examples of each lattice.
DEFAULT_PATTERNS = ('11bmb_3844', 'FAP', 'PBSO4')

N_TOP_CANDIDATES = 20
BROADENING_TAG = '1'


def _test_data_dir():
    from importlib.resources import files

    return Path(str(files('mlindex').joinpath('data', 'test_data')))


def _load_pattern(name):
    """Load one packaged pattern as a sorted q2 array of at most 20 peaks."""
    path = _test_data_dir().joinpath(name, f'{name}_peak_list.npy')
    if not path.is_file():
        raise SystemExit(
            f"No packaged peak list for '{name}' at {path}.\n"
            f"Available: {', '.join(sorted(p.name for p in _test_data_dir().iterdir() if p.is_dir()))}"
        )
    return np.sort(np.load(path))[:N_TOP_CANDIDATES]


def _load_peak_file(path):
    """Load a .npy of q2 values, the same units the indexer works in."""
    return np.sort(np.load(path))[:N_TOP_CANDIDATES]


def _build_optimizer(bravais_lattice, seed):
    """One manager optimizer for `bravais_lattice`, on a one-process communicator.

    Goes through `get_optimizers` rather than the per-system factories directly, so
    this script cannot drift from the mapping the indexer itself uses.
    """
    from mlindex.optimization.MPOptimizer import LocalComm
    from mlindex.optimization.UtilitiesOptimizer import get_optimizers

    organizers = {bravais_lattice: SimpleNamespace(
        manager=0, workers=[0], split_comm=LocalComm(1), color=None)}
    return get_optimizers(0, organizers, BROADENING_TAG, n_candidates_scale=1,
                          seed=seed)[bravais_lattice]


def _instrument(optimizer, totals):
    """Wrap the two timed methods on this instance only; nothing global changes."""
    for name in ('_generate_candidates_xnn', '_run_loop'):
        if not hasattr(optimizer, name):
            raise SystemExit(
                f"{type(optimizer).__name__} has no {name}. This script times that "
                f"method by name; if it was renamed, update the script."
            )

    inner_gen = optimizer._generate_candidates_xnn
    inner_loop = optimizer._run_loop

    def timed_gen(*args, **kwargs):
        start = time.perf_counter()
        try:
            return inner_gen(*args, **kwargs)
        finally:
            totals['gen'] += time.perf_counter() - start

    def timed_loop(*args, **kwargs):
        start = time.perf_counter()
        try:
            return inner_loop(*args, **kwargs)
        finally:
            totals['loop'] += time.perf_counter() - start

    optimizer._generate_candidates_xnn = timed_gen
    optimizer._run_loop = timed_loop


def _time_one_run(optimizer, q2_obs):
    """Return (gen, par) seconds for indexing `q2_obs` once."""
    totals = {'gen': 0.0, 'loop': 0.0}
    _instrument(optimizer, totals)
    try:
        optimizer.run(q2=q2_obs, n_top_candidates=N_TOP_CANDIDATES)
    finally:
        # Drop the wrappers so a later run re-wraps the real methods rather than
        # nesting one set of timers inside another.
        del optimizer._generate_candidates_xnn
        del optimizer._run_loop
    return totals['gen'], totals['loop'] - totals['gen']


def measure(bravais_lattices, patterns, repeats, warmup, seed, verbose=True):
    """Return {bravais_lattice: {'gen': [...], 'par': [...]}} of per-run seconds."""
    samples = {}
    for bravais_lattice in bravais_lattices:
        if verbose:
            print(f"  {bravais_lattice}: loading models...", end='', flush=True)
        optimizer = _build_optimizer(bravais_lattice, seed)
        gen_all, par_all = [], []
        for label, q2_obs in patterns:
            # Untimed first, so numba compilation and first-touch caching land here
            # rather than in a measurement.
            for _ in range(warmup):
                _time_one_run(optimizer, q2_obs)
            for _ in range(repeats):
                gen, par = _time_one_run(optimizer, q2_obs)
                gen_all.append(gen)
                par_all.append(par)
        samples[bravais_lattice] = {'gen': gen_all, 'par': par_all}
        if verbose:
            print(f"\r  {bravais_lattice}: gen {statistics.median(gen_all):6.2f} s   "
                  f"par {statistics.median(par_all):6.2f} s   "
                  f"({len(gen_all)} samples)      ")
        # The models are the memory cost; drop them before building the next lattice.
        del optimizer
    return samples


def _spread(values):
    """Half the min-to-max range, as a plain indication of run-to-run scatter."""
    return 0.5 * (max(values) - min(values))


def _format_table(samples):
    lines = ["| lattice | gen (s) | par (s) | gen spread | par spread | samples |",
             "|---|---|---|---|---|---|"]
    for bl, s in samples.items():
        lines.append(
            f"| {bl} | {statistics.median(s['gen']):.2f} | {statistics.median(s['par']):.2f} "
            f"| +/-{_spread(s['gen']):.2f} | +/-{_spread(s['par']):.2f} | {len(s['gen'])} |")
    return "\n".join(lines)


def _format_cost_dict(samples, order):
    """The measured table, laid out as `_BL_COST` is laid out in run.py."""
    rows = [('cF', 'cI', 'cP'), ('hP', 'hR'), ('tI', 'tP'),
            ('oC', 'oF', 'oI', 'oP'), ('mC', 'mP', 'aP')]
    out = ["_BL_COST = {"]
    for row in rows:
        present = [bl for bl in row if bl in samples]
        if not present:
            continue
        cells = [f"'{bl}': ({statistics.median(samples[bl]['gen']):.2f}, "
                 f"{statistics.median(samples[bl]['par']):.2f})," for bl in present]
        out.append("    " + " ".join(cells))
    leftover = [bl for bl in order if bl not in sum(rows, ())]
    for bl in leftover:
        out.append(f"    '{bl}': ({statistics.median(samples[bl]['gen']):.2f}, "
                   f"{statistics.median(samples[bl]['par']):.2f}),")
    out.append("    }")
    return "\n".join(out)


def _compare_with_current(samples):
    from mlindex.command_line.run import _BL_COST

    lines = ["| lattice | gen now | gen measured | par now | par measured | total ratio |",
             "|---|---|---|---|---|---|"]
    for bl, s in samples.items():
        gen_new = statistics.median(s['gen'])
        par_new = statistics.median(s['par'])
        gen_old, par_old = _BL_COST[bl]
        old_total, new_total = gen_old + par_old, gen_new + par_new
        ratio = new_total / old_total if old_total else float('inf')
        lines.append(f"| {bl} | {gen_old:.2f} | {gen_new:.2f} | {par_old:.2f} "
                     f"| {par_new:.2f} | {ratio:.2f}x |")
    return "\n".join(lines)


def _compare_plans(samples, process_counts):
    """Does the measured table change any allocation? That is the decision.

    The table only has to rank lattices well enough to plan. If every plan is
    unchanged, the numbers may have moved but nothing that depends on them has.
    """
    from mlindex.command_line import run as run_module

    if set(samples) != set(run_module.BRAVAIS_LATTICES):
        return ("Plans not compared: that needs all fourteen lattices, and this run "
                f"measured {len(samples)}.")

    measured = {bl: (statistics.median(s['gen']), statistics.median(s['par']))
                for bl, s in samples.items()}
    lattices = list(run_module.BRAVAIS_LATTICES)

    def plans_for(table):
        original = run_module._BL_COST
        run_module._BL_COST = table
        try:
            return {n: run_module.allocate_lattice_groups(lattices, n)
                    for n in process_counts}
        finally:
            run_module._BL_COST = original

    current = plans_for(dict(run_module._BL_COST))
    updated = plans_for(measured)

    lines = ["| processes | plan changes? | makespan now | makespan measured |",
             "|---|---|---|---|"]
    changed_any = False
    for n in process_counts:
        same = ([sorted(b) for b, _ in current[n]] == [sorted(b) for b, _ in updated[n]]
                and [k for _, k in current[n]] == [k for _, k in updated[n]])
        changed_any = changed_any or not same
        original = run_module._BL_COST
        run_module._BL_COST = dict(original)
        mk_now = max(run_module._group_cost(b, k) for b, k in current[n])
        run_module._BL_COST = measured
        mk_new = max(run_module._group_cost(b, k) for b, k in updated[n])
        run_module._BL_COST = original
        lines.append(f"| {n} | {'no' if same else 'YES'} | {mk_now:.2f} s | {mk_new:.2f} s |")
    verdict = ("At least one plan changes, so the table is worth updating."
               if changed_any else
               "No plan changes at any of these process counts, so the current table "
               "is still good enough and nothing needs editing.")
    return "\n".join(lines) + "\n\n" + verdict


def build_parser():
    parser = argparse.ArgumentParser(
        description=("Re-measure the per-Bravais-lattice cost table used to plan "
                     "--nproc allocation. Prints a ready-to-paste _BL_COST and says "
                     "whether any allocation plan would change."),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=("Takes about ten minutes for all fourteen lattices at the defaults. "
                "Set MLINDEX_MODELS_DIR to pin which model tree is measured."),
    )
    parser.add_argument(
        "--bravais-lattices", type=str, default=None,
        help="Comma-separated lattices to measure (default: all 14)")
    parser.add_argument(
        "--patterns", type=str, default=",".join(DEFAULT_PATTERNS),
        help=f"Comma-separated packaged pattern names (default: {','.join(DEFAULT_PATTERNS)})")
    parser.add_argument(
        "--peak-file", type=str, action='append', default=None,
        help="Path to a .npy of q2 values (1/Angstrom^2); repeatable. Overrides --patterns")
    parser.add_argument(
        "--repeats", type=int, default=3,
        help="Timed runs per lattice per pattern (default: 3)")
    parser.add_argument(
        "--warmup", type=int, default=1,
        help="Untimed runs before timing, for JIT and cache warm-up (default: 1)")
    parser.add_argument(
        "--seed", type=int, default=12345,
        help="Seed for the candidate search (default: 12345)")
    parser.add_argument(
        "--process-counts", type=str, default="4,8,10,14,16",
        help="Process counts at which to compare allocation plans (default: 4,8,10,14,16)")
    parser.add_argument(
        "--output", type=str, default=None,
        help="Write the full report to this file as well as printing it")
    return parser


def main():
    args = build_parser().parse_args()

    from mlindex.command_line.run import BRAVAIS_LATTICES

    if args.bravais_lattices:
        bravais_lattices = [bl.strip() for bl in args.bravais_lattices.split(',')]
        unknown = [bl for bl in bravais_lattices if bl not in BRAVAIS_LATTICES]
        if unknown:
            raise SystemExit(f"Unknown Bravais lattices: {', '.join(unknown)}")
    else:
        bravais_lattices = list(BRAVAIS_LATTICES)

    if args.peak_file:
        patterns = [(Path(p).name, _load_peak_file(p)) for p in args.peak_file]
    else:
        names = [n.strip() for n in args.patterns.split(',') if n.strip()]
        patterns = [(n, _load_pattern(n)) for n in names]
    if not patterns:
        raise SystemExit("No patterns to measure.")

    process_counts = [int(n) for n in args.process_counts.split(',') if n.strip()]

    print(f"Measuring {len(bravais_lattices)} lattices over {len(patterns)} patterns, "
          f"{args.repeats} timed runs each after {args.warmup} warm-up.\n"
          f"Patterns: {', '.join(name for name, _ in patterns)}\n")
    started = time.perf_counter()
    samples = measure(bravais_lattices, patterns, args.repeats, args.warmup, args.seed)
    elapsed = time.perf_counter() - started

    report = "\n".join([
        "# Measured Bravais-lattice cost table",
        "",
        f"Patterns: {', '.join(name for name, _ in patterns)}",
        f"{args.repeats} timed runs per lattice per pattern, after {args.warmup} warm-up; "
        f"median reported. Seed {args.seed}. Total measurement time {elapsed / 60:.1f} min.",
        "",
        "## Measured",
        "",
        _format_table(samples),
        "",
        "## Against the table in use",
        "",
        _compare_with_current(samples),
        "",
        "## Would any allocation plan change?",
        "",
        _compare_plans(samples, process_counts),
        "",
        "## Ready to paste into mlindex/command_line/run.py",
        "",
        "```python",
        _format_cost_dict(samples, bravais_lattices),
        "```",
        "",
    ])
    print("\n" + report)

    if args.output:
        # encoding is explicit: Windows would otherwise use the locale codepage.
        Path(args.output).write_text(report, encoding='utf-8')
        print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
