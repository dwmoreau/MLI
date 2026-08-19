"""S10 Variant A: rebuild the peaks the benchmark threw away, and prove the ones it kept are intact.

The literal hold-out -- fit on the twenty peaks, score on the ones beyond them -- needs peaks that
Benchmark A does not store. Only their count survives, as `n_peaks_available` (STATUS section 7,
R13). The full non-absent line list does still exist, in the generator's own source tables at
`mlindex/data/generated_datasets/dataset_<BL>.parquet`, so the hold-out set is reconstructible by
replaying `run_fom_mirror`'s mechanisms rather than by re-running the indexer.

Two things make that dangerous, and this script exists to make both safe.

**The frozen pattern must not move.** `add_q2_error` draws over its whole array in one call, so
noising twenty-five peaks on the entry's own RNG stream would change the noise on the first twenty
and the pool would no longer correspond to its own peak list. The surplus is therefore noised on a
*separate* derived stream, `extra:<identifier>`, and the original stream is replayed untouched.

**The replay must be proved, not assumed.** Every entry's first twenty peaks are regenerated here
through the same calls in the same order and compared with the stored `q2_obs`. If that gate does
not pass, the reconstruction is wrong and no Variant A number may be reported -- which is the whole
reason this is a separate script from the scoring driver rather than a stage inside it.

What the gate found, and why it is a tolerance rather than a bit comparison: every peak belonging
to the *true structure* replays bit for bit, in every bundle. The bundles carrying contaminants
(C3, C4, C5) disagree on the *contaminant* positions alone, by 3e-18 -- sub-ULP arithmetic noise in
`add_contaminants`, and the expected consequence of the pool having been generated on x86 while
this runs on arm64 (R9). So the tolerance is REPLAY_TOLERANCE below, twelve orders under the
smallest error the generator itself applies, and the count of bit-exact entries is reported beside
it rather than being the gate. The quantity Variant A actually depends on -- the top of the fitted
window, which decides where the hold-out lines start -- is in the bit-exact half.

    python mlindex/scripts/run_fom_cv_holdout.py --n-extra 5

Writes `mlindex/data/fom_cv/holdout_peaks_<bundle>.parquet` and the gate table
`docs/fom/artifacts/S10_cv_holdout_gate.csv`.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, 'mlindex', 'scripts'))

from mlindex.model_training import FomBenchmark  # noqa: E402
from mlindex.utilities.ErrorAdder import add_contaminants  # noqa: E402
from mlindex.utilities.ErrorAdder import add_q2_error  # noqa: E402
from mlindex.utilities.ErrorAdder import add_second_phase  # noqa: E402
from mlindex.utilities.ErrorAdder import select_peaks_with_dropout  # noqa: E402
import run_fom_mirror as mirror  # noqa: E402
from run_fom_zoo_features import commit_hash  # noqa: E402


# Peaks agree to 3e-18 where they disagree at all, against a generator error of ~1e-4 at the same
# q2 and a peak separation four orders larger again. A bit comparison would fail on arithmetic
# noise (R9); this fails on anything that could move an assignment.
REPLAY_TOLERANCE = 1e-15

# C0 is excluded from every metric (F-054) but its peak list is as replayable as any other, so the
# gate runs on all seven and the scoring driver decides what to evaluate.
DEFAULT_BUNDLES = (
    'error1_cont0',
    'error2_cont0',
    'error1_cont2',
    'error1_cont1_drop6',
    'error1_cont1_drop10',
    'error1_cont0_phase3',
    )


def peak_tables(bravais_lattices):
    """identifier -> full non-absent line list, for every lattice named.

    Read once and held, because the second-phase partner of an entry is drawn from the whole
    sampled set rather than from within its own lattice, so a per-lattice read would miss it.
    """
    lines = {}
    for bravais_lattice in sorted(set(bravais_lattices)):
        path = mirror.DATASET_DIRECTORY / f'dataset_{bravais_lattice}.parquet'
        frame = pd.read_parquet(
            path, columns=['identifier', f'q2_{mirror.BROADENING_TAG}']
            )
        for identifier, q2 in zip(frame['identifier'], frame[f'q2_{mirror.BROADENING_TAG}']):
            lines[identifier] = np.asarray(q2, dtype=np.float64)
    return lines


def replay_window(q2_full, entry, rng):
    """The entry's twenty peaks, regenerated through the generator's own calls in its own order.

    Mirrors `run_fom_mirror.prepare_peak_list` for the hkl-free path. The order is not incidental:
    each mechanism consumes draws from the same stream, so reordering them or skipping a zero-count
    one would desynchronise everything after it.
    """
    window = select_peaks_with_dropout(
        q2_full, mirror.N_PEAKS, int(entry['n_dropout']), rng
        )
    q2 = window[np.newaxis].copy()
    if entry['q2_error_multiplier'] > 0:
        q2 = add_q2_error(q2, None, float(entry['q2_error_multiplier']), rng)
    if entry['n_contaminants'] > 0:
        q2 = add_contaminants(
            q2, None, int(entry['n_contaminants']), rng,
            max_attempts=mirror.CONTAMINANT_MAX_ATTEMPTS,
            low_angle_bias=float(entry['contaminant_bias']),
            )
    return window, q2


def replay_second_phase(q2, entry, partner_q2, rng):
    if entry['second_phase_lines'] <= 0:
        return q2
    return add_second_phase(
        q2, None, partner_q2, int(entry['second_phase_lines']), rng,
        low_angle_bias=float(entry['second_phase_bias']),
        )


def holdout_lines(q2_full, window, n_extra, error_multiplier, identifier, seed):
    """The first `n_extra` true-structure lines above the fitted window, with the error applied.

    Defined by q2 rather than by index, and strictly *above* the window rather than "whatever is
    left over". Under the heavy-dropout bundles `select_peaks_with_dropout` backfills from the
    surplus, so a leftover-based definition would hand those bundles the interior lines dropout had
    just removed -- an easier hold-out test than the other bundles get, and a confound sitting
    exactly on the condition axis the comparison runs along.
    """
    positive = q2_full[q2_full > 0]
    extra = positive[positive > window.max()][:n_extra]
    if extra.size == 0:
        return extra
    extra = extra[np.newaxis].copy()
    if error_multiplier > 0:
        # Its own stream. See the module docstring: sharing the entry's stream would move the
        # twenty peaks the frozen candidate pool was generated from.
        rng = np.random.default_rng(mirror.derived_seed(f'extra:{identifier}', seed))
        extra = add_q2_error(extra, None, float(error_multiplier), rng)
    return extra[0]


def build_bundle(entries, lines, n_extra, seed):
    """One row per entry: the hold-out peaks, and whether the window replayed exactly."""
    rows = []
    for entry in entries.to_dict('records'):
        identifier = entry['entry_id']
        q2_full = lines.get(identifier)
        if q2_full is None:
            rows.append(dict(entry_id=identifier, condition_bundle=entry['condition_bundle'],
                             q2_holdout=np.zeros(0), n_holdout=0, replayed=False,
                             bit_exact=False, max_window_error=np.nan))
            continue
        rng = np.random.default_rng(mirror.derived_seed(f'noise:{identifier}', seed))
        window, q2 = replay_window(q2_full, entry, rng)
        partner = entry.get('second_phase_partner')
        if entry['second_phase_lines'] > 0:
            partner_q2 = lines.get(partner)
            q2 = replay_second_phase(q2, entry, partner_q2, rng) if partner_q2 is not None else q2

        stored = np.asarray(entry['q2_obs'], dtype=np.float64)
        replayed = q2[0]
        error = (float(np.max(np.abs(replayed - stored)))
                 if replayed.shape == stored.shape else np.nan)

        extra = holdout_lines(
            q2_full, window, n_extra, entry['q2_error_multiplier'], identifier, seed
            )
        rows.append(dict(
            entry_id=identifier,
            condition_bundle=entry['condition_bundle'],
            q2_holdout=extra,
            n_holdout=int(extra.size),
            replayed=bool(np.isfinite(error) and error <= REPLAY_TOLERANCE),
            bit_exact=bool(error == 0.0),
            max_window_error=error,
            ))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description='Rebuild S10 Variant A hold-out peaks and gate the replay against the pool.'
        )
    parser.add_argument('--benchmark-dir',
                        default=os.path.join('mlindex', 'data', 'fom_benchmark'))
    parser.add_argument('--out-dir', default=os.path.join('mlindex', 'data', 'fom_cv'))
    parser.add_argument('--artifact-dir', default=os.path.join('docs', 'fom', 'artifacts'))
    parser.add_argument('--bundles', nargs='+', default=list(DEFAULT_BUNDLES))
    parser.add_argument('--n-extra', type=int, default=5,
                        help='hold-out peaks per entry, counted upward from the fitted window')
    parser.add_argument('--seed', type=int, default=12345,
                        help='the generator base seed; must match the pool manifest')
    parser.add_argument('--splits', nargs='+', default=['fom-train', 'fom-dev'],
                        help='fom-test is sealed until S15')
    parser.add_argument('--limit-entries', type=int, default=None)
    parser.add_argument('--tag', default='S10_cv')
    args = parser.parse_args()

    started = time.perf_counter()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.artifact_dir, exist_ok=True)

    entries = FomBenchmark.load_entries(args.benchmark_dir)
    entries = entries.loc[entries['split'].isin(set(args.splits))]
    lines = peak_tables(entries['bravais_lattice_true'])

    summary = []
    for bundle in args.bundles:
        subset = entries.loc[entries['condition_bundle'] == bundle]
        if args.limit_entries is not None:
            subset = subset.iloc[:args.limit_entries]
        frame = build_bundle(subset, lines, args.n_extra, args.seed)
        frame.to_parquet(
            os.path.join(args.out_dir, f'holdout_peaks_{bundle}.parquet'), index=False
            )
        applicable = frame['n_holdout'] >= args.n_extra
        summary.append(dict(
            bundle=bundle,
            n_entries=int(frame.shape[0]),
            n_replayed=int(frame['replayed'].sum()),
            fraction_replayed=float(frame['replayed'].mean()),
            fraction_bit_exact=float(frame['bit_exact'].mean()),
            max_window_error=float(np.nanmax(frame['max_window_error']))
            if frame['max_window_error'].notna().any() else np.nan,
            n_with_full_holdout=int(applicable.sum()),
            fraction_applicable=float(applicable.mean()),
            mean_n_holdout=float(frame['n_holdout'].mean()),
            ))
        print(f'{bundle:22s} replayed {summary[-1]["fraction_replayed"]:.4f}  '
              f'bit-exact {summary[-1]["fraction_bit_exact"]:.4f}  '
              f'max error {summary[-1]["max_window_error"]:.2e}  '
              f'applicable {summary[-1]["fraction_applicable"]:.4f}')

    table = pd.DataFrame(summary)
    table.to_csv(
        os.path.join(args.artifact_dir, f'{args.tag}_holdout_gate.csv'), index=False
        )
    passed = bool((table['fraction_replayed'] == 1.0).all())
    meta = dict(
        commit=commit_hash(), tag=args.tag, n_extra=args.n_extra, seed=args.seed,
        replay_tolerance=REPLAY_TOLERANCE,
        bundles=list(args.bundles), splits=list(args.splits), gate_passed=passed,
        seconds=time.perf_counter() - started,
        note=('The gate is the proof that the surplus peaks belong to the same pattern the frozen '
              'candidate pool was generated from. Variant A is not reportable without it.'),
        )
    with open(os.path.join(args.artifact_dir, f'{args.tag}_holdout_meta.json'),
              'w', encoding='utf-8') as handle:
        json.dump(meta, handle, indent=2)
    print(f'\ngate {"PASSED" if passed else "FAILED"} in {meta["seconds"]:.1f}s')
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
