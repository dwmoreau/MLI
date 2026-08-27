"""S06 -- freeze the split every campaign-2 number is reported against.

This is the one artefact that must never be redrawn. `docs/sync_record.sh` checksums it by name,
the dump driver refuses to invent a split when it is absent, and every later result is a statement
about the entries it assigns.

WHAT IT TAKES FROM CAMPAIGN 1, AND WHAT IT CHANGES.
`run_fom_mirror_analysis.assign_splits` is sound and is reproduced here: the split is by SOURCE
ENTRY -- `identifier` is one row per CIF entry -- stratified by (Bravais lattice x volume decile)
and permuted within each cell, 60/20/20. Three things are different.

* **The decile is frozen here and joined thereafter, never recomputed.** A within-lattice
  percentile rank RISES WHEN ROWS ARE DROPPED, so campaign 1's downstream recompute moved 114 of
  5 922 entries, every one of them upward, and shifted the hard stratum from the 286 entries the
  split was balanced over to the 298 the pipeline used (R14). Writing it into the manifest is half
  the fix; `FomMetrics.entry_context` joining it is the other half.

* **Per-lattice entry counts, because the source population is not uniform and cannot be made
  so.** After the two eligibility gates -- validation-split entries only, and at least 20 non-zero
  peaks -- five lattices are hard-capped: cF has 106 eligible entries, cI 156, cP 321, oI 372,
  oF 715. No sampling parameter reaches 1 430 there. The three lattices where the campaign's gains
  live (mP 40 425, mC 15 565, aP 32 093) have room to be oversampled instead, which is what the
  hard stratum's reachability target needs. Aggregates are unweighted (PROTOCOL section 3 rule 6),
  so an unbalanced draw distorts nothing as long as the composition is recorded -- which is what
  `S06_split_composition.csv` is for.

* **The arms nest.** S07 runs a wide core arm and a narrower mechanism arm. `sample_entries`
  draws `rng.choice(size=n)` over the whole eligible frame, so a 3 000 draw is NOT a subset of a
  20 000 draw and the two arms would not be paired. The mechanism arm is therefore taken as a
  stratified prefix of the core sample, the same way S05 nested the sparsity ladder, and arm
  membership is a manifest column so the driver reads its entry list from here rather than
  re-deriving it from a sampling parameter that can drift.

    python mlindex/scripts/run_fom_split_manifest.py --stage census
    python mlindex/scripts/run_fom_split_manifest.py --stage freeze
    python mlindex/scripts/run_fom_split_manifest.py --stage composition

Run it with the laptop env:
    /Users/DWMoreau/miniforge3/envs/mli/bin/python
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

from mlindex.model_training import FomPatterns
from mlindex.scripts.run_fom_prune_rerun import BRAVAIS_LATTICES

# Campaign 1's fractions, unchanged. DWMM declined a fourth "design" split on 2026-08-24, so
# `fom-dev` absorbs design decisions again and the mitigation is documentary: each results
# document states what was read off it.
SPLIT_FRACTIONS = {'fom-train': 0.6, 'fom-dev': 0.2, 'fom-test': 0.2}
N_VOLUME_DECILES = 10

# Where the campaign's gains live, and the only lattices in the hard stratum.
HARD_LATTICES = ('mP', 'mC', 'aP')

# MEASURED, on campaign 1's threshold-0 hard dump restricted to the generation cut of 1.5, which
# is the same population the sizing is about: 243 source entries, all of them mP/mC/aP at volume
# decile 8 or 9 (verified against campaign 1's frozen manifest), under four hard condition
# bundles. 188 of the 243 carry a correct candidate somewhere in the pool in at least one bundle.
#
# **0.774 per ENTRY, against 0.413 per (entry, bundle) cell.** Both are right and they answer
# different questions; the stratum is sized per entry because METRICS.md section 5 asks for ~100
# *entries* per split with a reachable solution. The S06 handoff's 0.349 is campaign 1's Benchmark
# A at a higher cut and is superseded by this.
#
# Two reasons it is conservative for campaign 2. The cut is the same 1.5, at which reachability
# is identical to threshold 0 on both arms -- 401/401 hard, 204/204 general -- so nothing is lost
# to the cut. And campaign 2's hard half is five bundles rather than four, with the sparsity
# ladder at N = 4 and 6 rather than campaign 1's 6 and 10, so it is less severe.
#
# Bounded by C2-R-005: whether a hard pattern is solvable at all is substantially a property of
# the search's random draw, so the sizing uses the lower Wilson bound rather than the point.
HARD_REACHABILITY = 188 / 243
HARD_REACHABILITY_LOW = 0.717
HARD_MIN_DECILE = 8

ARTIFACT_DIR = os.path.join('docs', 'fom_campaign2', 'artifacts')
MANIFEST_NAME = 'S06_split_manifest.parquet'


def volume_decile(frame, n_deciles=N_VOLUME_DECILES):
    """Volume deciles **within each true Bravais lattice**, 0-based.

    Byte-for-byte the rule campaign 1 froze its own manifest with, including the boundary
    behaviour that comes with it: percentile ranks run from 1/n to 1, so the lowest cell is one
    entry short and the highest one long. Reproduced rather than corrected, because a decile is
    only meaningful if it means the same thing in both campaigns' manifests.
    """
    ranked = frame.groupby('bravais_lattice')['volume_true'].rank(method='first', pct=True)
    return np.clip((ranked * n_deciles).astype(int), 0, n_deciles - 1)


def census(bravais_lattices):
    """Eligible entries per lattice, after both gates, without loading the whole table.

    The gates are `sample_entries`' own and they shape the population: `~train` keeps only the
    validation split of the datasets the candidate generators were trained on, and the 20-peak
    floor drops anything the fixed-length ONNX generators cannot accept. So the benchmark is a
    sample of *20-peak-or-longer validation entries*, not of crystals, and that belongs in the
    methods section rather than being discovered later.
    """
    import pyarrow.parquet as pq

    rows = []
    for bravais_lattice in bravais_lattices:
        path = FomPatterns.DATASET_DIRECTORY / f'dataset_{bravais_lattice}.parquet'
        handle = pq.ParquetFile(path)
        n_total = handle.metadata.num_rows
        n_validation = n_eligible = 0
        for batch in handle.iter_batches(batch_size=2000,
                                         columns=['train', f'q2_{FomPatterns.BROADENING_TAG}']):
            train = batch.column('train').to_numpy(zero_copy_only=False).astype(bool)
            peaks = batch.column(f'q2_{FomPatterns.BROADENING_TAG}').to_pylist()
            validation = ~train
            n_validation += int(validation.sum())
            for position in np.nonzero(validation)[0]:
                if np.count_nonzero(np.asarray(peaks[position])) >= FomPatterns.N_PEAKS:
                    n_eligible += 1
        rows.append({'bravais_lattice': bravais_lattice, 'n_total': n_total,
                     'n_validation': n_validation, 'n_eligible': n_eligible,
                     'is_hard_lattice': bravais_lattice in HARD_LATTICES})
    return pd.DataFrame(rows)


def target_counts(census_frame, target_per_lattice, target_hard):
    """How many entries to draw per lattice, and what the source population allows.

    `n_capped` is not an implementation detail: it is the bound every per-lattice claim on cF,
    cI, cP, oI and oF carries, and it is a property of the source data rather than of the design.
    """
    frame = census_frame.copy()
    wanted = np.where(frame['is_hard_lattice'], target_hard, target_per_lattice)
    frame['n_wanted'] = wanted
    frame['n_drawn'] = np.minimum(wanted, frame['n_eligible'])
    frame['is_capped'] = frame['n_drawn'] < frame['n_wanted']
    return frame


def assign_splits(manifest, seed):
    """By source entry, stratified on (lattice, decile), permuted within each cell.

    Reproduces `run_fom_mirror_analysis.assign_splits`. Stratifying on the decile as well as the
    lattice is what stops a split from missing a volume stratum entirely -- which matters here
    more than it did for campaign 1, because five lattices contribute barely a hundred entries.
    """
    rng = np.random.default_rng(seed)
    names = list(SPLIT_FRACTIONS)
    fractions = np.array([SPLIT_FRACTIONS[name] for name in names])
    assignment = np.empty(manifest.shape[0], dtype=object)
    for _, group in manifest.groupby(['bravais_lattice', 'volume_decile'], sort=True):
        order = rng.permutation(group.shape[0])
        boundaries = np.round(np.cumsum(fractions) * group.shape[0]).astype(int)
        cell = np.empty(group.shape[0], dtype=object)
        start = 0
        for name, stop in zip(names, boundaries):
            cell[order[start:stop]] = name
            start = stop
        # A cell of one or two entries can leave a slot empty after rounding; those go to
        # fom-train, which is the split that is allowed to absorb an odd entry.
        cell[pd.isna(cell)] = 'fom-train'
        assignment[manifest.index.get_indexer(group.index)] = cell
    return assignment


def assign_arms(manifest, mechanism_fraction, seed):
    """Arm membership, nested: every mechanism entry is also a core entry.

    Drawn stratified on (lattice, decile, split) so the narrower arm keeps the wider one's shape
    -- and drawn from a separate generator, so changing the mechanism arm's size cannot shift the
    split assignment of a single entry.
    """
    rng = np.random.default_rng(seed + 1)
    arm = np.array(['core'] * manifest.shape[0], dtype=object)
    for _, group in manifest.groupby(['bravais_lattice', 'split'], sort=True):
        n_group = group.shape[0]
        n_mechanism = int(round(n_group * mechanism_fraction))
        if n_mechanism == 0:
            continue
        # Systematic on a decile-ordered list, not a draw per (lattice, decile, split) cell.
        # Rounding inside those cells is what makes the naive version undersample: cF contributes
        # about two entries per cell, `round(2 * 0.15)` is zero, and the lattice ends up with half
        # the share it was asked for -- exactly on the five lattices whose entry counts are
        # already capped by the source population and can least afford it.
        #
        # `i * n // n_mechanism` is strictly increasing for n_mechanism <= n, so the count is
        # exact, and taking it along a decile-sorted order spreads the picks evenly across
        # deciles. The permutation is the tie-break inside a decile, so the choice is still
        # random where it has a choice and deterministic in what it guarantees.
        jitter = rng.permutation(n_group)
        order = np.lexsort((jitter, group['volume_decile'].to_numpy()))
        picks = (np.arange(n_mechanism) * n_group) // n_mechanism
        positions = manifest.index.get_indexer(group.index)[order[picks]]
        arm[positions] = 'core+mechanism'
    return arm


def freeze(args):
    counts = target_counts(census(args.bravais_lattices.split(',')),
                           args.target_per_lattice, args.target_hard)
    frames = []
    for row in counts.itertuples():
        entries = FomPatterns.sample_entries(
            row.bravais_lattice, int(row.n_drawn), args.seed,
            columns=['identifier', 'database', 'bravais_lattice', 'lattice_system', 'train',
                     'reindexed_volume', f'q2_{FomPatterns.BROADENING_TAG}'])
        frames.append(pd.DataFrame({
            'identifier': entries['identifier'].to_numpy(),
            'source_db': entries['database'].to_numpy(),
            'bravais_lattice': entries['bravais_lattice'].to_numpy(),
            'lattice_system': entries['lattice_system'].to_numpy(),
            'volume_true': entries['reindexed_volume'].to_numpy(dtype=np.float64),
            }))
    manifest = pd.concat(frames, ignore_index=True)
    manifest['volume_decile'] = volume_decile(manifest)
    manifest['split'] = assign_splits(manifest, args.seed)
    manifest['arm'] = assign_arms(manifest, args.mechanism_fraction, args.seed)

    check_manifest(manifest)
    artifact_dir = Path(BASE) / args.artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / MANIFEST_NAME
    manifest.to_parquet(path, index=False)
    counts.to_csv(artifact_dir / 'S06_split_census.csv', index=False)

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(manifest.groupby(['bravais_lattice', 'split']).size().unstack(fill_value=0)
          .to_string())
    print(f'\n{manifest.shape[0]} entries, '
          f"{int((manifest['arm'] == 'core+mechanism').sum())} also in the mechanism arm")
    print(f'{path}\nsha256 {digest}')
    with open(artifact_dir / 'S06_split_manifest.sha256', 'w', encoding='utf-8') as handle:
        handle.write(f'{digest}  {MANIFEST_NAME}\n')
    return manifest


def check_manifest(manifest):
    """The assertions gate 3 asks for, in code rather than in a results document."""
    duplicated = manifest['identifier'].duplicated()
    if duplicated.any():
        raise SystemExit(f'{int(duplicated.sum())} identifiers appear more than once; the split '
                         'is by source entry and an entry cannot be in two splits')
    per_entry = manifest.groupby('identifier')['split'].nunique()
    if (per_entry > 1).any():
        raise SystemExit('an identifier carries two splits')
    unknown = set(manifest['split']) - set(SPLIT_FRACTIONS)
    if unknown:
        raise SystemExit(f'unexpected split labels: {sorted(unknown)}')
    if manifest['volume_decile'].isna().any():
        raise SystemExit('volume_decile has nulls; it is the column S06 exists to freeze')
    # Balance: every (lattice, decile) cell with enough entries must reach all three splits, or
    # the split is not stratified in the way every later per-lattice number assumes.
    cells = manifest.groupby(['bravais_lattice', 'volume_decile'])['split'].nunique()
    populated = manifest.groupby(['bravais_lattice', 'volume_decile']).size()
    unbalanced = cells[(cells < 3) & (populated >= 5)]
    if not unbalanced.empty:
        raise SystemExit(f'{len(unbalanced)} (lattice, decile) cells of 5+ entries do not reach '
                         f'all three splits: {unbalanced.index.tolist()[:5]}')
    return True


def composition(args):
    """Entries per (lattice, decile, split, arm), and what the hard stratum projects to.

    The projection is the gate: "projected reachable hard entries per split is at or above ~100,
    computed from the measured reachability rate at the chosen cut -- not from a guess. If the
    arithmetic does not reach it, the deliverable is a resized entry count, not a redefined
    stratum."
    """
    artifact_dir = Path(BASE) / args.artifact_dir
    manifest = pd.read_parquet(artifact_dir / MANIFEST_NAME)
    composition_frame = (manifest.groupby(['bravais_lattice', 'volume_decile', 'split', 'arm'])
                         .size().rename('n_entries').reset_index())
    composition_frame.to_csv(artifact_dir / 'S06_split_composition.csv', index=False)

    hard = manifest[manifest['bravais_lattice'].isin(HARD_LATTICES)
                    & (manifest['volume_decile'] >= HARD_MIN_DECILE)]
    rows = []
    for split in SPLIT_FRACTIONS:
        n_hard = int((hard['split'] == split).sum())
        rows.append({
            'split': split,
            'n_hard_entries': n_hard,
            'reachability_point': args.reachability,
            'reachability_low': args.reachability_low,
            'projected_reachable_point': round(n_hard * args.reachability, 1),
            'projected_reachable_low': round(n_hard * args.reachability_low, 1),
            'meets_target_100': bool(n_hard * args.reachability_low >= 100),
            })
    projection = pd.DataFrame(rows)
    projection.to_csv(artifact_dir / 'S06_hard_stratum_projection.csv', index=False)
    print(projection.to_string(index=False))
    print(f"\nhard stratum: {hard.shape[0]} entries "
          f"({', '.join(f'{bl} {int((hard.bravais_lattice == bl).sum())}' for bl in HARD_LATTICES)})")
    print('campaign 1 for contrast: 298 hard entries, 104 reachable in total, split 64/16/24')
    return projection


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--stage', choices=['census', 'freeze', 'composition'],
                        default='freeze')
    parser.add_argument('--bravais-lattices', default=','.join(BRAVAIS_LATTICES))
    parser.add_argument('--target-per-lattice', type=int, default=1200,
                        help='Entries wanted per ordinary lattice, capped by availability')
    parser.add_argument('--target-hard', type=int, default=1400,
                        help='Entries wanted per hard lattice (mP, mC, aP). The default is what '
                             'the sizing below needs for ~100 reachable hard entries in the '
                             'reporting split, with headroom for the reachability draw')
    parser.add_argument('--reachability', type=float, default=HARD_REACHABILITY,
                        help='Measured per-entry reachability of the hard stratum at the '
                             'generation cut. NOT a guess -- see the constant')
    parser.add_argument('--reachability-low', type=float, default=HARD_REACHABILITY_LOW,
                        help='Lower Wilson bound, which the sizing is done against')
    parser.add_argument('--mechanism-fraction', type=float, default=0.15,
                        help='Share of the core arm that also runs the mechanism bundles')
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--artifact-dir', default=ARTIFACT_DIR)
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    if args.stage == 'composition':
        composition(args)
        return
    if args.stage == 'census':
        frame = census(args.bravais_lattices.split(','))
        print(frame.to_string(index=False))
        print(f"\ntotal eligible {int(frame['n_eligible'].sum())}")
        artifact_dir = Path(BASE) / args.artifact_dir
        artifact_dir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(artifact_dir / 'S06_split_census.csv', index=False)
        return
    freeze(args)


if __name__ == '__main__':
    main()
