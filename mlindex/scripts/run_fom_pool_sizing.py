"""S06 -- size Benchmark B: how many rows, how much disk, and what K the subsampler needs.

S05 handed this forward as the one thing standing between the harness and the generation run.
Projected from its own gate run, 20 000 crystals at nine bundles is 1.5 billion survivor rows
(213 GB) and 7.5 billion pre-deduplication rows (670 GB) -- 883 GB against a 1 TB budget, before
labelling and consolidation write a second copy of the survivor stream. Something has to come out,
and which thing has to be a measurement rather than a guess.

WHY THE DEFAULT K WAS NEARLY A NO-OP. 1.5 billion rows over 20 000 x 9 cells is ~8 300 survivors
per cell across fourteen lattices -- under 600 per lattice. `--top-k 500` would therefore have
retained almost every row while the manifest advertised subsampling. K has to be set against the
measured per-lattice pool size, and the cost of a given K is not K but the size of the UNION of
the top K over every reported merit, because a row is kept if any merit ranks it highly.

TWO SOURCES, AND THEY ANSWER DIFFERENT HALVES.
  * `--stage merits` reads a persisted per-candidate merit table -- S03's threshold-0 capture is
    the one on disk, with all six campaign-2 merits at two stages -- and measures how much bigger
    the union is than K. That is a property of how the merits disagree, and it is the number no
    other artefact carries.
  * `--stage project` takes the per-lattice survivor counts the S06 iteration pilot measured at
    the generation cut, applies the union multiplier and the negative rate, and reports rows and
    bytes for a proposed grid.

The pilot's pools are POST-deduplication and the merit table is PRE-deduplication, so the
multiplier is carried across a boundary. It is reported with that stated rather than hidden: the
pre-deduplication population is larger and more redundant, so merits have more room to disagree
in it, which makes the multiplier an upper bound and the projection conservative.

    python mlindex/scripts/run_fom_pool_sizing.py --stage merits --arm general
    python mlindex/scripts/run_fom_pool_sizing.py --stage project

Run it with the laptop env:
    /Users/DWMoreau/miniforge3/envs/mli/bin/python
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

# S03's cut, which the benchmark generates behind (C2-Q-001).
CUT = 1.5

# The campaign-2 reduced merit set, at the later capture stage. Six columns from four calls, the
# whole classical core at 6.0x get_M20 (artifacts/S02_zoo_cost.csv). `Minfo` is deliberately not
# here: it ranks below a constant score (C2-F-012) and S02 dropped it from the schema.
MERIT_COLUMNS = ('M20_C', 'M_tilde_C', 'M_rev_C', 'M_sym_C', 'X_N_C', 'n_over_C', 'max_gap_C')

K_GRID = (50, 100, 200, 500)

# MEASURED on a real schema-v3 shard (S06, 2026-08-27): 1 514 candidate rows in 210 026 bytes.
# Used only to turn a row count into a disk figure, and stated rather than assumed. The shard is
# small, so parquet's per-file and per-column-chunk overhead is amortised over fewer rows than it
# will be at scale -- this is therefore an upper bound and the projection is conservative. The
# pre-deduplication stream measures 81.8 B/row on the same run, being narrower.
DEFAULT_BYTES_PER_ROW = 138.7
PREDOWNSAMPLE_BYTES_PER_ROW = 81.8

ARM_DIRECTORIES = {
    'general': os.path.join('mlindex', 'data', 'fom_prune_criterion', 'general'),
    'hard': os.path.join('mlindex', 'data', 'fom_prune_criterion', 'hard'),
    }

ARTIFACT_DIR = os.path.join('docs', 'fom_campaign2', 'artifacts')
PILOT_ROOT = os.path.join('mlindex', 'characterization', 'fom', 'iteration_pilot')


def union_top_k(frame, merit_columns, k_grid):
    """|union of the top K by each merit|, for each K, over one candidate pool.

    Ranks rather than `nlargest`, so a tie at the K-th place is resolved identically for every
    merit and cannot silently change the union's size.
    """
    n = frame.shape[0]
    sizes = {}
    ranks = {column: frame[column].rank(method='first', ascending=False).to_numpy()
             for column in merit_columns if column in frame.columns}
    for k in k_grid:
        if n <= k:
            sizes[k] = n
            continue
        keep = np.zeros(n, dtype=bool)
        for values in ranks.values():
            keep |= values <= k
        sizes[k] = int(keep.sum())
    return sizes


def measure_merits(args):
    """Union sizes per (lattice, K), over the pools of a persisted merit table."""
    directory = Path(BASE) / ARM_DIRECTORIES[args.arm]
    files = sorted(glob.glob(str(directory / 'merits_*.parquet')))
    if not files:
        raise SystemExit(f'no merit tables under {directory}')
    columns = (['entry_id', 'condition_bundle', 'bravais_lattice', 'm20_at_prune', 'is_correct']
               + list(MERIT_COLUMNS))

    rows = []
    for path in files:
        frame = pd.read_parquet(path, columns=columns)
        frame = frame[frame['m20_at_prune'] >= args.cut]
        if frame.empty:
            continue
        for key, pool in frame.groupby(['entry_id', 'condition_bundle', 'bravais_lattice'],
                                       sort=False):
            sizes = union_top_k(pool, MERIT_COLUMNS, K_GRID)
            record = {'entry_id': key[0], 'condition_bundle': key[1], 'bravais_lattice': key[2],
                      'pool_size': int(pool.shape[0]),
                      'n_correct': int(pool['is_correct'].to_numpy(dtype=bool).sum())}
            record.update({f'union_k{k}': size for k, size in sizes.items()})
            rows.append(record)
        del frame
    pools = pd.DataFrame(rows)

    summary = []
    for lattice in ['ALL'] + sorted(pools['bravais_lattice'].unique()):
        subset = pools if lattice == 'ALL' else pools[pools['bravais_lattice'] == lattice]
        row = {'arm': args.arm, 'bravais_lattice': lattice, 'n_pools': int(subset.shape[0]),
               'pool_mean': float(subset['pool_size'].mean()),
               'pool_median': float(subset['pool_size'].median()),
               'correct_rate': float(subset['n_correct'].sum() / subset['pool_size'].sum())}
        for k in K_GRID:
            union = subset[f'union_k{k}']
            row[f'union_k{k}_mean'] = float(union.mean())
            # How much more a K costs than K itself, which is the number the sizing needs.
            row[f'union_k{k}_multiplier'] = float(union.mean() / min(k, subset['pool_size'].mean()))
            row[f'retention_k{k}'] = float(union.sum() / subset['pool_size'].sum())
        summary.append(row)
    summary = pd.DataFrame(summary)

    artifact_dir = Path(BASE) / args.artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(artifact_dir / f'S06_pool_topk_union_{args.arm}.csv', index=False)
    with pd.option_context('display.width', 220, 'display.max_columns', 40):
        print(summary[['bravais_lattice', 'n_pools', 'pool_mean', 'correct_rate']
                      + [f'union_k{k}_mean' for k in K_GRID]
                      + [f'retention_k{k}' for k in K_GRID]].to_string(index=False))
    return summary


def load_pilot_pools(pilot_root, scale=1.0):
    """Per-lattice survivor counts at the generation cut, from the S06 iteration pilot.

    The only measured post-deduplication pool sizes campaign 2 has, and the reason the pilot
    writes pool sizes at all rather than only ceilings.
    """
    from mlindex.scripts.run_fom_iteration_pilot import scale_tag

    path = Path(BASE) / pilot_root / f'lattice_{scale_tag(scale)}.parquet'
    if not path.exists():
        raise SystemExit(f'{path} is missing; run the iteration pilot before projecting')
    frame = pd.read_parquet(path)
    return (frame.groupby('bravais_lattice')['pool_size']
            .agg(['mean', 'median', 'count']).rename(columns={'count': 'n_cells'}))


def project(args):
    """Rows and bytes for a proposed grid, from measured pool sizes and a chosen K."""
    pools = load_pilot_pools(args.pilot_root, args.scale)
    manifest_path = Path(BASE) / args.artifact_dir / 'S06_split_manifest.parquet'
    if manifest_path.exists():
        manifest = pd.read_parquet(manifest_path)
        entries_per_lattice = manifest.groupby('bravais_lattice').size()
        mechanism = (manifest[manifest['arm'].astype(str).str.contains('mechanism')]
                     .groupby('bravais_lattice').size())
    else:
        raise SystemExit(f'{manifest_path} is missing; freeze the split before projecting')

    union = None
    union_path = Path(BASE) / args.artifact_dir / f'S06_pool_topk_union_{args.union_arm}.csv'
    if union_path.exists():
        union = pd.read_csv(union_path).set_index('bravais_lattice')

    # EVERY pattern is indexed against ALL FOURTEEN lattices -- that is what `run.py` does and
    # what the benchmark stores -- so a lattice's pool contributes to every cell in the run, not
    # only to the cells whose *true* lattice it is. Getting this wrong understates the pool by
    # about the number of lattices; the entry counts below are per true lattice and are what set
    # the number of CELLS, and the pool sizes are per candidate lattice and are what set the rows
    # inside each cell. They multiply across, not elementwise.
    n_cells = int(sum(int(entries_per_lattice.get(lattice, 0)) * args.core_bundles
                      + int(mechanism.get(lattice, 0)) * args.mechanism_bundles
                      for lattice in entries_per_lattice.index))

    rows = []
    for lattice, pool in pools.iterrows():
        n_core = int(entries_per_lattice.get(lattice, 0))
        n_mechanism = int(mechanism.get(lattice, 0))
        pool_mean = float(pool['mean'])
        if union is not None and lattice in union.index:
            kept = float(union.loc[lattice, f'union_k{args.top_k}_mean'])
        else:
            kept = min(args.top_k * args.union_multiplier, pool_mean)
        kept = min(kept, pool_mean)
        # The rest, sampled, plus the correct candidates that are kept unconditionally. The
        # correct rate is small enough that it does not drive the arithmetic, but it is included
        # rather than waved away.
        sampled = max(0.0, pool_mean - kept) * args.negative_rate
        retained = kept + sampled
        rows.append({
            'bravais_lattice': lattice,
            'n_core_entries': n_core,
            'n_mechanism_entries': n_mechanism,
            # This lattice's share of every cell in the run, not of its own entries' cells.
            'pool_mean': pool_mean,
            'retained_per_cell': retained,
            'retention': retained / pool_mean if pool_mean else float('nan'),
            'rows_full': n_cells * pool_mean,
            'rows_retained': n_cells * retained,
            'gb_full': n_cells * pool_mean * args.bytes_per_row / 1e9,
            'gb_retained': n_cells * retained * args.bytes_per_row / 1e9,
            })
    projection = pd.DataFrame(rows)
    total = projection[['pool_mean', 'retained_per_cell', 'rows_full', 'rows_retained',
                        'gb_full', 'gb_retained']].sum()
    total['bravais_lattice'] = 'TOTAL'
    total['n_core_entries'] = projection['n_core_entries'].sum()
    total['n_mechanism_entries'] = projection['n_mechanism_entries'].sum()
    total['retention'] = total['retained_per_cell'] / total['pool_mean']
    projection = pd.concat([projection, pd.DataFrame([total])], ignore_index=True)
    projection.attrs['n_cells'] = n_cells

    artifact_dir = Path(BASE) / args.artifact_dir
    projection.to_csv(artifact_dir / 'S06_pool_projection.csv', index=False)
    with pd.option_context('display.width', 220, 'display.max_columns', 40):
        print(projection.to_string(index=False))
    print(f'\n{n_cells} cells (entry x bundle patterns), each indexed against all 14 lattices')
    print(f'K={args.top_k}, negative rate {args.negative_rate}, '
          f'{args.bytes_per_row:g} B/row, core bundles {args.core_bundles}, '
          f'mechanism bundles {args.mechanism_bundles}')
    print(f"survivors per cell: {total['pool_mean']:.0f} whole, "
          f"{total['retained_per_cell']:.0f} retained ({total['retention']:.1%})")
    print(f"survivor stream: {total['gb_full']:.0f} GB whole, "
          f"{total['gb_retained']:.0f} GB subsampled")
    return projection


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--stage', choices=['merits', 'project'], default='project')
    parser.add_argument('--arm', choices=sorted(ARM_DIRECTORIES), default='general')
    parser.add_argument('--union-arm', default='general',
                        help='Which measured union table the projection reads')
    parser.add_argument('--cut', type=float, default=CUT)
    parser.add_argument('--top-k', type=int, default=200)
    parser.add_argument('--negative-rate', type=float, default=0.05)
    parser.add_argument('--union-multiplier', type=float, default=2.5,
                        help='Fallback when no measured union table covers a lattice')
    parser.add_argument('--bytes-per-row', type=float, default=DEFAULT_BYTES_PER_ROW)
    parser.add_argument('--core-bundles', type=int, default=5)
    parser.add_argument('--mechanism-bundles', type=int, default=4)
    parser.add_argument('--scale', type=float, default=1.0,
                        help='Which pilot arm supplies the pool sizes')
    parser.add_argument('--pilot-root', default=PILOT_ROOT)
    parser.add_argument('--artifact-dir', default=ARTIFACT_DIR)
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    if args.stage == 'merits':
        measure_merits(args)
    else:
        project(args)


if __name__ == '__main__':
    main()
