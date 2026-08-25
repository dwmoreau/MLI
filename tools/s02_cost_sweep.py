#!/usr/bin/env python
"""S02's price table: what each ported merit costs, in get_M20 units.

DEVELOPMENT TOOL -- not part of the installed package.

Regenerates `docs/fom_campaign2/artifacts/S02_zoo_cost.csv` in one command, so the table in
`S02_zoo_port.md` is reproducible rather than transcribed (PROTOCOL section 5).

WHAT THIS IS FOR, AND WHAT IT IS NOT FOR. Execution time is not an exclusion criterion in
campaign 2 (DWMM, 2026-08-25): a merit that outperforms the rest is kept whatever it costs.
Campaign 2 also does not change the inner loop, and the prune and the final ranking each read
a merit once per candidate. So the only place a merit's cost still multiplies is S11's
extinction-group assignment, which evaluates one over up to 68 groups per candidate. That is
the number worth reading here. Every other row is archival -- it exists so that nobody ever
again quotes `../fom_campaign1/artifacts/S06_zoo_cost.csv`, whose every expensive entry was
an implementation rather than a property of the merit (C2-F-001).

`--revision 7af8bfc` is the "before": the module as it was immediately BEFORE the numba
rewrite. It is the right baseline rather than this branch's own pre-port tip, which does not
carry the hold-out or assignment functions at all and so cannot be priced on the same set.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import repro_fom_zoo

ROOT = Path(subprocess.run(['git', 'rev-parse', '--show-toplevel'],
                           cwd=str(Path(__file__).resolve().parent),
                           capture_output=True, check=True, text=True).stdout.strip())
DEFAULT_CSV = ROOT/'docs'/'fom_campaign2'/'artifacts'/'S02_zoo_cost.csv'

# (capture, regime, lattice, origin, sizes)
SWEEP = (
    ('captures/s02/zoo_mP.npz', 'inner loop', 'mP (monoclinic)',
     'S02 2026-08-25, 11bmb_3844, seed 12345, max-rows 20000, commit 1cbfe9b', (1000, 10000)),
    ('captures/s02/zoo_cP.npz', 'inner loop', 'cP (cubic)',
     'S02 2026-08-25, 11bmb_3844, seed 12345, max-rows 20000, commit 1cbfe9b', (1000, 10000)),
    ('captures/pool_mP.npz', 'frozen pool', 'mP (monoclinic)',
     'campaign 1, provenance unrecorded', (1000, 10000)),
    ('captures/pool_aP.npz', 'frozen pool', 'aP (triclinic)',
     'campaign 1, provenance unrecorded', (1000, 10000)),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--csv', type=str, default=str(DEFAULT_CSV))
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--revisions', type=str, nargs='+', default=[None, '7af8bfc'],
                        help='None means the working tree; 7af8bfc is the pre-rewrite module')
    args = parser.parse_args()

    if os.path.exists(args.csv):
        os.remove(args.csv)                      # the sweep owns the file; append is per-call
    for capture, regime, lattice, origin, sizes in SWEEP:
        path = ROOT/capture
        if not path.exists():
            print(f'SKIPPED {capture}: not on disk')
            continue
        for revision in args.revisions:
            revision = None if revision in (None, 'None', 'working-tree') else revision
            repro_fom_zoo.cost(argparse.Namespace(
                capture_file=str(path), sizes=list(sizes), repeats=args.repeats,
                revision=revision, csv=args.csv, regime=regime, lattice=lattice,
                capture_origin=origin,
                ))
    print(f'\nS02 price table: {args.csv}')


if __name__ == '__main__':
    main()
