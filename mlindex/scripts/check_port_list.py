"""Check that what a handoff's port table names as "keep" is actually on the branch.

RESEARCH CODE THAT NEEDS TO BE DELETED -- it compares a checkout against a handoff's prose, which
is a property of this porting pass and not of the package. P18 deletes it with the pass.

Every session from P05 to P12 ports from `fom_campaign2`, and a handoff's port table is the list of
what to take. Nothing checked that the list was taken. P04b's came out sixteen of twenty-seven
absent, and one of those was `reduced_merits` -- the campaign's merit-sidecar computation, which
groups by extinction group and truncates the peak list to the lattice's own count. Written fresh
instead, it did neither, every `M_sym` in the first floor run was wrong, and it cost two cluster
runs to find.

The output is a list to ACCOUNT FOR, not a pass or a fail. A symbol is absent for one of three
reasons and the session says which:

  present            nothing to do
  renamed            say what to, so the next reader can follow it
  dropped            say who owns it now and where that was decided

    python -m mlindex.scripts.check_port_list --symbols reduced_merits,mcnemar,derive_flags
    python -m mlindex.scripts.check_port_list --symbols-file docs/.../P05_keep.txt
"""

import argparse
import subprocess
import sys
import uuid


def is_present(symbol, paths=('mlindex/',)):
    """Whether `symbol` appears as a whole word in tracked source under `paths`."""
    result = subprocess.run(['git', 'grep', '-lw', symbol, '--', *paths],
                            capture_output=True, text=True)
    return bool(result.stdout.strip())


def absent_sentinel():
    """A symbol that cannot be in the tree, generated rather than written down.

    The first sentinel was a literal in this file, so once the file was tracked `git grep` found
    it here and the checker declared itself broken. A random name cannot appear in any source,
    including this one -- which is the only way to be sure the negative case is really negative.
    """
    return f'absent_{uuid.uuid4().hex}'


def verify_the_checker():
    """Refuse to report until the check can be shown to answer both ways.

    The first version of this used `\\b` in a `git grep -E` pattern, which is not a word boundary
    in POSIX ERE, so every symbol read as absent -- twenty-seven of twenty-seven, including ones
    the caller was importing. A check that can only say "no" is worse than no check, because its
    output looks like a finding.
    """
    if not is_present('mcnemar'):
        raise SystemExit('The checker is broken: a symbol known to be present reads as absent.')
    if is_present(absent_sentinel()):
        raise SystemExit('The checker is broken: an impossible symbol reads as present.')


def build_parser():
    parser = argparse.ArgumentParser(
        description='Check a handoff port list against the branch, symbol by symbol.',
        epilog='An absence is not a failure. It is present, renamed, or dropped by a recorded '
               'decision, and the session says which in its artefact.')
    parser.add_argument('--symbols', default=None, metavar='A,B',
                        help='Comma-separated symbols the port table names as keep.')
    parser.add_argument('--symbols-file', default=None, metavar='PATH',
                        help='One symbol per line; blank lines and # comments ignored.')
    parser.add_argument('--paths', default='mlindex/', metavar='A,B',
                        help='Where to look (default: mlindex/).')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    symbols = [s for s in (args.symbols or '').split(',') if s]
    if args.symbols_file:
        with open(args.symbols_file, encoding='utf-8') as handle:
            symbols += [line.split('#')[0].strip() for line in handle
                        if line.split('#')[0].strip()]
    if not symbols:
        raise SystemExit('Give --symbols or --symbols-file.')

    verify_the_checker()
    paths = tuple(p for p in args.paths.split(',') if p)
    absent = [s for s in symbols if not is_present(s, paths)]

    print(f'{len(symbols) - len(absent)} of {len(symbols)} present under {", ".join(paths)}')
    if absent:
        print('\nabsent -- account for each as present, renamed, or dropped:')
        for symbol in absent:
            print(f'  {symbol}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
