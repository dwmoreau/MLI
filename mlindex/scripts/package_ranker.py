"""Turn a fitted ranker into the directory `mlindex.run` loads.

Reads a fit directory `run_ranker --stage fit` wrote (`model.joblib`, `calibrators.npz`,
`specification.json`) and writes, into an empty directory:

    model.onnx            the classifier, as one exact TreeEnsemble node (`IOManagers`)
    calibrators.npz       the per-lattice isotonic knots, unchanged
    group_frequency.csv   the extinction-group frequency table the fit's export read
    specification.json    the inputs in matrix order, the encoding, and where the model came from

    python -m mlindex.scripts.package_ranker \
        --fit-dir ranker/fits/<commit>/ordinal_lr0.04_leaves127_iter1600_seed12345 \
        --group-frequency group_frequency.csv --cut 3.5 --depth 20 \
        --out mlindex/models/ranker_1

`--cut` and `--depth` are the export's: the M20 cut and the per-lattice depth of the rows the
model was fitted and calibrated on. They are recorded, not applied.
"""

import argparse
import json
import shutil
from pathlib import Path

from mlindex.model_training.FomCombiner import FomCombiner
from mlindex.utilities.IOManagers import SKLearnManager
from mlindex.utilities.Ranker import read_group_frequency
from mlindex.utilities.Ranker import write_calibrators


def package(fit_dir, group_frequency, cut, depth, out):
    """Write the shipped ranker directory `out` from the fit at `fit_dir`."""
    fit_dir, out = Path(fit_dir), Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'{out} is not empty; a packaged ranker is never overwritten')
    combiner = FomCombiner.load(fit_dir)
    read_group_frequency(group_frequency)
    out.mkdir(parents=True, exist_ok=True)
    SKLearnManager(filename=str(out / 'model'), model_type='onnx').save(
        model=combiner.model, n_features=len(combiner.matrix_names))
    write_calibrators(out / 'calibrators.npz', combiner.calibrators)
    shutil.copyfile(group_frequency, out / 'group_frequency.csv')
    specification = dict(
        features=list(combiner.features), matrix_names=list(combiner.matrix_names),
        encoding=combiner.encoding, onnx='model.onnx', group_frequency='group_frequency.csv',
        cut=float(cut), depth=int(depth), seed=combiner.meta.get('seed'),
        source_fit=fit_dir.name, meta=combiner.meta)
    with open(out / 'specification.json', 'w', encoding='utf-8') as handle:
        json.dump(specification, handle, indent=2)
    return out


def build_parser():
    parser = argparse.ArgumentParser(
        description='Turn a fitted ranker into the directory mlindex.run loads.',
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument('--fit-dir', required=True, metavar='DIR',
                        help='A fit directory from run_ranker --stage fit.')
    parser.add_argument('--group-frequency', required=True, metavar='CSV',
                        help='The table the fit\'s export read (make_group_frequency).')
    parser.add_argument('--cut', required=True, type=float, metavar='M20',
                        help='The M20 cut of the export the model was fitted on.')
    parser.add_argument('--depth', required=True, type=int, metavar='N',
                        help='The per-lattice depth of the export\'s evaluation rows.')
    parser.add_argument('--out', required=True, metavar='DIR', help='An empty directory.')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    out = package(args.fit_dir, args.group_frequency, args.cut, args.depth, args.out)
    print(f'wrote {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
