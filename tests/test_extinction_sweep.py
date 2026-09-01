"""S11's offline sweep: the sidecar contract, and the group index that means nothing on its own.

The sweep re-runs the extinction argmax under every criterion and persists the winner per
candidate. What it stores is an INDEX into the lattice's group-key list, not the key itself --
43 million repeated strings is the alternative -- so the index and the key list in `_meta.json`
have to stay in step.

What could be silently wrong here: the index is read against the wrong key list. `merit_at_prune`
failed exactly this way, four of seven entries mislabelled because a frame builder sorted the
names while the manifest wrote capture order, and nothing was able to detect it (C2-F-067).
Position is the only label an index has, so these tests pin the round trip.
"""
import os
import sys

import numpy as np
import pytest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from mlindex.model_training.FomBenchmark import extinction_group_sweep
from mlindex.scripts.run_fom_extinction_sweep import (
    CANDIDATE_COLUMNS, JOIN_KEYS, SKIP_LATTICES, candidate_tasks,
    )
from mlindex.utilities.ExtinctionCounts import LATTICE_SYSTEM, get_absence_counts
from mlindex.utilities.FigureOfMerits import EXTINCTION_CRITERIA
from mlindex.utilities.Q2Calculator import Q2Calculator
from mlindex.utilities.SpaceGroups import get_spacegroup_hkl_ref

POOL = os.path.join(BASE, 'mlindex', 'data', 'fom_full_c2_pool')


def _sweep(bravais_lattice='mP', lattice_system='monoclinic', n=8, seed=3):
    rng = np.random.default_rng(seed)
    hkl_ref = np.load(os.path.join(BASE, 'mlindex', 'models', f'{lattice_system}_1', 'data',
                                   f'hkl_ref_{bravais_lattice}.npy'))
    xnn_true = np.array([[1/8.0**2, 1/9.0**2, 1/11.0**2, 0.01]])
    q2_ref = Q2Calculator(lattice_system=lattice_system, hkl=hkl_ref, tensorflow=False,
                          representation='xnn').get_q2(xnn_true)[0]
    q2_obs = np.sort(q2_ref[q2_ref > 0])[:20]
    xnn = xnn_true + rng.normal(0, 2e-5, size=(n, 4))
    return q2_obs, xnn, extinction_group_sweep(q2_obs, xnn, lattice_system, bravais_lattice,
                                               hkl_ref=hkl_ref)


def test_the_group_index_round_trips_through_the_key_list_the_sweep_enumerates():
    """An index is meaningless without the list it indexes; they must be the SAME list.

    And not merely the same keys. `get_absence_counts` holds them alphabetically while
    `get_spacegroup_hkl_ref` yields them in insertion order -- same set, different order, on mP,
    oP, tP and oC at least. Recording the wrong one in `_meta.json` mislabels every stored index
    without changing a row count or raising anything, which is precisely how `merit_at_prune`
    mislabelled four of seven entries (C2-F-067).
    """
    from mlindex.model_training.FomBenchmark import spacegroup_reference_sets
    _, _, (keys, winners, _, _, _, _) = _sweep()
    assert keys == list(spacegroup_reference_sets('monoclinic', 'mP'))
    assert set(keys) == set(get_absence_counts('mP'))
    assert keys != list(get_absence_counts('mP')), 'the two orders are expected to differ here'
    for criterion in EXTINCTION_CRITERIA:
        assert winners[criterion].min() >= 0
        assert winners[criterion].max() < len(keys)


def test_the_driver_records_the_key_order_the_sweep_uses():
    """The meta writer and the sweep must agree, or every persisted index is off."""
    from mlindex.model_training.FomBenchmark import spacegroup_reference_sets
    from mlindex.utilities.ExtinctionCounts import LATTICE_SYSTEM
    for lattice in ('mP', 'oP', 'tP', 'oC'):
        system = LATTICE_SYSTEM[lattice]
        hkl_ref = np.load(os.path.join(BASE, 'mlindex', 'models', f'{system}_1', 'data',
                                       f'hkl_ref_{lattice}.npy'))
        recorded = list(spacegroup_reference_sets(system, lattice))
        assert recorded == list(get_spacegroup_hkl_ref(hkl_ref, bravais_lattice=lattice))
        assert recorded != list(get_absence_counts(lattice))


def test_the_sweep_returns_one_column_per_group_for_every_reported_array():
    q2_obs, xnn, (keys, winners, M20, scores, n_cal, n_absent) = _sweep()
    shape = (xnn.shape[0], len(keys))
    assert M20.shape == shape and n_cal.shape == shape and n_absent.shape == shape
    for criterion in EXTINCTION_CRITERIA:
        assert scores[criterion].shape == shape
        assert winners[criterion].shape == (xnn.shape[0],)


def test_the_absence_count_is_zero_at_the_generic_group_and_positive_elsewhere():
    """The count is a subtraction against the generic list, so the generic group deletes nothing.

    Also the negative control for the fixed counting window: if the window moved with the group,
    the generic group's own count could drift off zero.
    """
    _, _, (keys, _, _, _, _, n_absent) = _sweep()
    generic = keys.index(list(get_absence_counts('mP'))[0])
    assert np.all(n_absent[:, generic] == 0)
    assert n_absent.max() > 0


def test_M20_scores_are_shared_rather_than_recomputed_per_criterion():
    """`scores['M20']` is the same M20 the other criteria are tie-broken against."""
    _, _, (_, _, M20, scores, _, _) = _sweep()
    assert np.array_equal(scores['M20'], M20)


def test_triclinic_is_skipped_by_the_driver():
    """One group, one possible choice: every arm is identical and the rows carry no information."""
    assert 'aP' in SKIP_LATTICES


@pytest.mark.skipif(not os.path.exists(POOL), reason='the retained pool is not on this machine')
def test_the_driver_skips_triclinic_and_resumes_by_default():
    tasks = candidate_tasks(POOL, os.path.join(POOL, 'does_not_exist'), EXTINCTION_CRITERIA,
                            1000, None, None, overwrite=False)
    lattices = {os.path.basename(task[0]).rsplit('_', 1)[1].split('.')[0] for task in tasks}
    assert 'aP' not in lattices
    assert lattices, 'the pool should have produced work for the other lattices'


@pytest.mark.skipif(not os.path.exists(POOL), reason='the retained pool is not on this machine')
def test_every_column_the_sweep_projects_exists_in_the_pool():
    """`schema_arrow`, not `schema` -- parquet flattens `xnn` to `xnn.list.element`.

    A membership test against the flattened names drops the column silently: the read succeeds and
    the sweep raises later, somewhere else.
    """
    import pyarrow.parquet as pq
    from pathlib import Path
    path = sorted(Path(POOL).glob('candidates*_mP.parquet'))[0]
    names = set(pq.ParquetFile(path).schema_arrow.names)
    assert set(CANDIDATE_COLUMNS) <= names
    assert set(JOIN_KEYS) <= names
    assert 'xnn' not in set(pq.ParquetFile(path).schema.names)


def test_every_lattice_has_a_displacement_radius():
    """The stability arm scales its displacement by each lattice's own `neighbor_radius`.

    Scraped from the optimizer factories, so a refactor that moves the literal would otherwise
    drop a lattice from the table without anything failing.
    """
    from mlindex.scripts.run_fom_extinction_eval import _neighbor_radii
    radii = _neighbor_radii()
    assert set(radii) == set(LATTICE_SYSTEM)
    assert all(value > 0 for value in radii.values())
