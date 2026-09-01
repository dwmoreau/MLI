"""S11: mapping `extinction_group_true` onto the group keys the assignment rule chooses between.

The rule picks one of up to 68 extinction groups per candidate; scoring it needs the truth column
and the candidate's `spacegroup` key to be comparable. They are written in two notations, and the
obvious repair -- normalise the dashes and the whitespace -- is the wrong one. It leaves 69 of 530
pool entries unmatched and invites a set-valued fudge for symbols like `C c c (ab)` that are not
actually ambiguous.

What could be silently wrong here: a truth symbol scored as a MISS because the two tables in
`SpaceGroups.py` write the same group differently. That inflates every rule's error rate by the
same amount, so it does not show up as an anomaly in a comparison -- it just quietly moves the
level, on exactly the centred lattices where `M_sym` is already known to lose (C2-F-096). Hence
`admissible_group_keys` raises rather than returning empty, and hence these tests.
"""
import os
import sys

import pandas as pd
import pytest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from mlindex.model_training.FomBenchmark import (
    UnknownExtinctionGroupError, admissible_group_keys, extinction_group_key_map,
    unreachable_group_keys,
    )
from mlindex.utilities.ExtinctionCounts import get_absence_counts
from mlindex.utilities.SpaceGroups import map_spacegroup_to_extinction_group

BRAVAIS_LATTICES = ('cF', 'cI', 'cP', 'hP', 'hR', 'mC', 'mP', 'oC', 'oF', 'oI', 'oP', 'tI', 'tP',
                    'aP')

# The EXPO table carries no monoclinic c- or a-glide row, so these four mP keys cannot be named by
# any truth symbol. Frozen as a fixture: if the table ever gains those rows this test says so, and
# mP's accuracy bound (C2-R, S11) can be lifted.
UNREACHABLE = {
    'mP': ('P 1 21/a 1 e.g. P 1 21/a 1', 'P 1 21/c 1 e.g. P 1 21/c 1',
           'P 1 a 1 e.g. P 1 a 1', 'P 1 c 1 e.g. P 1 c 1'),
    }

POOL_ENTRIES = os.path.join(BASE, 'mlindex', 'data', 'fom_full_c2_pool', 'entries.parquet')


def test_every_group_key_resolves_or_is_known_unreachable():
    """No key falls through the mapping unnoticed -- it resolves, or it is in the frozen set."""
    for bravais_lattice in BRAVAIS_LATTICES:
        keys = set(get_absence_counts(bravais_lattice))
        resolved = set().union(*extinction_group_key_map(bravais_lattice).values())
        unreachable = set(unreachable_group_keys(bravais_lattice))
        assert resolved | unreachable == keys
        assert not resolved & unreachable


def test_the_unreachable_set_is_exactly_the_four_monoclinic_glide_keys():
    """Pinned, because it bounds mP's assignment accuracy and nothing else would report it."""
    for bravais_lattice in BRAVAIS_LATTICES:
        assert unreachable_group_keys(bravais_lattice) == UNREACHABLE.get(bravais_lattice, ())


def test_no_truth_symbol_admits_more_than_one_key():
    """The mapping is 1-to-1, so assignment accuracy needs no set-valued scoring.

    Measured while planning S11: the apparent ambiguity in `C c c (ab)` / `I - - (ab)` / `P - 21 -`
    is notation, not crystallography. If this ever fails, set-valued scoring becomes necessary and
    the accuracy tables have to grow a second denominator.
    """
    for bravais_lattice in BRAVAIS_LATTICES:
        for symbol, keys in extinction_group_key_map(bravais_lattice).items():
            assert len(keys) == 1, (bravais_lattice, symbol, sorted(keys))


def test_the_rhombohedral_obverse_symbol_resolves_through_its_representative():
    """`R 3` must map to the obverse symbol, which is what makes hR resolve at all.

    `map_spacegroup_to_extinction_group` scans a table with duplicate codes for R3 and returns the
    first match. hR's truth is written `R (obv) - -` with two dashes against the key's three, so no
    character-level rule reaches it; the `e.g.` route does, and only because of first-match-wins.
    """
    symbol, code = map_spacegroup_to_extinction_group('R3')
    assert symbol.strip() == 'R (obv) – –'
    assert code == 189
    assert admissible_group_keys('hR', 'R (obv) – –') == frozenset({'R - - - e.g. R 3'})


def test_a_symbol_the_vocabulary_cannot_express_raises():
    """Never scored as a miss -- that would charge the rule for a defect in the mapping."""
    with pytest.raises(UnknownExtinctionGroupError):
        admissible_group_keys('oP', 'not a real extinction group')


@pytest.mark.skipif(not os.path.exists(POOL_ENTRIES), reason='the retained pool is not on this machine')
def test_every_truth_value_in_the_retained_pool_resolves():
    """530 of 530, frozen. This is the claim the accuracy tables rest on."""
    entries = pd.read_parquet(
        POOL_ENTRIES, columns=['entry_id', 'bravais_lattice_true', 'extinction_group_true']
        ).drop_duplicates('entry_id')
    for bravais_lattice, group in entries.groupby('bravais_lattice_true'):
        for symbol in group['extinction_group_true'].unique():
            assert len(admissible_group_keys(bravais_lattice, symbol)) == 1
