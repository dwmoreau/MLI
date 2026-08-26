"""How many systematic absences an extinction group imposes, as an ordered scalar.

Campaign 1's learned combiner consumed the extinction group as a 158-level categorical, backed by a
median of 28 source crystals per level and by fewer than 30 for 86 of them (C2-F-003). DWMM's
replacement is the *count* of reference lines the chosen group removes beyond the Bravais lattice's
own centring absences: one ordered number instead of 158 unordered labels.

The count is a subtraction. `get_spacegroup_hkl_ref` builds each group's reference list by deleting
systematically absent reflections from the lattice's generic list, so

    n_absent_extra = len(hkl_ref) - len(hkl_ref_sg[group])

and the lattice's own absences are already baked into `hkl_ref` -- which is exactly the "additional
to the Bravais lattice's absences" the hypothesis is about. Every lattice has exactly one group at
count zero, the generic group, whose list *is* `hkl_ref`; `GENERIC_GROUP` names it per lattice.

Two counts, not one. The full count is dominated by high-angle lines nobody observed, so the
quantity with a mechanism attached is the count restricted to the merit's own counting window --
`absent_in_range`. Deleting an absent line inside that window lowers `get_M20`'s N, which raises its
expected discrepancy and so raises M20; deleting one outside it changes nothing at all.

Building the table needs cctbx, which is not a runtime dependency, so it is built once by
`build_absence_counts` and read back from `LOOKUP_PATH` by `get_absence_counts`. Inference never
calls cctbx.
"""
import json
import os

import numpy as np

from mlindex.utilities.SpaceGroups import get_spacegroup_hkl_ref

LOOKUP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'extinction_absence_counts.json')

# The lattice system each Bravais lattice's reference list is stored under, so the table can be
# rebuilt from `mlindex/models/{system}_1/data/hkl_ref_{lattice}.npy` without a Wrapper.
LATTICE_SYSTEM = {
    'cF': 'cubic', 'cI': 'cubic', 'cP': 'cubic',
    'hP': 'hexagonal', 'hR': 'rhombohedral',
    'mC': 'monoclinic', 'mP': 'monoclinic',
    'oC': 'orthorhombic', 'oF': 'orthorhombic', 'oI': 'orthorhombic', 'oP': 'orthorhombic',
    'tI': 'tetragonal', 'tP': 'tetragonal',
    'aP': 'triclinic',
    }

_LOOKUP_CACHE = None


def build_absence_counts(hkl_ref, bravais_lattice):
    """{group key: lines this group removes from `hkl_ref`}. Needs cctbx.

    Keys are `get_spacegroup_hkl_ref`'s own -- "<extinction group> e.g. <spacegroup>" -- because
    those are what the dump's `spacegroup` column carries, so the table joins to a candidate frame
    without a second naming convention to keep in step.
    """
    hkl_ref_sg = get_spacegroup_hkl_ref(hkl_ref, bravais_lattice=bravais_lattice)
    return {key: int(hkl_ref.shape[0] - lines.shape[0]) for key, lines in hkl_ref_sg.items()}


def build_group_masks(hkl_ref, bravais_lattice):
    """{group key: boolean mask over `hkl_ref`, True where the group KEEPS the line}. Needs cctbx.

    The narrowed lists are subsets of `hkl_ref` in its own order, so membership recovers the mask
    without re-deriving the absence rules. Comparing rows as void-typed bytes is what makes that a
    single vectorised pass rather than a Python loop over up to 1 000 Miller indices.
    """
    hkl_ref_sg = get_spacegroup_hkl_ref(hkl_ref, bravais_lattice=bravais_lattice)
    reference = np.ascontiguousarray(hkl_ref).view(
        [('', hkl_ref.dtype)] * hkl_ref.shape[1]).ravel()
    masks = {}
    for key, lines in hkl_ref_sg.items():
        kept = np.ascontiguousarray(lines.astype(hkl_ref.dtype)).view(
            [('', hkl_ref.dtype)] * hkl_ref.shape[1]).ravel()
        masks[key] = np.isin(reference, kept)
    return masks


def get_absence_counts(bravais_lattice=None):
    """The committed table: {lattice: {group key: n_absent_extra}}, or one lattice's entry.

    Read from `LOOKUP_PATH` rather than derived, so this runs in the inference-only environment.
    """
    global _LOOKUP_CACHE
    if _LOOKUP_CACHE is None:
        with open(LOOKUP_PATH, encoding='utf-8') as _f:
            _LOOKUP_CACHE = json.load(_f)['counts']
    if bravais_lattice is None:
        return _LOOKUP_CACHE
    return _LOOKUP_CACHE[bravais_lattice]


def get_n_groups_searched(bravais_lattice):
    """How many extinction groups `assign_extinction_group`'s argmax ran over for this lattice.

    The look-elsewhere count `delta_merit_extinction` has to be judged against: a gain selected as
    the best of 68 alternatives (oP) is not the same quantity as the best of one (aP).
    """
    return len(get_absence_counts(bravais_lattice))


GENERIC_GROUP = {}


def get_generic_group(bravais_lattice):
    """The one group per lattice that removes nothing -- whose reference list is `hkl_ref` itself.

    This is why the merit against the full list is the merit at the generic group, and so why
    `delta_merit_extinction` is a plain difference of two already-computed columns.
    """
    if bravais_lattice not in GENERIC_GROUP:
        counts = get_absence_counts(bravais_lattice)
        zero = [key for key, value in counts.items() if value == 0]
        if len(zero) != 1:
            raise ValueError(
                f'{bravais_lattice}: expected exactly one group removing no lines, found {zero}')
        GENERIC_GROUP[bravais_lattice] = zero[0]
    return GENERIC_GROUP[bravais_lattice]


def absent_in_range(q2_ref_calc, keep_mask, cutoff):
    """Absent lines that fall inside the counting window, per candidate.

    `q2_ref_calc` is (n_candidates, n_ref) for the FULL reference list, `keep_mask` is (n_ref,)
    True where the chosen group keeps the line, and `cutoff` is (n_candidates,). Returns the count
    of dropped lines strictly below the cutoff, alongside the full list's in-range count so the
    scale-free fraction can be formed without a second pass.

    The comparison is strict `<` to match `get_M20`, whose N counts `q2_ref_calc < q2_calc[:, -1]`.
    A line exactly at the cutoff is excluded there and must be excluded here, or the fraction and
    the merit disagree about the same window.
    """
    in_range = q2_ref_calc < cutoff[:, np.newaxis]
    n_in_range = in_range.sum(axis=1)
    n_dropped = (in_range & ~keep_mask[np.newaxis, :]).sum(axis=1)
    return n_dropped.astype(np.int64), n_in_range.astype(np.int64)
