"""Mighell-Santoro lattice degeneracy, computed from the true cell alone.

Campaign 1 shipped `is_degenerate` **null**, so degenerates were excluded from every analysis at
a *measured* zero rather than a known one -- the include and exclude oracles were identical
because the column was empty, not because the effect was (campaign 1's R7, C2-Q-002). This module
populates it.

The source is Santoro & Mighell (1970), *Determination of Reduced Cells*, Acta Cryst **A26**,
124-127, in `fom_papers/`, read from the rendered pages rather than the text layer (PROTOCOL §6).

**What the class is.** The paper's footnote on p. 126 states it exactly:

    "the lattice symmetry determined by means of reduced cells is purely metric and it may be
    the same as or *higher* than the true symmetry of the lattice of the crystal structure."

**Where it comes from.** Niggli's reduced cell is defined by *main* conditions plus *special*
conditions (the paper's (3) for a Type I cell and (5) for a Type II cell). Each special condition
is a left-column equality between scalars that triggers a right-column relation. Of the equalities
the paper says (p. 125):

    "The equalities between the scalars shown in the left columns of conditions (3) and (5) may
    occur accidentally or systematically depending on the particular geometrical properties of a
    lattice."

**And the precise degenerate case is the boundary, not the trigger.** Santoro works it through on
p. 126 for condition (3c). With the left-column equality S23 = S22/2 in force, cell (7) is reduced
when S12 < 2*S13 and cell (8) is reduced when S12 > 2*S13, so the reduction is unambiguous either
way. It is at S12 = 2*S13 -- the right-column inequality holding *with equality* -- that

    "the two cells are indistinguishable"

and there matrix (9) "represents an end-centered monoclinic lattice, while matrices (7) and (8)
represent a triclinic lattice". A higher-symmetry lattice and a lower-symmetry one carry the same
scalars, so they index the same peak positions. That is the phenomenon `is_degenerate` marks.

**Why this definition and not another.** Three properties the campaign needs:

* **It never mentions sigma.** It is a statement about the lattice metric, never about how well
  anything fits observed peaks, so PROTOCOL §3 rule 4 -- sigma is never assumed known -- does not
  bite. This is what blocked campaign 1's version, which matched lines "within sigma(q2)".
* **It is unambiguous.** Niggli's reduced cell is unique in all cases; that is the paper's thesis
  and its closing recommendation over Buerger's algorithm, which gave five ambiguous cases in
  fifty triclinic reductions. Campaign 1's other blocker was that "the true cell's calculated
  lines" was ambiguous because the true cell's Bravais lattice implies a different reference list
  from a candidate's. A reduced cell has no such freedom.
* **It is computable from truth alone**, at generation time, which is where the schema puts the
  column.

**Do not conflate this with the campaign's dominant failure mode.** This degeneracy is a risk of
indexing at symmetry *higher* than truth. Campaign 1's measured failure was symmetry *lowering*:
85.5 % of wrong winners are of lower symmetry than the truth. Different phenomena, and a result
about one says nothing about the other.

Note also C2-Q-007, which is **not** this module's business: the repository standardises monoclinic
and triclinic cells by Delaunay/Selling (`Reindexing.py`), and this same paper observes that the
Delaunay endpoint "is not necessarily based on the shortest three non-coplanar translations and is
not unique in all cases". That is an observation about a different routine, not a defect claim, and
nobody owns it yet.
"""

import numpy as np


# The holohedry of each Bravais lattice, used only to obtain the primitive setting before
# reduction. Niggli reduction is defined on the primitive cell of the lattice, so a conventional
# centred cell must be transformed first -- reducing an mC cell as though it were primitive
# describes a different lattice.
BRAVAIS_HOLOHEDRY = {
    'cP': 'P m -3 m', 'cI': 'I m -3 m', 'cF': 'F m -3 m',
    'tP': 'P 4/m m m', 'tI': 'I 4/m m m',
    'oP': 'P m m m', 'oC': 'C m m m', 'oI': 'I m m m', 'oF': 'F m m m',
    'hP': 'P 6/m m m', 'hR': 'R -3 m :H',
    'mP': 'P 2/m', 'mC': 'C 2/m',
    'aP': 'P -1',
    }

# Which conventional cell parameters a Bravais lattice may vary independently. Used by the
# accidental-versus-systematic test: perturbing exactly these and holding everything else at the
# value its lattice type forces is what distinguishes an equality the lattice *requires* from one
# that merely happens to hold for this crystal.
FREE_PARAMETERS = {
    'cP': ('a',), 'cI': ('a',), 'cF': ('a',),
    'tP': ('a', 'c'), 'tI': ('a', 'c'),
    'oP': ('a', 'b', 'c'), 'oC': ('a', 'b', 'c'),
    'oI': ('a', 'b', 'c'), 'oF': ('a', 'b', 'c'),
    'hP': ('a', 'c'), 'hR': ('a', 'alpha'),
    'mP': ('a', 'b', 'c', 'beta'), 'mC': ('a', 'b', 'c', 'beta'),
    'aP': ('a', 'b', 'c', 'alpha', 'beta', 'gamma'),
    }

# The correctness labeller's own tolerance (`validate_candidate_known_bl(..., rtol=1e-2)`). The
# handoff specifies it and it is the right scale rather than merely a convenient one: if two cells
# agree more closely than the labeller can resolve, the labeller cannot tell them apart, and being
# indistinguishable *to this benchmark's own oracle* is the operational content of "degenerate".
DEFAULT_TOLERANCE = 1e-2

# Perturbations used to decide whether an equality is systematic. Small enough to stay in the same
# lattice type, large enough to break an accidental coincidence held only to DEFAULT_TOLERANCE.
SYSTEMATIC_PROBE_SCALE = 0.05
SYSTEMATIC_PROBE_COUNT = 4


def scalars_from_cell(unit_cell):
    """Santoro's S matrix (their equation 1) from a cell (a, b, c, alpha, beta, gamma), radians.

    S11 = a.a, S22 = b.b, S33 = c.c, S23 = b.c, S13 = a.c, S12 = a.b.
    """
    a, b, c, alpha, beta, gamma = (float(value) for value in unit_cell)
    return {
        'S11': a * a,
        'S22': b * b,
        'S33': c * c,
        'S23': b * c * np.cos(alpha),
        'S13': a * c * np.cos(beta),
        'S12': a * b * np.cos(gamma),
        }


def _normalised(scalars):
    """Scalars divided by S33, so every comparison is dimensionless.

    S33 is the largest diagonal by the main conditions (S11 <= S22 <= S33) and is strictly
    positive, so this is well defined. Normalising once here means a single absolute tolerance
    can be applied to every comparison, rather than a relative test that blows up wherever a
    scalar passes through zero -- and Type II cells have S12 <= 0, so they do.
    """
    scale = scalars['S33']
    return {name: value / scale for name, value in scalars.items()}


def cell_type(scalars, tolerance=DEFAULT_TOLERANCE):
    """Type I (all off-diagonal scalars positive) or Type II, per the paper's `General` section.

    "If one or more of the S_ij is zero, the cell will be considered to be of Type II."
    """
    normalised = _normalised(scalars)
    off_diagonal = [normalised['S23'], normalised['S13'], normalised['S12']]
    if all(value > tolerance for value in off_diagonal):
        return 1
    return 2


def _special_conditions(scalars, cell_type_number):
    """(name, left-hand equality, right-hand slack) for each special condition of this cell type.

    The *slack* is the right-column inequality written as `larger - smaller`, so a slack of zero
    is the boundary at which two cells become indistinguishable. Conditions whose right column is
    an equation rather than an inequality carry a slack of `None`: they have no boundary to sit
    on, so they cannot produce this degeneracy and are reported as triggers only.
    """
    s = _normalised(scalars)
    S11, S22, S33 = s['S11'], s['S22'], s['S33']
    S23, S13, S12 = s['S23'], s['S13'], s['S12']

    if cell_type_number == 1:
        # Conditions (3), positive reduced form, Type I cell, all angles < 90 degrees.
        return (
            ('3a', S11 - S22, S13 - S23),          # S11 = S22  ->  S23 <= S13
            ('3b', S22 - S33, S12 - S13),          # S22 = S33  ->  S13 <= S12
            ('3c', S23 - S22 / 2, 2 * S13 - S12),  # S23 = S22/2 -> S12 <= 2 S13
            ('3d', S13 - S11 / 2, 2 * S23 - S12),  # S13 = S11/2 -> S12 <= 2 S23
            ('3e', S12 - S11 / 2, 2 * S23 - S13),  # S12 = S11/2 -> S13 <= 2 S23
            )
    # Conditions (5), negative reduced form, Type II cell, all angles >= 90 degrees.
    absolute_sum = abs(S23) + abs(S13) + abs(S12)
    return (
        ('5a', S11 - S22, abs(S13) - abs(S23)),           # S11 = S22  -> |S23| <= |S13|
        ('5b', S22 - S33, abs(S12) - abs(S13)),           # S22 = S33  -> |S13| <= |S12|
        ('5c', abs(S23) - S22 / 2, None),                 # |S23| = S22/2 -> S12 = 0
        ('5d', abs(S13) - S11 / 2, None),                 # |S13| = S11/2 -> S12 = 0
        ('5e', abs(S12) - S11 / 2, None),                 # |S12| = S11/2 -> S13 = 0
        ('5f', absolute_sum - (S11 + S22) / 2,            # the sum condition ->
         2 * abs(S13) + abs(S12) - S11),                  #   S11 <= 2|S13| + |S12|
        )


def degenerate_conditions(unit_cell, tolerance=DEFAULT_TOLERANCE):
    """Which special conditions this *already reduced* cell sits on the boundary of.

    Returns the condition names for which the left-column equality holds AND the right-column
    inequality is tight -- Santoro's "indistinguishable" case. Takes a Niggli reduced primitive
    cell; `reduced_cell` produces one.
    """
    scalars = scalars_from_cell(unit_cell)
    fired = []
    for name, left, slack in _special_conditions(scalars, cell_type(scalars, tolerance)):
        if abs(left) > tolerance:
            continue
        if slack is None:
            continue
        if abs(slack) <= tolerance:
            fired.append(name)
    return tuple(fired)


def triggered_conditions(unit_cell, tolerance=DEFAULT_TOLERANCE):
    """Which special conditions are *triggered* -- the left-column equality alone.

    Weaker than `degenerate_conditions` and reported beside it because it is the literal referent
    of the paper's "equalities ... may occur accidentally or systematically" sentence. Stored so a
    later session can re-read the column under that broader definition without regenerating the
    pool (PROTOCOL §3 rule 8).
    """
    scalars = scalars_from_cell(unit_cell)
    return tuple(name for name, left, _ in _special_conditions(scalars, cell_type(scalars,
                                                                                 tolerance))
                 if abs(left) <= tolerance)


def reduced_cell(unit_cell, bravais_lattice):
    """The Niggli reduced cell of the lattice, as (a, b, c, alpha, beta, gamma) in radians.

    The conventional cell is transformed to its primitive setting first. Reducing a centred cell
    as though it were primitive describes a different lattice, so this step is not optional.

    `cctbx` is imported here rather than at module scope: it is a dataset-generation dependency,
    not a runtime one, and the end-user path (`pip install mlindex` -> download models -> run)
    must not acquire it.
    """
    from cctbx import crystal

    a, b, c, alpha, beta, gamma = (float(value) for value in unit_cell)
    symmetry = crystal.symmetry(
        unit_cell=(a, b, c, np.degrees(alpha), np.degrees(beta), np.degrees(gamma)),
        space_group_symbol=BRAVAIS_HOLOHEDRY[bravais_lattice],
        )
    primitive = symmetry.change_basis(symmetry.change_of_basis_op_to_primitive_setting())
    niggli = primitive.unit_cell().niggli_cell().parameters()
    return (niggli[0], niggli[1], niggli[2],
            np.radians(niggli[3]), np.radians(niggli[4]), np.radians(niggli[5]))


def _perturbed_cells(unit_cell, bravais_lattice, rng):
    """Cells of the same Bravais lattice with the free parameters jogged.

    Only the parameters `FREE_PARAMETERS` allows are moved; everything else stays at whatever
    value the lattice type forces. So an equality that the lattice *requires* survives every
    perturbation, and one that merely happens to hold does not.
    """
    names = ('a', 'b', 'c', 'alpha', 'beta', 'gamma')
    free = FREE_PARAMETERS[bravais_lattice]
    for _ in range(SYSTEMATIC_PROBE_COUNT):
        values = list(float(value) for value in unit_cell)
        for index, name in enumerate(names):
            if name in free:
                values[index] *= 1.0 + rng.uniform(-SYSTEMATIC_PROBE_SCALE,
                                                   SYSTEMATIC_PROBE_SCALE)
        # A lattice with a = b by construction must keep it after the jog, or the probe would
        # report every such equality accidental. Re-impose the ties the lattice type implies.
        if bravais_lattice in ('cP', 'cI', 'cF'):
            values[1] = values[2] = values[0]
        elif bravais_lattice in ('tP', 'tI', 'hP'):
            values[1] = values[0]
        elif bravais_lattice == 'hR':
            values[1] = values[2] = values[0]
            values[4] = values[5] = values[3]
        yield tuple(values)


def is_degenerate(unit_cell, bravais_lattice, tolerance=DEFAULT_TOLERANCE, seed=0):
    """Is this entry a Mighell-Santoro degenerate?

    True when the Niggli reduced cell of the true lattice sits on the boundary of one of
    Santoro's special conditions -- so a lattice of higher metric symmetry carries the same
    scalars and indexes the same peak positions -- **accidentally**, meaning the coincidence is
    not forced by the lattice type.

    A systematically degenerate lattice is not a degenerate at all; it is just symmetry, and the
    indexer is entitled to find it. Systematicity is decided by perturbation: jog the free
    parameters of the true cell, re-reduce, and see whether the boundary condition survives. An
    equality the lattice requires survives every jog; a coincidence does not. This is preferred
    over hand-deriving the constraint algebra for fourteen lattices, which would be fourteen
    opportunities to get it silently wrong.

    Returns (is_degenerate, accidental_conditions, systematic_conditions). Exactly one of the
    two tuples is populated: a lattice whose degeneracy survives every perturbation is
    systematically degenerate and is not a degenerate at all.
    """
    reduced = reduced_cell(unit_cell, bravais_lattice)
    fired = degenerate_conditions(reduced, tolerance)
    if not fired:
        return False, (), ()

    # The probe asks whether the perturbed lattice is degenerate AT ALL, not whether the same
    # named condition fires again. Niggli reduction may return a relabelled basis -- axes
    # permuted, the cell type flipped -- in which case the identical geometric degeneracy
    # reappears under a different condition name. Comparing names therefore reports a
    # systematically degenerate lattice as accidental, which is how the first version of this
    # function put 36.7 % of mC entries in the wrong class. Whether a boundary is reachable at
    # all is the property that is invariant under relabelling, and it is also the physically
    # meaningful one: does the lattice *type* force a degeneracy, or does this crystal hit one?
    rng = np.random.default_rng(seed)
    escaped = False
    probed = 0
    for probe in _perturbed_cells(unit_cell, bravais_lattice, rng):
        try:
            probe_reduced = reduced_cell(probe, bravais_lattice)
        except Exception:
            # A jog that produces a cell cctbx will not reduce says nothing about systematicity.
            continue
        probed += 1
        if not degenerate_conditions(probe_reduced, tolerance):
            escaped = True
            break

    if probed == 0:
        # No usable probe, so systematicity is undecided. Report the degeneracy and attribute it
        # to neither class rather than guessing; the conditions are stored either way.
        return True, fired, ()
    if escaped:
        return True, fired, ()
    return False, (), fired
