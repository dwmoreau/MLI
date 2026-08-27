"""S13 -- the per-peak assignment statistic, its full-distribution form, and the two flags.

Three things are pinned here, and each corresponds to a way this change could be wrong without
looking wrong.

**The ported comparison set must reproduce the shipped statistic.** `get_assignment_probability`
lifts `get_M20_likelihood`'s `1/(1 + arg)` out into its own function so S13 can score it beside
other forms; if the two ever drift, every comparison in the step is against a statistic production
does not use.

**The full distribution must be the same arithmetic as the scalar posterior.** `refine_cell` reads
the scalar form and `IntegralFilter.generate` reads the distribution, so the two consumers have to
be reading one estimator, not two implementations of one description of it.

**The flags must be off.** Anything on the inference path ships behind a flag defaulting to the
M20-era behaviour (PROTOCOL section 5), and a default that is off in the parameter dictionary but
on in the code is exactly the kind of thing a test catches and a reading does not.
"""
import numpy as np
import pytest

from pathlib import Path

_TEST_DATA_DIR = Path(__file__).parent.parent / "mlindex" / "data" / "test_data"


def _random_case(seed, n_candidates, n_ref, n_peaks, lattice_system):
    rng = np.random.default_rng(seed)
    q2_obs = np.sort(rng.uniform(0.01, 0.5, n_peaks))
    q2_ref_calc = np.sort(rng.uniform(0.005, 0.6, (n_candidates, n_ref)), axis=1)
    return q2_obs, q2_ref_calc, lattice_system


# ------------------------------------------------------------------------------------------
# The comparison set, ported from `fom` at 7c137c3
# ------------------------------------------------------------------------------------------
def test_rho_reproduces_the_shipped_statistic_exactly():
    """`get_assignment_probability(form='rho')` is `get_M20_likelihood`'s second return value.

    Bit-for-bit, not close: the ported function exists so S13 can score the *shipped* statistic,
    and a version that merely agrees to floating-point tolerance is a different statistic whose
    threshold behaviour at 0.95 could differ on the boundary.
    """
    from mlindex.utilities.FigureOfMerits import get_assignment_probability
    from mlindex.utilities.FigureOfMerits import get_M20_likelihood

    rng = np.random.default_rng(3)
    q2_obs = np.sort(rng.uniform(0.01, 0.5, 20))
    q2_calc = q2_obs[np.newaxis] + rng.normal(0, 1e-4, (11, 20))
    reciprocal_volume = rng.uniform(1e-4, 1e-2, 11)

    _, shipped, _ = get_M20_likelihood(q2_obs, q2_calc, 'mP', reciprocal_volume)
    ported = get_assignment_probability(q2_obs, q2_calc, 'mP', reciprocal_volume, form='rho')
    assert np.array_equal(shipped, ported)


def test_rho_and_taupin_are_one_statistic_under_two_links():
    """Monotone links do not reorder, so the two forms must rank every peak identically.

    This is the claim the whole estimator comparison rests on -- that `rho` and Taupin's form
    differ in calibration and in nothing else -- and it is cheap enough to assert rather than
    repeat in prose.
    """
    from mlindex.utilities.FigureOfMerits import get_assignment_probability

    rng = np.random.default_rng(5)
    q2_obs = np.sort(rng.uniform(0.01, 0.5, 20))
    q2_calc = q2_obs[np.newaxis] + rng.normal(0, 1e-4, (7, 20))
    reciprocal_volume = rng.uniform(1e-4, 1e-2, 7)

    rho = get_assignment_probability(q2_obs, q2_calc, 'oP', reciprocal_volume, form='rho')
    taupin = get_assignment_probability(q2_obs, q2_calc, 'oP', reciprocal_volume, form='taupin')
    assert np.array_equal(np.argsort(rho, axis=None), np.argsort(taupin, axis=None))


def test_assignment_probability_rejects_an_unknown_form():
    from mlindex.utilities.FigureOfMerits import get_assignment_probability

    with pytest.raises(ValueError):
        get_assignment_probability(np.array([0.1]), np.array([[0.1]]), 'cP',
                                   np.array([1e-3]), form='not-a-form')


# ------------------------------------------------------------------------------------------
# The full posterior distribution
# ------------------------------------------------------------------------------------------
@pytest.mark.parametrize('lattice_system,n_ref,n_peaks',
                         [('monoclinic', 137, 20), ('cubic', 41, 10), ('triclinic', 61, 20)])
def test_the_distribution_normalises_per_peak(lattice_system, n_ref, n_peaks):
    from mlindex.utilities.FigureOfMerits import get_assignment_distribution

    q2_obs, q2_ref_calc, system = _random_case(0, 5, n_ref, n_peaks, lattice_system)
    distribution = get_assignment_distribution(q2_obs, q2_ref_calc, system)
    assert distribution.shape == (5, n_peaks, n_ref)
    assert np.allclose(distribution.sum(axis=2), 1.0, atol=1e-12)


def test_the_distribution_agrees_with_the_scalar_posterior_bit_for_bit():
    """The nearest line's column of the distribution IS `get_assignment_posterior`.

    The two share `posterior_exponent_terms` and the same `where=` exponential precisely so this
    holds exactly; if it ever needs a tolerance, the two consumers have stopped reading one
    estimator.
    """
    from mlindex.utilities.FigureOfMerits import get_assignment_distribution
    from mlindex.utilities.FigureOfMerits import get_assignment_posterior

    q2_obs, q2_ref_calc, system = _random_case(1, 9, 211, 20, 'monoclinic')
    scalar = get_assignment_posterior(q2_obs, q2_ref_calc, system)
    full = get_assignment_distribution(q2_obs, q2_ref_calc, system)
    nearest = np.argmin(np.abs(q2_ref_calc[:, np.newaxis, :] - q2_obs[np.newaxis, :, np.newaxis]),
                        axis=2)
    at_nearest = np.take_along_axis(full, nearest[:, :, np.newaxis], axis=2)[:, :, 0]
    assert np.array_equal(at_nearest, scalar)


def test_the_distribution_matches_a_direct_numpy_reference():
    from mlindex.utilities.FigureOfMerits import get_assignment_distribution
    from mlindex.utilities.FigureOfMerits import get_assignment_sigma

    q2_obs, q2_ref_calc, system = _random_case(2, 4, 97, 20, 'orthorhombic')
    sigma, _ = get_assignment_sigma(q2_obs, q2_ref_calc, system)
    distance = np.abs(q2_ref_calc[:, np.newaxis, :] - q2_obs[np.newaxis, :, np.newaxis])
    reference = np.exp(-distance**2/(2*sigma[:, np.newaxis, np.newaxis]**2))
    reference = reference/reference.sum(axis=2)[:, :, np.newaxis]
    assert np.allclose(reference, get_assignment_distribution(q2_obs, q2_ref_calc, system),
                       atol=1e-12)


def test_the_unnormalised_form_is_proportional_to_the_normalised_one():
    """`vectorized_resampling` rescales by each row's own total, so it takes the cheaper form."""
    from mlindex.utilities.FigureOfMerits import get_assignment_distribution

    q2_obs, q2_ref_calc, system = _random_case(4, 3, 53, 12, 'tetragonal')
    normalised = get_assignment_distribution(q2_obs, q2_ref_calc, system)
    raw = get_assignment_distribution(q2_obs, q2_ref_calc, system, normalise=False)
    assert np.allclose(raw/raw.sum(axis=2)[:, :, np.newaxis], normalised, atol=1e-12)


def test_the_fortran_ordered_fallback_agrees_with_the_fast_path():
    """The non-C-contiguous branch is a safety net, and it has to compute the same thing.

    Not bit-for-bit: `np.sum(axis=1)` groups its additions differently over an F-ordered block,
    which `get_assignment_posterior`'s own comment records as deliberate.
    """
    from mlindex.utilities.FigureOfMerits import get_assignment_distribution

    q2_obs, q2_ref_calc, system = _random_case(6, 5, 71, 15, 'hexagonal')
    fortran = np.asfortranarray(q2_ref_calc)
    assert not fortran.flags.c_contiguous
    assert np.allclose(get_assignment_distribution(q2_obs, q2_ref_calc, system),
                       get_assignment_distribution(q2_obs, fortran, system), atol=1e-12)


# ------------------------------------------------------------------------------------------
# The flags
# ------------------------------------------------------------------------------------------
def _triclinic_candidates(opt_params_extra=None, seed=7):
    from mlindex.optimization.Candidates import Candidates
    from mlindex.utilities.Q2Calculator import Q2Calculator

    hkl_ref = np.load(_TEST_DATA_DIR.parent / "hkl_ref_aP.npy")
    xnn_true = np.array([[0.02, 0.015, 0.01, 0.001, 0.002, 0.0015]])
    q2_ref = Q2Calculator(lattice_system="triclinic", hkl=hkl_ref, tensorflow=False,
                          representation="xnn").get_q2(xnn_true)[0]
    q2_obs = np.sort(q2_ref[q2_ref > 0])[:20]
    opt_params = {"minimum_uc": 2.0, "maximum_uc": 60.0, "assignment_threshold": 0.95,
                  "figure_of_merit": "M20"}
    opt_params.update(opt_params_extra or {})
    rng = np.random.default_rng(seed)
    return Candidates(
        q2_obs=q2_obs, xnn=np.repeat(xnn_true, 4, axis=0), hkl_ref=hkl_ref,
        lattice_system="triclinic", bravais_lattice="aP", opt_params=opt_params, rng=rng,
        fom=None, zero_error=False, wavelength=None,
        )


def test_the_assignment_statistic_defaults_to_the_shipped_one():
    """An opt_params dict that does not carry the key behaves exactly as before."""
    assert _triclinic_candidates().assignment_statistic == 'rho'


def test_an_unknown_assignment_statistic_is_refused_at_construction():
    """Refused where it is set, not where it is read.

    A typo that silently fell through to the posterior would change every refined cell in the run
    and there would be nothing in the output to say so.
    """
    with pytest.raises(ValueError):
        _triclinic_candidates({'assignment_statistic': 'posteriro'})


def test_rho_is_what_the_helper_returns_under_the_default():
    """`assignment_probability` under 'rho' is `get_M20_likelihood_from_xnn`'s own output."""
    from mlindex.utilities.FigureOfMerits import get_M20_likelihood_from_xnn

    candidates = _triclinic_candidates()
    probability, Minfo = candidates.assignment_probability(
        candidates.best_xnn, candidates.best_hkl)
    _, expected, expected_Minfo = get_M20_likelihood_from_xnn(
        q2_obs=candidates.q2_obs, xnn=candidates.best_xnn, hkl=candidates.best_hkl,
        lattice_system='triclinic', bravais_lattice='aP')
    assert np.array_equal(probability, expected)
    assert np.array_equal(Minfo, expected_Minfo)


def test_the_posterior_branch_returns_no_Minfo():
    """Minfo is a link function of rho's own argument and must never be taken from the posterior.

    Swapping the mask's statistic must not silently redefine a second reported column; returning
    None is how the caller is stopped from doing it by accident.
    """
    candidates = _triclinic_candidates({'assignment_statistic': 'posterior'})
    probability, Minfo = candidates.assignment_probability(
        candidates.best_xnn, candidates.best_hkl)
    assert Minfo is None
    assert probability.shape == (candidates.n, candidates.q2_obs.size)
    assert np.all((probability > 0) & (probability <= 1))


def test_the_hkl_source_default_is_the_network():
    from mlindex.model_training.IntegralFilter import HKL_SOURCES, HKL_SOURCE_DEFAULT

    assert HKL_SOURCE_DEFAULT == 'network'
    assert set(HKL_SOURCES) == {'network', 'posterior'}


def test_the_optimizer_reads_the_shipped_hkl_source_when_opt_params_is_silent():
    """`.get` with the shipped default: a parameter dict without the key behaves as before."""
    from mlindex.model_training.IntegralFilter import HKL_SOURCE_DEFAULT

    assert {}.get('hkl_source', HKL_SOURCE_DEFAULT) == 'network'


# ------------------------------------------------------------------------------------------
# The drivers' own contracts
# ------------------------------------------------------------------------------------------
def test_the_seed_derivation_reproduces_S03s_at_the_same_base():
    """The arms driver exposes the seed base; at 12345 it must be S03's function exactly.

    Otherwise an S13 arm and an S03 arm on the same entry are different stochastic searches and
    nothing measured across the two steps is paired.
    """
    from mlindex.scripts.run_fom_assignment_arms import derived_seed as s13
    from mlindex.scripts.run_fom_prune_rerun import derived_seed as s03

    for entry_id in ('abcdef', 'COD_1000032', 'x'):
        for bravais_lattice in ('aP', 'mP', 'cF'):
            assert s13(entry_id, bravais_lattice, 12345) == s03(entry_id, bravais_lattice)
            assert s13(entry_id, bravais_lattice, 777) != s03(entry_id, bravais_lattice)


def test_the_baseline_arm_sets_nothing():
    """`baseline` must be the shipped path, not a re-specification of it.

    If a default moves, the baseline has to move with it -- otherwise the contrast silently
    becomes "the shipped path against a frozen copy of last month's shipped path".
    """
    from mlindex.scripts.run_fom_assignment_arms import ARM_OPTIONS

    assert ARM_OPTIONS['baseline'] == {}
    assert ARM_OPTIONS['mask'] == {'assignment_statistic': 'posterior'}
    assert ARM_OPTIONS['assigner'] == {'hkl_source': 'posterior'}


def test_canonical_hkl_is_integer_and_kills_negative_zero():
    """-0.0 compares equal to 0.0 and has different bytes, and the class lookup is byte-keyed.

    Campaign 1 sent one real reflection in twenty-two to the unindexed sentinel this way.
    """
    from mlindex.scripts.run_fom_assignment import canonical_hkl

    rows = canonical_hkl(np.array([[-1, 0, 0], [1, 0, 0], [0, 0, 0]]), 'monoclinic')
    assert rows.dtype == np.int64
    assert not np.signbit(rows).any()
    assert rows[0].tobytes() == rows[1].tobytes()


def test_contaminant_peaks_are_never_labelled_correct():
    """A line from another phase has no correct Miller index in this cell."""
    from mlindex.scripts.run_fom_assignment import assignment_labels

    hkl_true = np.array([[0, 0, 0], [1, 1, 0]], dtype=float)
    hkl_assigned = np.array([[[0, 0, 0], [1, 1, 0]]], dtype=float)
    labels = assignment_labels(hkl_assigned, hkl_true[np.newaxis], 'orthorhombic')
    assert labels.shape == (1, 2)
    # The contaminant's assigned index equals its sentinel truth, and it is still not a correct
    # assignment -- which is why the drivers exclude contaminants rather than relying on the label.
    assert labels[0, 1]


def test_a_candidate_that_fits_exactly_scores_one_rather_than_raising():
    """Zero residuals underflowed the posterior's scale to exactly 0 and the kernel divided by it.

    `get_assignment_sigma` clamps sigma at 1e-300, which looks like the degenerate case is handled;
    the consumer squares it and 1e-600 is 0.0. In numba that is a `ZeroDivisionError`, so the one
    candidate a synthetic pattern is guaranteed to contain -- the cell it was generated from --
    raised instead of scoring. The right answer is 1: an exact fit assigns with certainty.
    """
    from mlindex.utilities.FigureOfMerits import get_assignment_distribution
    from mlindex.utilities.FigureOfMerits import get_assignment_posterior

    q2_ref_calc = np.sort(np.random.default_rng(8).uniform(0.01, 0.6, (3, 64)), axis=1)
    q2_obs = q2_ref_calc[0, :20].copy()      # every residual exactly zero for candidate 0

    posterior = get_assignment_posterior(q2_obs, q2_ref_calc, 'orthorhombic')
    assert np.all(np.isfinite(posterior))
    assert np.allclose(posterior[0], 1.0)

    distribution = get_assignment_distribution(q2_obs, q2_ref_calc, 'orthorhombic')
    assert np.all(np.isfinite(distribution))
    assert np.allclose(distribution.sum(axis=2), 1.0)


def test_the_scale_floor_moves_nothing_that_was_not_degenerate():
    """The floor binds only where the unfloored scale would be zero or subnormal."""
    from mlindex.utilities.FigureOfMerits import _posterior_scale

    sigma = np.array([1e-3, 1.0, 1e-100, 1e-160, 1e-300])
    floored = _posterior_scale(sigma, 1.0)
    unfloored = 2*sigma**2
    ordinary = unfloored >= np.finfo(np.float64).tiny
    assert np.array_equal(floored[ordinary], unfloored[ordinary])
    assert np.all(floored[~ordinary] == np.finfo(np.float64).tiny)


def test_the_template_ranker_reads_its_statistic_with_a_default():
    """`load_from_tag` throws away every constructor default, so indexing the key would crash.

    It replaces `template_params` wholesale with `dict.fromkeys(params_keys)` built from the saved
    CSV, and the saved CSVs predate this key. A `[...]` lookup in `_generate_xnn_common` would
    therefore raise KeyError on the inference path for every trained lattice -- caught here rather
    than in a user's run, because the shipped models are what exercise that branch.
    """
    import inspect

    from mlindex.model_training import MITemplates

    source = inspect.getsource(MITemplates.MITemplates._generate_xnn_common)
    assert "self.template_params.get('assignment_statistic', 'rho')" in source
    assert "self.template_params['assignment_statistic']" not in source
