"""The condition set is the benchmark's identity: which crystals under which noise.

These tests pin the properties other code relies on rather than the values themselves, except
where a value being wrong would silently invalidate a comparison.
"""

import types

from mlindex.model_training import BenchmarkConditions as conditions


def test_every_bundle_has_its_own_tag():
    """Two bundles sharing a tag share an output directory and one manifest, and the second
    overwrites the first with no error."""
    tags = conditions.tags()
    assert len(tags) == len(conditions.CONDITIONS)
    assert len(set(tags)) == len(tags)


def test_every_tag_carries_the_prefix():
    """The prefix is what keeps this condition set's directories apart from another pass's."""
    assert all(tag.startswith(f'{conditions.TAG_PREFIX}_') for tag in conditions.tags())


def test_the_tag_rule_has_one_implementation():
    """`bundle_tag` accepts anything with the attribute names, which is how a run assembled from
    command-line flags tags itself. If that stops agreeing with a Condition's own tag, a
    hand-assembled run writes into the table-driven run's directory."""
    for condition in conditions.CONDITIONS:
        namespace = types.SimpleNamespace(
            error_multiplier=condition.error_multiplier,
            n_contaminants=condition.n_contaminants,
            intercept_scale=condition.intercept_scale,
            n_dropout=condition.n_dropout,
            second_phase_lines=condition.second_phase_lines,
            second_phase_bias=condition.second_phase_bias,
        )
        assert conditions.bundle_tag(namespace) == condition.tag


def test_a_non_default_axis_always_appears_in_the_tag():
    """Otherwise two bundles differing only on that axis collide."""
    for condition in conditions.CONDITIONS:
        if condition.intercept_scale != conditions.NOMINAL_INTERCEPT_SCALE:
            assert 'icept' in condition.tag
        if condition.n_dropout:
            assert f'drop{condition.n_dropout}' in condition.tag
        if condition.second_phase_lines:
            assert f'phase{condition.second_phase_lines}' in condition.tag


def test_the_control_is_not_zero_error():
    """With no measurement error the residual denominator vanishes and M20 diverges, which would
    make every residual-denominator merit unusable on the one bundle meant to prove the pipeline
    is sound."""
    control = conditions.BY_KEY['control']
    assert control.error_multiplier > 0
    assert control.error_multiplier < conditions.BY_KEY['nominal'].error_multiplier


def test_contamination_is_measured_at_one_and_two_lines():
    """A single point on the axis cannot show whether an effect scales with contamination."""
    counts = sorted(condition.n_contaminants for condition in conditions.CONDITIONS
                    if condition.axis == 'contamination')
    assert counts == [1, 2]


def test_the_sparsity_ladder_is_ascending_and_its_maximum_drives_the_nesting():
    """The holes are drawn once at the largest rung and prefixed, so the rungs must be sorted and
    the maximum must be the largest of them."""
    assert list(conditions.SPARSITY_LADDER) == sorted(conditions.SPARSITY_LADDER)
    assert conditions.MAX_NESTED_DROPOUT == max(conditions.SPARSITY_LADDER)


def test_the_hard_bundles_are_the_severe_end_and_are_flagged_not_derived():
    """Membership is a per-condition flag. Deriving it from the axis would sweep in a mild bundle
    added to an axis that already has a severe one, silently enlarging the hard population."""
    hard = [condition for condition in conditions.CONDITIONS if condition.is_hard]
    assert conditions.HARD_BUNDLES == tuple(condition.tag for condition in hard)
    assert conditions.BY_KEY['nominal'].tag not in conditions.HARD_BUNDLES
    assert conditions.BY_KEY['control'].tag not in conditions.HARD_BUNDLES
    # The milder rung of each two-point axis is not hard; the severe one is.
    assert not conditions.BY_KEY['contaminated1'].is_hard
    assert conditions.BY_KEY['contaminated2'].is_hard
    assert not conditions.BY_KEY['sparse2'].is_hard
    assert conditions.BY_KEY['sparse6'].is_hard


def test_the_digest_changes_when_the_table_does():
    """Generation records this digest and reduction refuses to pair arms across a change to it."""
    before = conditions.condition_set_digest()
    original = conditions.CONDITIONS[0].error_multiplier
    try:
        conditions.CONDITIONS[0].error_multiplier = original + 1.0
        assert conditions.condition_set_digest() != before
    finally:
        conditions.CONDITIONS[0].error_multiplier = original
    assert conditions.condition_set_digest() == before


def test_the_condition_row_reports_every_field_a_comparison_depends_on():
    row = conditions.condition_row(conditions.BY_KEY['second_phase'])
    for field in ('key', 'tag', 'axis', 'error_law', 'error_multiplier', 'intercept_scale',
                  'n_contaminants', 'n_dropout', 'second_phase_lines', 'second_phase_bias',
                  'is_hard'):
        assert field in row
    assert row['error_law'] == 'gaussian'
