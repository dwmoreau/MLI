"""One pattern's candidate pool, generated separately by each generator.

The indexer draws its candidates from three generators at once and refines the mixture. To choose
how much of the budget each generator should get, they have to be drawn separately and kept apart,
with every candidate's distance to the true cell recorded -- which is what this does, for one
pattern whose answer is known.

It is generation only. Nothing here refines, scores or mixes; a pool is written once and then
scored many times, because the scoring is what gets changed and the generation is what costs.
"""
import numpy as np

from mlindex.utilities.Allocation import largest_remainder
from mlindex.utilities.Reindexing import reindex_entry_triclinic
from mlindex.utilities.UnitCellTools import fix_unphysical
from mlindex.utilities.UnitCellTools import get_xnn_from_unit_cell


def refuse_unfilled(xnn, generator_names, candidates_per_model):
    """Every slot starts as NaN and is filled by the generator that owns it.

    A slot left over -- which happens when a generator's share does not divide evenly among its
    split groups -- would otherwise reach the caller as a candidate at an undefined distance, and
    a distance of NaN is silently treated as "beyond the curve" by everything downstream.
    """
    unfilled = np.isnan(xnn).any(axis=-1).sum(axis=0)
    if unfilled.any():
        raise ValueError(
            f'candidate slots were never filled: '
            f'{dict(zip(generator_names, unfilled.tolist()))}. {candidates_per_model} does not '
            f'divide among the split groups of every generator'
            )


def generate_candidate_pools(optimizer, entry, candidates_per_model, rng):
    """Draw `candidates_per_model` candidates from each generator for one known-answer pattern.

    Returns (xnn, xnn_true, generator_names):

      xnn              (candidates_per_model, n_generators, n_cell_parameters), every candidate's
                       position in the lattice system's own xnn parameters
      xnn_true         (n_cell_parameters,), the pattern's real answer
      generator_names  the generator each column came from, in column order

    Distances are the caller's one-liner, `np.linalg.norm(xnn - xnn_true, axis=-1)`. The positions
    are what is returned because a candidate's neighbours cannot be recovered from its distance.

    A generator split across several split groups fills consecutive slices of its own column, and
    each column is then permuted, because both the forest and the network emit their candidates in
    split-group order and the network emits its most probable cells first. Without that, taking the
    first k of a column would take a biased k rather than a sample of it.
    """
    abnn_top_n = None
    n_sub_generators = dict()
    candidates_per_sub_model = dict()
    generator_names = []
    for generator_info in optimizer.opt_params['generator_info']:
        if generator_info['generator'] in n_sub_generators.keys():
            n_sub_generators[generator_info['generator']] += 1
        else:
            n_sub_generators[generator_info['generator']] = 1
            generator_names.append(generator_info['generator'])
    # A generator's share is divided among its split groups, and the division rarely comes out
    # even -- hexagonal has eight groups and a budget of 1996. Floor-dividing leaves the remainder
    # ungenerated, so the shared allocator is used: equal weights, leftovers to the earliest
    # groups by its tie rule, counts summing to the budget exactly. Identical to the even split
    # this replaces, which `test_ties_go_to_the_earliest_claimant` pins.
    for key, n_sub in n_sub_generators.items():
        candidates_per_sub_model[key] = largest_remainder(
            np.ones(n_sub), candidates_per_model).tolist()
    taken = {key: 0 for key in n_sub_generators}

    xnn_true = np.array(entry['reindexed_xnn'])[optimizer.wrapper.data_params['unit_cell_indices']]
    q2 = np.array(entry['q2'])[:optimizer.n_peaks]
    xnn = np.full(
        (candidates_per_model, len(n_sub_generators.keys()), xnn_true.size),
        np.nan
        )
    filled = np.zeros(len(n_sub_generators.keys()), dtype=int)

    for generator_info in optimizer.opt_params['generator_info']:
        name = generator_info['generator']
        wanted = candidates_per_sub_model[name][taken.get(name, 0)] if name in taken else None
        taken[name] = taken.get(name, 0) + 1
        if generator_info['generator'] == 'trees':
            generator_unit_cells = optimizer.wrapper.random_forest_generator[generator_info['split_group']].generate(
                wanted, rng,  q2,
                )
        elif generator_info['generator'] == 'abnn':
            if abnn_top_n is None:
                abnn_top_n = optimizer.wrapper.abnn_generator[generator_info['split_group']].model_params['n_volumes']
            generator_unit_cells = optimizer.wrapper.abnn_generator[generator_info['split_group']].generate(
                wanted, rng, q2,
                top_n=abnn_top_n,
                batch_size=2,
                )
        elif generator_info['generator'] == 'templates':
            generator_unit_cells = optimizer.wrapper.miller_index_templator[optimizer.bravais_lattice].generate(
                wanted, rng, q2, 
                )
        else:
            # As in MPIOptimizer: without this the previous generator's cells are reused.
            raise ValueError(
                f"unknown generator {generator_info['generator']!r} in generator_info"
                )

        generator_unit_cells = fix_unphysical(
            unit_cell=generator_unit_cells,
            rng=rng,
            minimum_unit_cell=optimizer.opt_params['minimum_uc'],
            maximum_unit_cell=optimizer.opt_params['maximum_uc'],
            lattice_system=optimizer.wrapper.data_params['lattice_system']
            )
        if optimizer.wrapper.data_params['lattice_system'] == 'triclinic':
            generator_unit_cells, _ = reindex_entry_triclinic(generator_unit_cells)
        generator_xnn = get_xnn_from_unit_cell(
            generator_unit_cells,
            partial_unit_cell=True,
            lattice_system=optimizer.wrapper.data_params['lattice_system']
            )
        generator_index = list(n_sub_generators.keys()).index(generator_info['generator'])
        start = filled[generator_index]
        stop = start + generator_xnn.shape[0]
        xnn[start: stop, generator_index] = generator_xnn
        filled[generator_index] = stop

    # The forest emits its candidates in split-group order, or in dominant-zone bin order, so the
    # column is permuted before anyone takes a prefix of it. The permutation is drawn as an index
    # so that it moves every column of xnn together; drawing it this way takes the same numbers
    # off the generator as permuting the values would.
    tree_index = list(n_sub_generators.keys()).index('trees')
    order = rng.permutation(candidates_per_model)
    xnn[:, tree_index] = xnn[order, tree_index]

    # The network's candidates are ordered too, and in two tiers: within each split group the
    # first abnn_top_n are its most probable cells and the rest are resampled Miller-index
    # labellings of them. The tiers are permuted separately and the probable ones kept in front,
    # so that a prefix of the column is the best of each tier rather than the best of one group.
    abnn_index = list(n_sub_generators.keys()).index('abnn')
    if abnn_top_n < min(candidates_per_sub_model['abnn']):
        index_top_n = []
        index_lower = []
        start = 0
        for count in candidates_per_sub_model['abnn']:
            index_top_n.extend(range(start, start + abnn_top_n))
            index_lower.extend(range(start + abnn_top_n, start + count))
            start += count
        order = np.concatenate([
            rng.permutation(np.array(index_top_n, dtype=int)),
            rng.permutation(np.array(index_lower, dtype=int))
            ])
        xnn[:start, abnn_index] = xnn[order, abnn_index]

    refuse_unfilled(xnn, generator_names, candidates_per_model)
    return xnn, xnn_true, generator_names
