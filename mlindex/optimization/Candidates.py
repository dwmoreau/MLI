import numpy as np

from mlindex.optimization.CandidateOptLoss import CandidateOptLoss
from mlindex.utilities.FigureOfMerits import get_M20
from mlindex.utilities.FigureOfMerits import get_M_rev_sym
from mlindex.utilities.FigureOfMerits import get_X_N
from mlindex.utilities.FigureOfMerits import get_n_over
from mlindex.utilities.FigureOfMerits import get_M20_likelihood
from mlindex.utilities.FigureOfMerits import get_M20_likelihood_from_xnn
from mlindex.utilities.FigureOfMerits import get_multiplicity_taupin88
from mlindex.utilities.MillerIndexAssignment import vectorized_subsampling
from mlindex.utilities.numba_functions import fast_assign
from mlindex.utilities.Q2Calculator import Q2Calculator
from mlindex.utilities.Reindexing import monoclinic_standardization
from mlindex.utilities.Reindexing import reindex_entry_basic
from mlindex.utilities.Reindexing import selling_reduction
from mlindex.utilities.SpaceGroups import get_spacegroup_hkl_ref
from mlindex.utilities.UnitCellTools import fix_unphysical
from mlindex.utilities.UnitCellTools import get_hkl_matrix
from mlindex.utilities.UnitCellTools import get_reciprocal_unit_cell_from_xnn
from mlindex.utilities.UnitCellTools import get_xnn_from_reciprocal_unit_cell
from mlindex.utilities.UnitCellTools import get_xnn_from_unit_cell
from mlindex.utilities.UnitCellTools import get_unit_cell_from_xnn
from mlindex.utilities.UnitCellTools import get_unit_cell_volume
from mlindex.utilities.UnitCellTools import reciprocal_uc_conversion


# The merits a research run can capture at the prune site. The cut reads a merit once per
# candidate, so this is one extra evaluation of each -- but it is off by default and the shipped
# path never touches it: `prune_criterion_capture` is not a CLI option and never will be
# (C2-F-008). It exists because campaign 1 stored only M20 at the cut, which is the one question
# its 57-million-row threshold-0 dump cannot answer (C2-R-001).
PRUNE_CAPTURE_MERITS = ('M20', 'M_tilde', 'M_rev', 'M_sym', 'X_N', 'n_over', 'max_gap')


class Candidates:
    def __init__(self, q2_obs, xnn, hkl_ref, lattice_system, bravais_lattice, opt_params, rng, fom, zero_error, wavelength):
        self.lattice_system = lattice_system
        self.bravais_lattice = bravais_lattice
        self.minimum_unit_cell = opt_params['minimum_uc']
        self.maximum_unit_cell = opt_params['maximum_uc']
        self.assignment_threshold = opt_params['assignment_threshold']
        self.figure_of_merit = opt_params['figure_of_merit']
        self.rng = rng
        self.fom = fom
        self.zero_error = zero_error
        self.wavelength = wavelength
        self.hkl_ref = hkl_ref
        self.hkl_ref_length = hkl_ref.shape[0]

        self.q2_obs = q2_obs
        self.n_peaks = self.q2_obs.size
        self.xnn = xnn
        self.n = self.xnn.shape[0]
        self.candidate_index = np.arange(self.n)
        self.fix_out_of_range_candidates()
        # Copied AFTER the repair, not before. fix_out_of_range_candidates replaces self.xnn
        # for any candidate outside the allowed cell range, and assign_hkls below scores the
        # repaired array -- so a copy taken first pairs the pre-repair cell with the post-repair
        # M20, and the two then describe different candidates for as long as that candidate goes
        # unimproved. Measured at 4 of 8 062 triclinic candidates on one pattern; the repair is
        # where the Selling reduction is fragile, so triclinic is where it bites.
        self.best_xnn = self.xnn.copy()

        self.q2_calculator = Q2Calculator(
            lattice_system=self.lattice_system,
            hkl=self.hkl_ref,
            tensorflow=False,
            representation='xnn'
            )
        # Research capture at the prune site, off unless opt_params asks for it.
        self.prune_criterion_capture = bool(opt_params.get('prune_criterion_capture', False))
        self.m20_at_prune = None
        self.merit_at_prune = None

        self.assign_hkls()
        if self.zero_error:
            self.correct_zero_error()
            self.best_zeropoint = self.zeropoint.copy()
        self.best_hkl = self.hkl.copy()
        self.best_M20 = self.M20

    def fix_bad_conversions(self):
        bad_conversions = np.sum(np.isnan(self.reciprocal_unit_cell), axis=1) > 0
        good_indices = np.arange(self.reciprocal_unit_cell.shape[0])[~bad_conversions]
        n_bad = np.sum(bad_conversions)
        if n_bad > 0:
            if n_bad > bad_conversions.size - n_bad:
                good_indices = self.rng.choice(good_indices, replace=True, size=n_bad)
            else:
                good_indices = self.rng.choice(good_indices, replace=False, size=n_bad)
            self.xnn[bad_conversions] = self.xnn[good_indices]
            self.reciprocal_unit_cell[bad_conversions] = self.reciprocal_unit_cell[good_indices]
            self.unit_cell[bad_conversions] = self.unit_cell[good_indices]

    def update_xnn_from_unit_cell(self):
        self.reciprocal_unit_cell = reciprocal_uc_conversion(
            self.unit_cell, partial_unit_cell=True, lattice_system=self.lattice_system
            )
        self.xnn = get_xnn_from_reciprocal_unit_cell(
            self.reciprocal_unit_cell, partial_unit_cell=True, lattice_system=self.lattice_system
            )
        self.fix_bad_conversions()

    def update_unit_cell_from_xnn(self):
        self.reciprocal_unit_cell = get_reciprocal_unit_cell_from_xnn(
            self.xnn, partial_unit_cell=True, lattice_system=self.lattice_system
            )
        self.unit_cell = reciprocal_uc_conversion(
            self.reciprocal_unit_cell, partial_unit_cell=True, lattice_system=self.lattice_system
            )
        self.fix_bad_conversions()

    def fix_out_of_range_candidates(self):
        self.xnn = fix_unphysical(
            xnn=self.xnn,
            rng=self.rng,
            minimum_unit_cell=self.minimum_unit_cell,
            maximum_unit_cell=self.maximum_unit_cell,
            lattice_system=self.lattice_system,
            )
        self.update_unit_cell_from_xnn()

    def assign_hkls(self):
        q2_ref_calc = self.q2_calculator.get_q2(self.xnn)
        hkl_assign = fast_assign(self.q2_obs, q2_ref_calc)
        self.hkl = np.take(self.hkl_ref, hkl_assign, axis=0)
        q2_calc = np.take_along_axis(q2_ref_calc, hkl_assign, axis=1)
        self.M20 = get_M20(self.q2_obs, q2_calc, q2_ref_calc)

    def correct_zero_error(self):
        q2_ref_calc = self.q2_calculator.get_q2(self.xnn)

        # Initialize a target function
        target_function = CandidateOptLoss(
            np.repeat(self.q2_obs[np.newaxis], self.n, axis=0),
            lattice_system=self.lattice_system,
            )
        target_function.update(self.hkl, self.xnn)

        # Get a per-peak correction for zero and add these to q_calc
        delta_gn = target_function.gauss_newton_step_zero_error(self.xnn, self.wavelength)
        self.xnn += delta_gn[:, :-1]
        self.zeropoint = delta_gn[:, -1]

        # Reassign hkl's
        q2_ref_calc = self.q2_calculator.get_q2(self.xnn)
        q2_ref_calc = target_function.apply_zeropoint(self.zeropoint, self.wavelength, q2_ref_calc)
        hkl_assign = fast_assign(self.q2_obs, q2_ref_calc)
        self.hkl = np.take(self.hkl_ref, hkl_assign, axis=0)
        q2_calc = np.take_along_axis(q2_ref_calc, hkl_assign, axis=1)
        self.M20 = get_M20(self.q2_obs, q2_calc, q2_ref_calc)

    def iteration_worker_common(self, target_function):
        self.xnn += target_function.gauss_newton_step(self.xnn)
        self.fix_out_of_range_candidates()
        self.assign_hkls()
        if self.zero_error:
            self.correct_zero_error()
        improved = self.M20 > self.best_M20
        self.best_M20[improved] = self.M20[improved]
        self.best_xnn[improved] = self.xnn[improved]
        self.best_hkl[improved] = self.hkl[improved]
        if self.zero_error:
            self.best_zeropoint[improved] = self.zeropoint[improved]

    def deterministic(self, iteration_info):
        target_function = CandidateOptLoss(
            np.repeat(self.q2_obs[np.newaxis], self.n, axis=0),
            lattice_system=self.lattice_system,
            )
        target_function.update(self.hkl, self.xnn)
        self.iteration_worker_common(target_function)

    def random_subsampling(self, iteration_info):
        n_keep = self.n_peaks - iteration_info['n_drop']
        if iteration_info['uniform_sampling']:
            subsampled_indices = self.rng.permuted(
                np.repeat(np.arange(self.n_peaks)[np.newaxis], self.n, axis=0),
                axis=1
                )[:, :n_keep]
        else:
            arg = 1 / self.q2_obs
            p = np.repeat((arg / np.sum(arg))[np.newaxis], self.n, axis=0)
            subsampled_indices = vectorized_subsampling(p, n_keep, self.rng)
        hkl_subsampled = np.take_along_axis(
            self.hkl, subsampled_indices[:, :, np.newaxis], axis=1
            )
        q2_subsampled = np.take(self.q2_obs, subsampled_indices)
        target_function = CandidateOptLoss(q2_subsampled, lattice_system=self.lattice_system)
        target_function.update(hkl_subsampled, self.xnn)
        self.iteration_worker_common(target_function)

    def random_subsampling_power(self, iteration_info):
        n_keep = self.n_peaks - iteration_info['n_drop']
        if iteration_info['uniform_sampling']:
            subsampled_indices = self.rng.permuted(
                np.repeat(np.arange(self.n_peaks)[np.newaxis], self.n, axis=0),
                axis=1
                )[:, :n_keep]
        else:
            arg = 1 / self.q2_obs
            p = np.repeat((arg / np.sum(arg))[np.newaxis], self.n, axis=0)
            subsampled_indices = vectorized_subsampling(p, n_keep, self.rng)
        power = -iteration_info['power'] * np.ones((self.n, self.n_peaks))
        np.put_along_axis(power, subsampled_indices, values=1, axis=1)

        target_function = CandidateOptLoss(
            np.repeat(self.q2_obs[np.newaxis], self.n, axis=0),
            lattice_system=self.lattice_system,
            )
        target_function.update(self.hkl, self.xnn, power=power)
        self.iteration_worker_common(target_function)

    def random_power(self, iteration_info):
        """
        svd-index:  Whkl = dobs^m |d2theta|, m: 0 <-> 4
        here:       W = 1 / (q2_obs |dq2|)
            q2 = 1/dobs
                    W = d_obs^(2m) / |dq2|   m: 0 <-> 2
                    W = 1 / (q2_obs^m |dq2|) m: 0 <-> 2
        """
        a = iteration_info['power_range'][1] - iteration_info['power_range'][0]
        b = iteration_info['power_range'][0]
        power = a*self.rng.random(size=(self.n, self.n_peaks)) + b

        target_function = CandidateOptLoss(
            np.repeat(self.q2_obs[np.newaxis], self.n, axis=0),
            lattice_system=self.lattice_system,
            )
        target_function.update(self.hkl, self.xnn, power)
        self.iteration_worker_common(target_function)

    def refine_cell(self):
        # This updates the unit cell only with the peaks assigned at > threshold probability.
        _, probability, _ = get_M20_likelihood_from_xnn(
            q2_obs=self.q2_obs,
            xnn=self.best_xnn,
            hkl=self.best_hkl,
            lattice_system=self.lattice_system,
            bravais_lattice=self.bravais_lattice,
            )
        indexed_peaks = probability > self.assignment_threshold
        n_indexed_peaks = np.sum(indexed_peaks, axis=1)
        unique_n_indexed_peaks = np.unique(n_indexed_peaks)
        refined_xnn = self.best_xnn.copy()
        for n in unique_n_indexed_peaks:
            candidate_indices = n_indexed_peaks == n
            subsampled_indices = np.argwhere(indexed_peaks[candidate_indices])
            # subsampled_indices: n_candidates x n_peaks
            # hkl:                n_candidates x n_peaks x 3
            subsampled_indices = subsampled_indices[:, 1].reshape((candidate_indices.sum(), n))
            hkl_subsampled = np.take_along_axis(
                self.best_hkl[candidate_indices],
                subsampled_indices[:, :, np.newaxis],
                axis=1
                )
            q2_subsampled = np.take(self.q2_obs, subsampled_indices)
            target_function = CandidateOptLoss(
                q2_subsampled,
                lattice_system=self.lattice_system,
                )
            target_function.update(hkl_subsampled, refined_xnn[candidate_indices])
            refined_xnn[candidate_indices] += target_function.gauss_newton_step(
                refined_xnn[candidate_indices]
                )
        refined_xnn = fix_unphysical(
            xnn=refined_xnn,
            rng=self.rng,
            minimum_unit_cell=self.minimum_unit_cell,
            maximum_unit_cell=self.maximum_unit_cell,
            lattice_system=self.lattice_system,
            )
        q2_ref_calc = self.q2_calculator.get_q2(refined_xnn)
        hkl_assign = fast_assign(self.q2_obs, q2_ref_calc)
        refined_hkl = np.take(self.hkl_ref, hkl_assign, axis=0)
        refined_q2_calc = np.take_along_axis(q2_ref_calc, hkl_assign, axis=1)

        if self.zero_error:
            target_function_zp = CandidateOptLoss(
                np.repeat(self.q2_obs[np.newaxis], self.n, axis=0),
                lattice_system=self.lattice_system,
                )
            target_function_zp.update(refined_hkl, refined_xnn)
            # Get a per-peak correction for zero and add these to q_calc
            delta_gn = target_function_zp.gauss_newton_step_zero_error(refined_xnn, self.wavelength)
            refined_xnn += delta_gn[:, :-1]
            refined_zeropoint = delta_gn[:, -1]

            q2_ref_calc = self.q2_calculator.get_q2(refined_xnn)
            q2_ref_calc = target_function_zp.apply_zeropoint(refined_zeropoint, self.wavelength, q2_ref_calc)
            hkl_assign = fast_assign(self.q2_obs, q2_ref_calc)
            refined_hkl = np.take(self.hkl_ref, hkl_assign, axis=0)
            refined_q2_calc = np.take_along_axis(q2_ref_calc, hkl_assign, axis=1)
            refined_xnn = fix_unphysical(
                xnn=refined_xnn,
                rng=self.rng,
                minimum_unit_cell=self.minimum_unit_cell,
                maximum_unit_cell=self.maximum_unit_cell,
                lattice_system=self.lattice_system,
                )

        refined_M20 = get_M20(self.q2_obs, refined_q2_calc, q2_ref_calc)
        improved = refined_M20 > self.best_M20
        self.best_hkl[improved] = refined_hkl[improved]
        self.best_M20[improved] = refined_M20[improved]
        self.best_xnn[improved] = refined_xnn[improved]
        if self.zero_error:
            self.best_zeropoint[improved] = refined_zeropoint[improved]

    def prune_below_m20(self, threshold=5.0):
        """Drop every candidate below the bar, keeping the best one if nothing clears it.

        `threshold` may be a scalar or a mapping keyed by Bravais lattice. The mapping form is
        what a per-lattice cut needs: one M20 value means a different surviving fraction on every
        lattice -- 1.3 % on oF against 6.1 % on cI at the same 5.0 -- so a single scalar is
        already a per-lattice policy, just an unchosen one (C2-F-023). Whether the values should
        differ is not settled; the shape is here so that settling it does not need a redesign.

        The bar is applied to `best_M20`, the running maximum over the search's iterates, computed
        against the lattice's full reference list. That is NOT the M20 the candidate goes on to
        report: `refine_cell` runs after this, and `assign_extinction_group` then replaces the
        stored value with the maximum over the lattice's extinction groups, which can only raise
        it. A correct cell deleted here at 3.5 is not a cell that would have reported 3.5 -- it is
        one that had not finished being fitted (C2-F-002, C2-F-020).
        """
        if isinstance(threshold, dict):
            threshold = threshold[self.bravais_lattice]
        keep = self.best_M20 >= threshold
        if not np.any(keep):
            keep[np.argmax(self.best_M20)] = True

        if self.prune_criterion_capture:
            # The value the rule tested, and the same for every candidate criterion, recorded
            # BEFORE the mask is applied so the survivors carry what decided their survival.
            # Keeping it is what lets one run answer every higher threshold by restriction
            # instead of needing a second run whose repair RNG has diverged.
            self.merit_at_prune = {name: values[keep] for name, values
                                   in self._capture_merits_at_prune().items()}
            self.m20_at_prune = self.best_M20[keep].copy()

        self.best_xnn = self.best_xnn[keep]
        self.best_M20 = self.best_M20[keep]
        self.best_hkl = self.best_hkl[keep]
        if self.zero_error:
            self.best_zeropoint = self.best_zeropoint[keep]
        self.n = self.best_xnn.shape[0]

    def _capture_merits_at_prune(self):
        """Every candidate criterion on the cells as they stand at the cut.

        Deliberately a one-shot recompute here rather than a running maximum beside `best_M20`:
        the inner loop is out of scope (DWMM, 2026-08-24), so the iterate this scores is the one
        M20 selected, which is exactly what a drop-in replacement cut would see.

        Route matters to the last bit. `q2_ref_calc` comes from the optimiser's own calculator on
        `best_xnn` and the assignment is redone with `fast_assign`, so the recomputed M20 must
        equal `best_M20` exactly -- that identity is the gate on this whole capture. Rebuilding
        q2_calc from the stored Miller indices instead would differ by an ULP that moves a line
        across M20's own cut-off (F-095).

        `get_M20` is called LAST because it is the only one of the four that modifies
        `q2_ref_calc`, via np.putmask.

        ONE KNOWN DISAGREEMENT, and it is not this routine's. The recomputed M20 equals
        `best_M20` on every candidate except those `fix_unphysical` repaired at construction and
        that never improved afterwards: `__init__` copies `best_xnn` *before*
        `fix_out_of_range_candidates` replaces `self.xnn`, then computes `best_M20` from the
        repaired array -- so for those rows the two describe different cells (C2-F-026). Measured
        at 4 of 8 062 triclinic candidates and 0 of 10 292 monoclinic and cubic ones. The
        difference is a diagnostic, not a defect in the capture, and it heals before anything is
        reported because `assign_extinction_group` recomputes M20 from `best_xnn` at the end.
        """
        if self.zero_error:
            raise NotImplementedError(
                'prune_criterion_capture does not support zero-error refinement: the per-'
                'candidate zeropoint would have to be applied to q2_ref_calc here, and the '
                'captured merits would not reproduce the pipeline value without it.'
                )
        q2_ref_calc = self.q2_calculator.get_q2(self.best_xnn)
        hkl_assign = fast_assign(self.q2_obs, q2_ref_calc)
        q2_calc = np.take_along_axis(q2_ref_calc, hkl_assign, axis=1)

        M_tilde, M_rev, M_sym = get_M_rev_sym(self.q2_obs, q2_calc, q2_ref_calc)
        n_over, max_gap = get_n_over(self.q2_obs, q2_calc, q2_ref_calc)
        X_N = get_X_N(self.q2_obs, q2_calc, q2_ref_calc)
        M20 = get_M20(self.q2_obs, q2_calc, q2_ref_calc)

        return {'M20': M20, 'M_tilde': M_tilde, 'M_rev': M_rev, 'M_sym': M_sym,
                'X_N': X_N.astype(np.float64), 'n_over': n_over.astype(np.float64),
                'max_gap': max_gap.astype(np.float64)}

    def standardize_cell(self):
        # These do a quick standardization of monoclinic and triclinic candidates. It is just a
        # Selling reduction.
        # This is performed in this function because the Miller indices are reassigned at the
        # end. Miller index transformations are not tracked in monoclinic_standardization
        if self.lattice_system in ['monoclinic', 'triclinic']:
            best_unit_cell = get_unit_cell_from_xnn(
                self.best_xnn, partial_unit_cell=True, lattice_system=self.lattice_system
                )
            if self.lattice_system == 'monoclinic':
                best_standardized_unit_cell = monoclinic_standardization(
                    best_unit_cell, partial_unit_cell=True
                    )
            elif self.lattice_system == 'triclinic':
                best_standardized_unit_cell, _, _ = selling_reduction(best_unit_cell)

            # Some of the Selling reductions result in failures - these unit cells become NAN
            # Find these and replace them with the initial unit cell.
            failed = np.any(np.isnan(best_standardized_unit_cell), axis=1)
            if np.sum(failed) > 0:
                #print('Failure in Standardization - NaN check\n    numpy warning has been caught and corrected')
                best_standardized_unit_cell[failed] = best_unit_cell[failed]

            # Some of the standardized unit cells are unphysical. The standardization fails but does
            # not return a NAN. Find these and replace them with the initial unit cell
            best_standardized_unit_cell_check = fix_unphysical(
                unit_cell=best_standardized_unit_cell,
                rng=self.rng,
                minimum_unit_cell=self.minimum_unit_cell,
                maximum_unit_cell=self.maximum_unit_cell,
                lattice_system=self.lattice_system,
                )
            failed = np.any(
                best_standardized_unit_cell_check != best_standardized_unit_cell,
                axis=1
                )
            if np.sum(failed) > 0:
                #print('Failure Standardization - unphysical check\n    numpy warning has been caught and corrected')
                best_standardized_unit_cell[failed] = best_unit_cell[failed]

            # Still some unphysical unit cells make it through. These cannot be converted between
            # reciprocal and real space.
            best_standardized_xnn = get_xnn_from_unit_cell(
                best_standardized_unit_cell,
                partial_unit_cell=True,
                lattice_system=self.lattice_system
                )
            failed = np.any(np.isnan(best_standardized_xnn), axis=1)
            #if np.sum(failed) > 0:
            #    print('Failure in Standardization - Final check\n    numpy warning has been caught and corrected')
            # The write-back is unconditional. It used to sit inside `if np.sum(failed) > 0`, so on
            # any run where nothing NaN'd out -- the common case -- the monoclinic standardization
            # and the triclinic Selling reduction above were computed and then thrown away, and
            # best_xnn kept its unstandardized value. The two branches above are not affected:
            # they write into a local. This one is the only one that has to reach self.
            self.best_xnn[~failed] = best_standardized_xnn[~failed]

    def correct_off_by_two(self):
        #mult_factor = np.array([1, 1/2, np.sqrt(2), 2, 3, 4])
        mult_factor = np.array([1, 1/2, np.sqrt(2), 2])
        if self.lattice_system == 'cubic':
            mult_factors = mult_factor[:, np.newaxis]
        elif self.lattice_system in ['hexagonal', 'tetragonal']:
            mult_factors = np.ones((mult_factor.size**2, 2))
            mf_index = 0
            for mf0 in mult_factor:
                for mf1 in mult_factor:
                    mult_factors[mf_index, 0] = mf0
                    mult_factors[mf_index, 1] = mf1
                    mf_index += 1
        elif self.lattice_system == 'rhombohedral':
            mult_factors = np.ones((mult_factor.size, 2))
            mult_factors[:, 0] = mult_factor
            mult_factors[:, 1] = mult_factor
        elif self.lattice_system in ['orthorhombic', 'monoclinic', 'triclinic']:
            mult_factors = np.ones((mult_factor.size**3, self.xnn.shape[1]))
            mf_index = 0
            for mf0 in mult_factor:
                for mf1 in mult_factor:
                    for mf2 in mult_factor:
                        mult_factors[mf_index, 0] = mf0
                        mult_factors[mf_index, 1] = mf1
                        mult_factors[mf_index, 2] = mf2
                        if self.lattice_system == 'monoclinic':
                            mult_factors[mf_index, 3] = np.sqrt(mf0 * mf2)
                        elif self.lattice_system == 'triclinic':
                            mult_factors[mf_index, 3] = np.sqrt(mf1 * mf2)
                            mult_factors[mf_index, 4] = np.sqrt(mf0 * mf2)
                            mult_factors[mf_index, 5] = np.sqrt(mf0 * mf1)
                        mf_index += 1

        # Only the winning factor per candidate is ever read, so a running best
        # replaces the per-factor arrays. Those were [n, n_mult, n_peaks, 3] and
        # [n, n_mult, xnn_length]: 103 MB and 7 MB for a 3519-candidate
        # monoclinic run, of which at most 1/64 was used. np.argmax takes the
        # *first* maximum, so the update below is guarded by a strict '>' and
        # selects the same factor. Measured 1.41x with bit-identical output
        # (tools/repro_offbytwo.py).
        best_mf_M20 = np.full(self.n, -np.inf)
        best_mf_index = np.zeros(self.n, dtype=int)
        best_mf_hkl = np.zeros([self.n, self.n_peaks, 3])
        best_mf_xnn = np.zeros([self.n, self.best_xnn.shape[1]])
        if self.zero_error:
            best_mf_zeropoint = np.zeros(self.n)
        for mf_index in range(mult_factors.shape[0]):
            xnn_mult = mult_factors[mf_index, :][np.newaxis]**2 * self.best_xnn
            xnn_mult = fix_unphysical(
                xnn=xnn_mult,
                rng=self.rng,
                minimum_unit_cell=self.minimum_unit_cell,
                maximum_unit_cell=self.maximum_unit_cell,
                lattice_system=self.lattice_system,
                )
            q2_ref_calc_mult = self.q2_calculator.get_q2(xnn_mult)
            hkl_assign = fast_assign(self.q2_obs, q2_ref_calc_mult)
            hkl_mult = np.take(self.hkl_ref, hkl_assign, axis=0)
            q2_calc_mult = np.take_along_axis(q2_ref_calc_mult, hkl_assign, axis=1)
            if self.zero_error:
                target_function_zp = CandidateOptLoss(
                    np.repeat(self.q2_obs[np.newaxis], self.n, axis=0),
                    lattice_system=self.lattice_system,
                    )
                target_function_zp.update(hkl_mult, xnn_mult)
                # Get a per-peak correction for zero and add these to q_calc
                delta_gn = target_function_zp.gauss_newton_step_zero_error(xnn_mult, self.wavelength)
                xnn_mult += delta_gn[:, :-1]
                zeropoint_mult = delta_gn[:, -1]

                q2_ref_calc_mult = target_function_zp.apply_zeropoint(
                    zeropoint_mult, self.wavelength, q2_ref_calc_mult
                )
                hkl_assign = fast_assign(self.q2_obs, q2_ref_calc_mult)
                hkl_mult = np.take(self.hkl_ref, hkl_assign, axis=0)
                q2_calc_mult = np.take_along_axis(q2_ref_calc_mult, hkl_assign, axis=1)
            M20_mult = get_M20(self.q2_obs, q2_calc_mult, q2_ref_calc_mult)

            # Strict '>' so an equal M20 never displaces an earlier factor,
            # matching np.argmax's first-maximum rule.
            better = M20_mult > best_mf_M20
            best_mf_M20[better] = M20_mult[better]
            best_mf_index[better] = mf_index
            best_mf_hkl[better] = hkl_mult[better]
            best_mf_xnn[better] = xnn_mult[better]
            if self.zero_error:
                best_mf_zeropoint[better] = zeropoint_mult[better]

        # Index 0 of mult_factors is always the identity transformation (mult_factor[0]=1).
        # When a non-identity factor gives a better M20, keep the original candidate and
        # append the off-by-two corrected version rather than replacing the original.
        improved = best_mf_index != 0
        if np.any(improved):
            self.best_xnn = np.concatenate([self.best_xnn, best_mf_xnn[improved]], axis=0)
            self.best_M20 = np.concatenate([self.best_M20, best_mf_M20[improved]])
            self.best_hkl = np.concatenate([self.best_hkl, best_mf_hkl[improved]], axis=0)
            if self.zero_error:
                self.best_zeropoint = np.concatenate(
                    [self.best_zeropoint, best_mf_zeropoint[improved]])
            if self.m20_at_prune is not None:
                # An appended row is a rescaling of its parent, and the parent is what the cut
                # tested. Giving the child its parent's at-prune values is what makes a
                # restriction of this run reproduce the cut exactly: a child whose parent the cut
                # would have deleted disappears with it.
                self.m20_at_prune = np.concatenate(
                    [self.m20_at_prune, self.m20_at_prune[improved]])
                self.merit_at_prune = {
                    name: np.concatenate([values, values[improved]])
                    for name, values in self.merit_at_prune.items()}
            self.n = self.best_xnn.shape[0]

        # do quick reindexing to enforce constraints
        if self.lattice_system == 'triclinic':
            # The triclinic reindexing is a Selling reduction performed in direct space.
            # There are numerical issues with my implementation of this reduction that need to be
            # worked on. It occasionally produces triclinic unit cells that are not physical. This
            # produces a np.nan and a warning when the unit cell is converted from direct to
            # reciprocal space.
            unit_cell = get_unit_cell_from_xnn(
                self.best_xnn, partial_unit_cell=True, lattice_system=self.lattice_system
                )
            unit_cell = reindex_entry_basic(
                unit_cell,
                lattice_system=self.lattice_system,
                bravais_lattice=self.bravais_lattice,
                space='direct'
                )
            unit_cell = fix_unphysical(
                unit_cell=unit_cell,
                rng=self.rng,
                minimum_unit_cell=self.minimum_unit_cell, 
                maximum_unit_cell=self.maximum_unit_cell,
                lattice_system=self.lattice_system,
                )
            self.best_xnn = get_xnn_from_unit_cell(
                unit_cell, partial_unit_cell=True, lattice_system=self.lattice_system
                )
        else:
            reciprocal_unit_cell = get_reciprocal_unit_cell_from_xnn(
                self.best_xnn, partial_unit_cell=True, lattice_system=self.lattice_system
                )
            reciprocal_unit_cell = reindex_entry_basic(
                reciprocal_unit_cell,
                lattice_system=self.lattice_system,
                bravais_lattice=self.bravais_lattice,
                space='reciprocal'
                )
            self.best_xnn = get_xnn_from_reciprocal_unit_cell(
                reciprocal_unit_cell, partial_unit_cell=True, lattice_system=self.lattice_system
                )

    def assign_extinction_group(self):
        hkl_ref_sg = get_spacegroup_hkl_ref(self.hkl_ref, bravais_lattice=self.bravais_lattice)
        spacegroups = list(hkl_ref_sg.keys())

        M20 = np.zeros((self.n, len(spacegroups)))
        hkl = np.zeros([self.n, self.n_peaks, 3, len(spacegroups)])
        if self.zero_error:
            best_zeropoint = self.best_zeropoint

        for spacegroup_index, spacegroup in enumerate(spacegroups):
            q2_ref_calc = Q2Calculator(
                lattice_system=self.lattice_system,
                hkl=hkl_ref_sg[spacegroup],
                tensorflow=False,
                representation='xnn'
                ).get_q2(self.best_xnn)

            if self.zero_error:
                target_function_zp = CandidateOptLoss(
                    np.repeat(self.q2_obs[np.newaxis], self.best_xnn.shape[0], axis=0),
                    lattice_system=self.lattice_system,
                )
                q2_ref_calc = target_function_zp.apply_zeropoint(
                    best_zeropoint, self.wavelength, q2_ref_calc
                )
            hkl_assign = fast_assign(self.q2_obs, q2_ref_calc)
            hkl[:, :, :, spacegroup_index] = np.take(hkl_ref_sg[spacegroup], hkl_assign, axis=0)
            q2_calc = np.take_along_axis(q2_ref_calc, hkl_assign, axis=1)
            M20[:, spacegroup_index] = get_M20(self.q2_obs, q2_calc, q2_ref_calc)

        best_indices = np.argmax(M20, axis=1)
        self.best_spacegroup = list(np.take(spacegroups, best_indices))
        self.best_M20 = np.take_along_axis(M20, best_indices[:, np.newaxis], axis=1)[:, 0]
        self.best_hkl = np.take_along_axis(
            hkl, best_indices[:, np.newaxis, np.newaxis, np.newaxis], axis=3
            )[:, :, :, 0]


    def calculate_peaks_indexed(self):
        if self.zero_error:
            target_function_zp = CandidateOptLoss(
                np.repeat(self.q2_obs[np.newaxis], self.n, axis=0),
                lattice_system=self.lattice_system,
                )
            hkl2 = get_hkl_matrix(self.best_hkl, self.lattice_system)
            q2_calc = np.sum(hkl2 * self.best_xnn[:, np.newaxis, :], axis=2)
            q2_calc = target_function_zp.apply_zeropoint(self.best_zeropoint, self.wavelength, q2_calc)

            reciprocal_unit_cell = get_reciprocal_unit_cell_from_xnn(
                self.best_xnn, partial_unit_cell=True, lattice_system=self.lattice_system
                )
            reciprocal_volume = get_unit_cell_volume(
                reciprocal_unit_cell, partial_unit_cell=True, lattice_system=self.lattice_system
                )
            _, probability, self.best_Minfo = get_M20_likelihood(
                self.q2_obs, q2_calc, self.bravais_lattice, reciprocal_volume
                )
        else:
            _, probability, self.best_Minfo = get_M20_likelihood_from_xnn(
                q2_obs=self.q2_obs,
                xnn=self.best_xnn,
                hkl=self.best_hkl,
                lattice_system=self.lattice_system,
                bravais_lattice=self.bravais_lattice,
                )

        self.n_indexed = np.sum(
            probability > self.assignment_threshold,
            axis=1, dtype=int
            )
        probability_ = probability.copy()
