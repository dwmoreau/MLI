!!! THIS IS MEANT TO BE RUN IN IPYTHON !!!

import numpy as np
import os
from mpi4py import MPI
import pandas as pd

from mlindex.optimization.Optimizer import OptimizerBase
from mlindex.optimization.Optimizer import OptimizerManager
from mlindex.optimization.Optimizer import Candidates
from mlindex.optimization.UtilitiesOptimizer import get_triclinic_optimizer
from mlindex.optimization.UtilitiesOptimizer import get_orthorhombic_optimizer
from mlindex.optimization.UtilitiesOptimizer import get_monoclinic_optimizer
from mlindex.optimization.UtilitiesOptimizer import get_tetragonal_optimizer
from mlindex.utilities.FigureOfMerits import get_M20
from mlindex.utilities.FigureOfMerits import get_M20_from_xnn
from mlindex.utilities.FigureOfMerits import get_q2_calc_triplets
from mlindex.utilities.FigureOfMerits import get_M20_likelihood

from mlindex.model_training.IntegralFilter import IntegralFilter
from mlindex.model_training.MITemplates import MITemplates
from mlindex.optimization.CandidateOptLoss import CandidateOptLoss

%load_ext line_profiler

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n_ranks = comm.Get_size()
split_comm = comm.Split(color=rank, key=rank)

load_data = True
broadening_tag = '1'
q2_error_params = np.array([0.000000001, 0])
n_top_candidates = 20
rng = np.random.default_rng()

tag = 'C7N2O2Cl_fd'
base_dir = '/global/cfs/cdirs/m4064/dwmoreau/MLI/mlindex/data/experimental_data/gsas'
unit_cell = np.array([8.8698, 8.8698, 12.6644, np.pi/2, np.pi/2, np.pi/2]),
bravais_lattice = 'mP'
#optimizer = get_tetragonal_optimizer(bravais_lattice, broadening_tag, split_comm, options=None)
#optimizer = get_triclinic_optimizer(bravais_lattice, broadening_tag, split_comm, options=None)
#optimizer = get_orthorhombic_optimizer(bravais_lattice, broadening_tag, split_comm, options=None)
optimizer = get_monoclinic_optimizer(bravais_lattice, broadening_tag, 1, split_comm, options=None)
q2_obs = np.load(os.path.join(base_dir, tag, f'{tag}_peak_list.npy'))


%lprun -f OptimizerBase.run_common optimizer.run(q2=q2_obs, n_top_candidates=n_top_candidates)
Total time: 14.1579 s
File: /global/cfs/cdirs/m4064/dwmoreau/MLI/mlindex/optimization/Optimizer.py
Function: OptimizerBase.run_common at line 569

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   569                                               def run_common(self, n_top_candidates):
   570         1      62200.0  62200.0      0.0          self.comm.Bcast(self.q2_obs, root=self.root)
   571         1      32984.0  32984.0      0.0          self.triplets = self.comm.bcast(self.triplets, root=self.root)
   572         1 4680268685.0 4.68e+09     33.1          candidates = self.generate_candidates_rank()
   573         1       6392.0   6392.0      0.0          if self.opt_params['redistribution_testing']:
   574                                                       return None
   575                                           
   576         3       7143.0   2381.0      0.0          for iteration_info in self.opt_params['iteration_info']:
   577        63     102977.0   1634.6      0.0              for iter_index in range(iteration_info['n_iterations']):
   578        61      89251.0   1463.1      0.0                  if iteration_info['worker'] == 'random_subsampling':
   579        60 5620753408.0 9.37e+07     39.7                      candidates.random_subsampling(iteration_info)
   580         1        541.0    541.0      0.0                  elif iteration_info['worker'] == 'random_subsampling_power':
   581                                                               candidates.random_subsampling_power(iteration_info)
   582         1        461.0    461.0      0.0                  elif iteration_info['worker'] == 'random_power':
   583                                                               candidates.random_power(iteration_info)
   584         1        721.0    721.0      0.0                  elif iteration_info['worker'] == 'deterministic':
   585         1  100378476.0    1e+08      0.7                      candidates.deterministic(iteration_info)
   586                                           
   587                                                   # This meant to be run at the end of optimization to remove very similar candidates
   588                                                   # If this isn't run, the results will be spammed with many candidates that are nearly
   589                                                   # identical.
   590                                                   # This method takes pairwise differences in Xnn space and combines candidates that are 
   591                                                   # closer than some given radius
   592                                                   # If this were performed with all the entries, it would be slow and memory intensive.
   593                                                   # Instead the candidates are sorted by reciprocal unit cell volume and filtering is
   594                                                   # performed in chunks.
   595                                           
   596                                                   # Check to see if a better M20 score can be found by multiplying the unit cell by 2 along
   597                                                   # each axis. This also performs a quick reindexing.
   598                                                   # Check which spacegroup gives the best M20 score.
   599                                                   # Then calculate the number of assigned peaks (probability > 50%)
   600         1   98680360.0 9.87e+07      0.7          candidates.refine_cell()
   601         1   51916721.0 5.19e+07      0.4          candidates.standardize_cell()
   602         1 1580213089.0 1.58e+09     11.2          candidates.correct_off_by_two()
   603         1  570126527.0  5.7e+08      4.0          candidates.assign_extinction_group()
   604         1    8546168.0 8.55e+06      0.1          candidates.calculate_peaks_indexed()
   605         1       2665.0   2665.0      0.0          if self.opt_params['convergence_testing']:
   606                                                       self.convergence_testing(candidates)
   607                                                   else:
   608         1 1446744335.0 1.45e+09     10.2              self.downsample_candidates(candidates, n_top_candidates)


%lprun -f OptimizerManager.generate_candidates_rank optimizer.run(q2=q2_obs, n_top_candidates=n_top_candidates)
Total time: 4.69495 s
File: /global/cfs/cdirs/m4064/dwmoreau/MLI/mlindex/optimization/Optimizer.py
Function: OptimizerManager.generate_candidates_rank at line 764

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   764                                               def generate_candidates_rank(self):
   765         1       2886.0   2886.0      0.0          if self.opt_params['convergence_testing']:
   766                                                       candidate_unit_cells_all = perturb_xnn(
   767                                                           self.xnn_true,
   768                                                           convergence_candidates=self.opt_params['convergence_candidates'],
   769                                                           convergence_distances=self.opt_params['convergence_distances'],
   770                                                           minimum_uc=self.opt_params['minimum_uc'],
   771                                                           maximum_uc=self.opt_params['maximum_uc'],
   772                                                           lattice_system=self.lattice_system,
   773                                                           rng=self.rng
   774                                                       )
   775                                                   else:
   776         1        561.0    561.0      0.0              candidate_unit_cells_all = []
   777        15      19237.0   1282.5      0.0              for generator_info in self.opt_params['generator_info']:
   778        14      34275.0   2448.2      0.0                  if generator_info['generator'] == 'trees':
   779        12  757191133.0 6.31e+07     16.1                      generator_unit_cells = self.wrapper.random_forest_generator[generator_info['split_group']].generate(
   780         6       4569.0    761.5      0.0                          generator_info['n_unit_cells'], self.rng, self.q2_obs,
   781                                                                   )
   782         8       5840.0    730.0      0.0                  elif generator_info['generator'] == 'templates':
   783         2 2009497938.0    1e+09     42.8                      generator_unit_cells = self.wrapper.miller_index_templator[self.bravais_lattice].generate(
   784         1        711.0    711.0      0.0                          generator_info['n_unit_cells'], self.rng, self.q2_obs,
   785                                                                   )
   786         7       3557.0    508.1      0.0                  elif generator_info['generator'] == 'integral_filter':
   787                                                               # We only do one inference, so batch_size=total_size=1 makes sense 
   788                                                               # but batch size of 2 is faster than one ....
   789        12 1650458203.0 1.38e+08     35.2                      generator_unit_cells = self.wrapper.integral_filter_generator[generator_info['split_group']].generate(
   790         6       4774.0    795.7      0.0                          generator_info['n_unit_cells'], self.rng, self.q2_obs,
   791         6       2325.0    387.5      0.0                          batch_size=2, 
   792                                                                   )
793         1        682.0    682.0      0.0                  elif generator_info['generator'] in ['random', 'predicted_volume']:
   794         2   56592559.0 2.83e+07      1.2                      generator_unit_cells = self.wrapper.random_unit_cell_generator[self.bravais_lattice].generate(
   795         1       1723.0   1723.0      0.0                          generator_info['n_unit_cells'], self.rng, self.q2_obs,
   796         1        371.0    371.0      0.0                          model=generator_info['generator'],
   797                                                                   )
   798        14      18777.0   1341.2      0.0                  candidate_unit_cells_all.append(generator_unit_cells)
   799         1      50217.0  50217.0      0.0              candidate_unit_cells_all = np.concatenate(candidate_unit_cells_all, axis=0)
   800                                           
   801         2     197129.0  98564.5      0.0          candidate_unit_cells_all = fix_unphysical(
   802         1        401.0    401.0      0.0              unit_cell=candidate_unit_cells_all,
   803         1        380.0    380.0      0.0              rng=self.rng,
   804         1        811.0    811.0      0.0              minimum_unit_cell=self.opt_params['minimum_uc'],
   805         1        470.0    470.0      0.0              maximum_unit_cell=self.opt_params['maximum_uc'],
   806         1        410.0    410.0      0.0              lattice_system=self.lattice_system
   807                                                       )
   808         2      71999.0  35999.5      0.0          candidate_unit_cells_all = reindex_entry_basic(
   809         1        350.0    350.0      0.0              candidate_unit_cells_all,
   810         1        371.0    371.0      0.0              lattice_system=self.lattice_system,
   811         1        350.0    350.0      0.0              bravais_lattice=self.bravais_lattice,
   812         1        370.0    370.0      0.0              space='direct'
   813                                                       )
   814         2    2937967.0 1.47e+06      0.1          candidate_xnn_all = get_xnn_from_unit_cell(
   815         1        381.0    381.0      0.0              candidate_unit_cells_all,
   816         1        381.0    381.0      0.0              partial_unit_cell=True,
   817         1        331.0    331.0      0.0              lattice_system=self.lattice_system
   818                                                       )
   819                                           
   820         1        842.0    842.0      0.0          if self.opt_params['redistribution_testing']:
   821                                                       self.redistrubution_testing(candidate_xnn_all)
   822         1        641.0    641.0      0.0          elif self.opt_params['convergence_testing'] == False:
   823         1  141014249.0 1.41e+08      3.0              candidate_xnn_all = self.redistribute_xnn(candidate_xnn_all)
   824                                           
   825         1      12253.0  12253.0      0.0          self.sent_candidates = np.zeros(self.n_ranks, dtype=int)
   826         2       4117.0   2058.5      0.0          for rank_index in range(self.n_ranks):
   827         1       5631.0   5631.0      0.0              self.sent_candidates[rank_index] = candidate_xnn_all[rank_index::self.n_ranks].shape[0]
   828         1        831.0    831.0      0.0              if rank_index == self.root:
   829         1        621.0    621.0      0.0                  candidate_xnn_rank = candidate_xnn_all[rank_index::self.n_ranks]
   830                                                       else:
   831                                                           self.comm.send(candidate_xnn_all[rank_index::self.n_ranks], dest=rank_index)
   832         1   76803447.0 7.68e+07      1.6          return self.generate_candidates_common(candidate_xnn_rank)


%lprun -f IntegralFilter.generate optimizer.run(q2=q2_obs, n_top_candidates=n_top_candidates)
Total time: 1.64433 s
File: /global/cfs/cdirs/m4064/dwmoreau/MLI/mlindex/model_training/IntegralFilter.py
Function: IntegralFilter.generate at line 849

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   849                                               def generate(self, n_unit_cells, rng, q2_obs, top_n=None, batch_size=None):
   850         6      33935.0   5655.8      0.0          from mlindex.utilities.Q2Calculator import Q2Calculator
   851         6      13727.0   2287.8      0.0          from mlindex.utilities.numba_functions import fast_assign
   852         6       2486.0    414.3      0.0          if top_n is None:
   853         6       5411.0    901.8      0.0              top_n = self.model_params['n_volumes']
   854                                           
   855        12     212596.0  17716.3      0.0          q2_calculator = Q2Calculator(
   856         6       2594.0    432.3      0.0              lattice_system=self.lattice_system,
   857         6       2716.0    452.7      0.0              hkl=self.hkl_ref,
   858         6       2153.0    358.8      0.0              tensorflow=False,
   859         6       2115.0    352.5      0.0              representation='xnn'
   860                                                       )
   861                                           
   862         6       3746.0    624.3      0.0          if top_n > n_unit_cells:
   863                                                       xnn_gen, _ = self.predict_xnn(n_unit_cells, q2_obs=q2_obs[np.newaxis], batch_size=batch_size)
   864                                                       xnn_gen = xnn_gen[0]
   865                                                       q2_ref_calc = q2_calculator.get_q2(xnn_gen)
   866                                                       hkl_assign = fast_assign(q2_obs, q2_ref_calc)
   867                                                       hkl = np.take(self.hkl_ref, hkl_assign, axis=0)
   868                                                   else:
   869         6       3427.0    571.2      0.0              n_unit_cells_per_pred = n_unit_cells // top_n
   870         6       3757.0    626.2      0.0              n_extra = n_unit_cells % top_n
   871         6      12925.0   2154.2      0.0              xnn_gen = np.zeros((n_unit_cells, self.unit_cell_length))
   872         6      26550.0   4425.0      0.0              hkl_assign = np.zeros((n_unit_cells, self.data_params['n_peaks']), dtype=int)
   873                                           
   874                                                       # If top_n == 5, then self.predict_xnn generates 5 unit cells
   875                                                       # xnn_pred: 1, top_n, unit_cell_length
   876         6  147308263.0 2.46e+07      9.0              xnn_pred, _ = self.predict_xnn(top_n, q2_obs=q2_obs[np.newaxis], batch_size=batch_size)
   877         6       6643.0   1107.2      0.0              xnn_pred = xnn_pred[0]
   878         6      10680.0   1780.0      0.0              xnn_gen[:top_n] = xnn_pred
   879         6    1396884.0 232814.0      0.1              q2_ref_calc = q2_calculator.get_q2(xnn_pred)
   880         6    7999719.0 1.33e+06      0.5              hkl_assign[:top_n] = fast_assign(q2_obs, q2_ref_calc)
   881                                           
   882                                                       # Resampling needs to generate n_unit_cells_per_pred - 1 unit cells from each prediction
   883                                                       # hkl_softmax: top_n, n_peaks, hkl_ref_length
   884        12 1161838774.0 9.68e+07     70.7              hkl_softmax = self.predict_hkl(
   885         6      70296.0  11716.0      0.0                  np.repeat(q2_obs[np.newaxis], repeats=top_n, axis=0),
   886         6       2574.0    429.0      0.0                  xnn_pred,
   887         6       2275.0    379.2      0.0                  batch_size=batch_size
   888                                                           )
   889         6       4258.0    709.7      0.0              start = top_n
   890        24      17213.0    717.2      0.0              for gen_index in range(n_unit_cells_per_pred - 1):
   891                                                           # This generates top_n unit cells per iteration
   892        18  269068028.0 1.49e+07     16.4                  hkl_assign[start: start + top_n], _ = vectorized_resampling(hkl_softmax, rng)
   893        18      38004.0   2111.3      0.0                  xnn_gen[start: start + top_n] = xnn_pred
   894        18       9105.0    505.8      0.0                  start += top_n
   895         6   35152982.0 5.86e+06      2.1              hkl_assign[start: start + n_extra], _ = vectorized_resampling(hkl_softmax[:n_extra], rng)
   896         6    1778036.0 296339.3      0.1              hkl = np.take_along_axis(self.hkl_ref[:, np.newaxis, :], hkl_assign[:, :, np.newaxis], axis=0)
   897                                           
   898                                                   # hkl: n_unit_cells, n_peaks, 3
   899        12      79184.0   6598.7      0.0          target_function = CandidateOptLoss(
   900         6      88740.0  14790.0      0.0              np.repeat(q2_obs[np.newaxis], repeats=n_unit_cells, axis=0), 
   901         6       4047.0    674.5      0.0              lattice_system=self.lattice_system,
   902                                                       )
   903         6    3295582.0 549263.7      0.2          target_function.update(hkl, xnn_gen)
   904         6   11708315.0 1.95e+06      0.7          xnn_gen += target_function.gauss_newton_step(xnn_gen)
   905         6     907484.0 151247.3      0.1          xnn_gen = fix_unphysical(xnn=xnn_gen, rng=self.rng, lattice_system=self.lattice_system)
   906        12    3204989.0 267082.4      0.2          unit_cell_gen = get_unit_cell_from_xnn(
   907         6       2956.0    492.7      0.0              xnn_gen, partial_unit_cell=True, lattice_system=self.lattice_system
   908                                                       )
   909         6       3615.0    602.5      0.0          return unit_cell_gen


%lprun -f IntegralFilter.predict_hkl optimizer.run(q2=q2_obs, n_top_candidates=n_top_candidates)
Total time: 1.16196 s
File: /global/cfs/cdirs/m4064/dwmoreau/MLI/mlindex/model_training/IntegralFilter.py
Function: IntegralFilter.predict_hkl at line 802

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   802                                               def predict_hkl(self, q2_obs, xnn, batch_size=None):
   803         6      24077.0   4012.8      0.0          q2_obs_scaled = q2_obs / self.q2_obs_scale
   804                                           
   805                                                   #print(f'\n Regression inferences for {self.split_group}')
   806         6       3236.0    539.3      0.0          if batch_size is None:
   807                                                       batch_size = self.model_params['batch_size']
   808                                           
   809                                                   # predict_on_batch helps with a memory leak...
   810         6       4470.0    745.0      0.0          N = q2_obs_scaled.shape[0]
   811         6    9050715.0 1.51e+06      0.8          hkl_softmax = np.zeros((N, self.data_params['n_peaks'], self.hkl_ref.shape[0]))
   812         6       5280.0    880.0      0.0          if self.model_params['mode'] == 'inference':
   813         6      27132.0   4522.0      0.0              q2_obs_scaled_f32 = q2_obs_scaled.astype(np.float32)
   814         6       9748.0   1624.7      0.0              xnn_f32 = xnn.astype(np.float32)
   815       906     473322.0    522.4      0.0              for pred_index in range(xnn.shape[0]):
   816       900     589265.0    654.7      0.1                  inputs = {
   817       900     915866.0   1017.6      0.1                      'input_0': q2_obs_scaled_f32[pred_index][np.newaxis],
   818       900     614673.0    683.0      0.1                      'input_1': xnn_f32[pred_index][np.newaxis]
   819                                                               }
   820       900 1150238477.0 1.28e+06     99.0                  hkl_softmax[pred_index] = self.calibration_onnx_model.run(None, inputs)[0]
   821                                                   elif self.model_params['mode'] == 'training':
   822                                                       n_batches = N // batch_size
   823                                                       left_over = N % batch_size
   824                                           
   825                                                       for batch_index in range(n_batches + 1):
   826                                                           start = batch_index * batch_size
   827                                                           if batch_index == n_batches:
   828                                                               batch_inputs = (
   829                                                                   np.zeros((batch_size, self.data_params['n_peaks'])),
   830                                                                   np.zeros((batch_size, self.unit_cell_length))
   831                                                                   )
   832                                                               batch_inputs[0][:left_over] = q2_obs_scaled[start: start + left_over]
   833                                                               batch_inputs[0][left_over:] = q2_obs_scaled[0]
   834                                                               batch_inputs[1][:left_over] = xnn[start: start + left_over]
   835                                                               batch_inputs[1][left_over:] = xnn[0]
   836                                                           else:
   837                                                               batch_inputs = (
   838                                                                   q2_obs_scaled[start: start + batch_size],
   839                                                                   xnn[start: start + batch_size]
   840                                                                   )
   841                                           
   842                                                           outputs = self.calibration_model.predict_on_batch(batch_inputs)
   843                                                           if batch_index == n_batches:
   844                                                               hkl_softmax[start:] = outputs[:left_over]
   845                                                           else:
   846                                                               hkl_softmax[start: start + batch_size] = outputs
   847         6       3246.0    541.0      0.0          return hkl_softmax


%lprun -f MITemplates.generate_xnn optimizer.run(q2=q2_obs, n_top_candidates=n_top_candidates)
Total time: 2.91135 s
File: /global/cfs/cdirs/m4064/dwmoreau/MLI/mlindex/model_training/MITemplates.py
Function: MITemplates.generate_xnn at line 429

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   429                                               def generate_xnn(self, q2_obs, indices=None):
   430         1       1332.0   1332.0      0.0          if indices is None:
   431         1   35310124.0 3.53e+07      1.2              hkl2 = get_hkl_matrix(self.hkl_ref[self.miller_index_templates], self.lattice_system)
   432         1      14317.0  14317.0      0.0              n_templates = self.template_params['n_templates']
   433                                                   else:
   434                                                       hkl2 = get_hkl_matrix(self.hkl_ref[self.miller_index_templates[indices]], self.lattice_system)
   435                                                       n_templates = indices.size
   436                                           
   437                                                   # q2_calc should increase monotonically. Sort hkl2 then re-solve for xnn iteratively.
   438         1       3306.0   3306.0      0.0          q2_obs_template = q2_obs[:self.template_params['n_peaks_template']]
   439         1       2454.0   2454.0      0.0          q2_obs_calibration = q2_obs[:self.template_params['n_peaks_calibration']]
   440         1     101028.0 101028.0      0.0          xnn = np.zeros((n_templates, self.unit_cell_length))
   441         1       2495.0   2495.0      0.0          sigma = q2_obs_template[np.newaxis]
   442         1      18174.0  18174.0      0.0          hessian_prefactor = (1 / sigma**2)[:, :, np.newaxis, np.newaxis]
   443         1   78466703.0 7.85e+07      2.7          term0 = np.matmul(hkl2[:, :, :, np.newaxis], hkl2[:, :, np.newaxis, :])
   444         1   91565877.0 9.16e+07      3.1          H = np.sum(hessian_prefactor * term0, axis=1)
   445         1   59580378.0 5.96e+07      2.0          good = np.linalg.matrix_rank(H, hermitian=True) == self.unit_cell_length
   446         1     637459.0 637459.0      0.0          xnn = xnn[good]
   447         1    2827296.0 2.83e+06      0.1          hkl2 = hkl2[good]
   448         6      11543.0   1923.8      0.0          for index in range(5):
   449         5   19731612.0 3.95e+06      0.7              q2_calc = (hkl2 @ xnn[:, :, np.newaxis])[:, :, 0]
   450         5       6814.0   1362.8      0.0              if index != 0:
   451         4   38319239.0 9.58e+06      1.3                  sort_indices = q2_calc.argsort(axis=1)
   452         4   16161131.0 4.04e+06      0.6                  q2_calc = np.take_along_axis(q2_calc, sort_indices, axis=1)
   453         4  127148463.0 3.18e+07      4.4                  hkl2 = np.take_along_axis(hkl2, sort_indices[:, :, np.newaxis], axis=1)
   454                                           
   455         5    9810448.0 1.96e+06      0.3              residuals = (q2_calc - q2_obs_template[np.newaxis]) / sigma
   456         5    3835106.0 767021.2      0.1              dlikelihood_dq2_pred = residuals / sigma
   457         5   97951602.0 1.96e+07      3.4              dloss_dxnn = np.sum(dlikelihood_dq2_pred[:, :, np.newaxis] * hkl2, axis=1)
   458         5  416656565.0 8.33e+07     14.3              term0 = np.matmul(hkl2[:, :, :, np.newaxis], hkl2[:, :, np.newaxis, :])
   459         5  420524869.0 8.41e+07     14.4              H = np.sum(hessian_prefactor * term0, axis=1)
   460         5  111719610.0 2.23e+07      3.8              delta_gn = -np.matmul(np.linalg.inv(H), dloss_dxnn[:, :, np.newaxis])[:, :, 0]
   461         5     485386.0  97077.2      0.0              xnn += delta_gn
   462         5   17224473.0 3.44e+06      0.6              xnn = fix_unphysical(xnn=xnn, rng=self.rng, lattice_system=self.lattice_system)
   463         1 1363232916.0 1.36e+09     46.8          return self._generate_xnn_common(q2_obs, xnn)

%lprun -f MITemplates._generate_xnn_common optimizer.run(q2=q2_obs, n_top_candidates=n_top_candidates)
Total time: 1.32669 s
File: /global/cfs/cdirs/m4064/dwmoreau/MLI/mlindex/model_training/MITemplates.py
Function: MITemplates._generate_xnn_common at line 486

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   486                                               def _generate_xnn_common(self, q2_obs, xnn):
   487         1       4278.0   4278.0      0.0          q2_obs_template = q2_obs[:self.template_params['n_peaks_template']]
   488         1        911.0    911.0      0.0          q2_obs_calibration = q2_obs[:self.template_params['n_peaks_calibration']]
   489                                                   # Now prepare each template for calibration, which does not involve the same
   490                                                   # number of peaks as the templates.
   491                                                   # First, find the best Miller index assignments using all calibration peaks
   492         2      67266.0  33633.0      0.0          q2_calculator = Q2Calculator(
   493         1       1863.0   1863.0      0.0              lattice_system=self.lattice_system,
   494         1       1583.0   1583.0      0.0              hkl=self.hkl_ref[:, :self.template_params['n_peaks_calibration']],
   495         1        611.0    611.0      0.0              tensorflow=False,
   496         1        551.0    551.0      0.0              representation='xnn'
   497                                                       )
   498         1  101404006.0 1.01e+08      7.6          q2_ref_calc = q2_calculator.get_q2(xnn)
   499         1  456662929.0 4.57e+08     34.4          hkl_assign_calibration = fast_assign(q2_obs_calibration, q2_ref_calc)
   500                                                   # Now remove templates that have non-unique Miller index assignments up to n_peaks_template
   501         2   62818431.0 3.14e+07      4.7          hkl_assign_template, unique_indices = np.unique(
   502         1       5911.0   5911.0      0.0              hkl_assign_calibration[:, :self.template_params['n_peaks_template']],
   503         1        541.0    541.0      0.0              axis=0, return_index=True
   504                                                       )
   505         1       1002.0   1002.0      0.0          n_templates = unique_indices.size
   506         1    1813094.0 1.81e+06      0.1          xnn = xnn[unique_indices]
   507         1    1406746.0 1.41e+06      0.1          hkl_assign_calibration = hkl_assign_calibration[unique_indices]
   508         2   10307365.0 5.15e+06      0.8          hkl_template = np.take(
   509         1       2083.0   2083.0      0.0              self.hkl_ref[:, :self.template_params['n_peaks_template']], hkl_assign_template, axis=0
   510                                                       )
   511         2   10492089.0 5.25e+06      0.8          hkl_calibration = np.take(
   512         1       2645.0   2645.0      0.0              self.hkl_ref[:, :self.template_params['n_peaks_calibration']], hkl_assign_calibration, axis=0
   513                                                       )
   514                                           
   515                                                   # Second, update the unit cell given the assignments up to n_template_peaks
   516         2      15660.0   7830.0      0.0          target_function = CandidateOptLoss(
   517         1     293969.0 293969.0      0.0              np.repeat(q2_obs_template[np.newaxis], n_templates, axis=0), 
   518         1        842.0    842.0      0.0              lattice_system=self.lattice_system,
   519                                                       )
   520         1   72856542.0 7.29e+07      5.5          target_function.update(hkl_template[:, :self.template_params['n_peaks_template']], xnn)
   521         1  248596873.0 2.49e+08     18.7          xnn += target_function.gauss_newton_step(xnn)
   522         1    3004638.0    3e+06      0.2          xnn = fix_unphysical(xnn=xnn, rng=self.rng, lattice_system=self.lattice_system)
   523         1   10905652.0 1.09e+07      0.8          hkl2 = get_hkl_matrix(hkl_calibration, self.lattice_system)
   524         1    3959709.0 3.96e+06      0.3          q2_calc = (hkl2 @ xnn[:, :, np.newaxis])[:, :, 0]
   525         1    1746009.0 1.75e+06      0.1          residuals = (q2_calc - q2_obs_calibration[np.newaxis]) / q2_obs_calibration[np.newaxis]
   526                                           
   527                                                   # Third, downsample to removes redundant unit cells
   528                                                   # Downsampling happens in chunks of xnn sorted by reciprocal space volume.
   529                                                   # If the chunk size is large, downsampling is extremely slow.
   530                                                   # If the chunk size is small, not enough redundant lattices get removed
   531                                                   # Running it twice with a small chunk size removes more lattices
   532                                                   # while also being reasonably fast.
   533         3       4920.0   1640.0      0.0          for _ in range(2):
   534         2  115716426.0 5.79e+07      8.7              xnn, q2_calc = self.downsample_candidates(xnn, q2_calc, residuals)
   535                                                   # Third, calculate the values needed for calibration
   536         3    1760295.0 586765.0      0.1          reciprocal_volume = get_unit_cell_volume(get_reciprocal_unit_cell_from_xnn(
   537         1        671.0    671.0      0.0              xnn, partial_unit_cell=True, lattice_system=self.lattice_system
   538         1        582.0    582.0      0.0              ), partial_unit_cell=True, lattice_system=self.lattice_system)
   539         1  123207726.0 1.23e+08      9.3          q2_ref_calc = q2_calculator.get_q2(xnn)
   540         1    2253555.0 2.25e+06      0.2          q2_calc_max = q2_calc.max(axis=1)
   541         1   68558922.0 6.86e+07      5.2          N_pred = np.count_nonzero(q2_ref_calc < q2_calc_max[:, np.newaxis], axis=1)
   542         2   28814162.0 1.44e+07      2.2          _, probability, _ = get_M20_likelihood(
   543         1       1182.0   1182.0      0.0              q2_obs=q2_obs_calibration,
   544         1        581.0    581.0      0.0              q2_calc=q2_calc,
   545         1       1022.0   1022.0      0.0              bravais_lattice=self.bravais_lattice,
   546         1        471.0    471.0      0.0              reciprocal_volume=reciprocal_volume
   547                                                       )
   548         1        681.0    681.0      0.0          return xnn, probability, N_pred, q2_calc_max


%lprun -f Candidates.random_subsampling optimizer.run(q2=q2_obs, n_top_candidates=n_top_candidates)
Total time: 7.33498 s
File: /global/cfs/cdirs/m4064/dwmoreau/MLI/mlindex/optimization/Optimizer.py
Function: Candidates.random_subsampling at line 149

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   149                                               def random_subsampling(self, iteration_info):
   150        60     112835.0   1880.6      0.0          n_keep = self.n_peaks - iteration_info['n_drop']
   151        60      59779.0    996.3      0.0          if iteration_info['uniform_sampling']:
   152                                                       subsampled_indices = self.rng.permuted(
   153                                                           np.repeat(np.arange(self.n_peaks)[np.newaxis], self.n, axis=0),
   154                                                           axis=1
   155                                                           )[:, :n_keep]
   156                                                   else:
   157        60     394597.0   6576.6      0.0              arg = 1 / self.q2_obs
   158        60    6008398.0 100140.0      0.1              p = np.repeat((arg / np.sum(arg))[np.newaxis], self.n, axis=0)
   159        60  421357588.0 7.02e+06      5.7              subsampled_indices = vectorized_subsampling(p, n_keep, self.rng)
   160       120   65883447.0 549028.7      0.9          hkl_subsampled = np.take_along_axis(
   161        60     124653.0   2077.6      0.0              self.hkl, subsampled_indices[:, :, np.newaxis], axis=1
   162                                                       )
   163        60    7294159.0 121569.3      0.1          q2_subsampled = np.take(self.q2_obs, subsampled_indices)
   164        60     713874.0  11897.9      0.0          target_function = CandidateOptLoss(q2_subsampled, lattice_system=self.lattice_system)
   165        60   97947034.0 1.63e+06      1.3          target_function.update(hkl_subsampled, self.xnn)
   166        60 6735083749.0 1.12e+08     91.8          self.iteration_worker_common(target_function)


%lprun -f Candidates.iteration_worker_common optimizer.run(q2=q2_obs, n_top_candidates=n_top_candidates)
Total time: 6.70474 s
File: /global/cfs/cdirs/m4064/dwmoreau/MLI/mlindex/optimization/Optimizer.py
Function: Candidates.iteration_worker_common at line 128

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   128                                               def iteration_worker_common(self, target_function):
   129        61  590092393.0 9.67e+06      8.8          self.xnn += target_function.gauss_newton_step(self.xnn)
   130        61  249365715.0 4.09e+06      3.7          self.fix_out_of_range_candidates()
   131        61 5847841822.0 9.59e+07     87.2          self.assign_hkls()
   132        61      97850.0   1604.1      0.0          if self.triplets is None:
   133        61     601853.0   9866.4      0.0              improved = self.M20 > self.best_M20
   134                                                   else:
   135                                                       improved = self.M_triplets.sum(axis=1) > self.best_M_triplets.sum(axis=1)
   136                                                       self.best_M_triplets[improved] = self.M_triplets[improved]
   137        61    1827835.0  29964.5      0.0          self.best_M20[improved] = self.M20[improved]
   138        61    5062854.0  82997.6      0.1          self.best_xnn[improved] = self.xnn[improved]
   139        61    9845825.0 161407.0      0.1          self.best_hkl[improved] = self.hkl[improved]


%lprun -f Candidates.assign_hkls optimizer.run(q2=q2_obs, n_top_candidates=n_top_candidates)
Total time: 5.57941 s
File: /global/cfs/cdirs/m4064/dwmoreau/MLI/mlindex/optimization/Optimizer.py
Function: Candidates.assign_hkls at line 111

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   111                                               def assign_hkls(self):
   112        62  810169727.0 1.31e+07     14.5          q2_ref_calc = self.q2_calculator.get_q2(self.xnn)
   113        62 3757223874.0 6.06e+07     67.3          hkl_assign = fast_assign(self.q2_obs, q2_ref_calc)
   114        62   33329070.0 537565.6      0.6          self.hkl = np.take(self.hkl_ref, hkl_assign, axis=0)
   115        62   79417459.0 1.28e+06      1.4          q2_calc = np.take_along_axis(q2_ref_calc, hkl_assign, axis=1)
   116        62      80987.0   1306.2      0.0          if not self.triplets is None:
   117                                                       self.M_triplets = get_M_triplet(
   118                                                           self.q2_obs,
   119                                                           q2_calc,
   120                                                           self.triplets,
   121                                                           self.hkl,
   122                                                           self.xnn,
   123                                                           self.lattice_system,
   124                                                           self.bravais_lattice
   125                                                           )
   126        62  899189007.0 1.45e+07     16.1          self.M20 = get_M20(self.q2_obs, q2_calc, q2_ref_calc)


%lprun -f get_M20 optimizer.run(q2=q2_obs, n_top_candidates=n_top_candidates)
Total time: 1.31805 s
File: /global/cfs/cdirs/m4064/dwmoreau/MLI/mlindex/utilities/FigureOfMerits.py
Function: get_M20 at line 16

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    16                                           def get_M20(q2_obs, q2_calc, q2_ref_calc):
    17       287   40193476.0 140047.0      3.0      discrepancy = np.mean(np.abs(q2_obs[np.newaxis] - q2_calc), axis=1)
    18       287  394757995.0 1.38e+06     30.0      smaller_ref_peaks = q2_ref_calc < q2_calc[:, -1][:, np.newaxis]
    19       287  472238952.0 1.65e+06     35.8      np.putmask(q2_ref_calc, ~smaller_ref_peaks, 0)
    20       287  184274188.0 642070.3     14.0      last_smaller_ref_peak = np.max(q2_ref_calc, axis=1)
    21       287  204618098.0 712955.0     15.5      N = np.sum(smaller_ref_peaks, axis=1)
    22                                           
    23                                               # There is an unknown issue that causes q2_calc to be all zero
    24                                               # These cases are caught and the M20 score is returned as zero.
    25                                               # Also catch cases where N == 0 for all peaks
    26       287   14895429.0  51900.4      1.1      good_indices = np.logical_and(q2_calc.sum(axis=1) != 0, N != 0)
    27       287     767553.0   2674.4      0.1      expected_discrepancy = np.zeros(q2_calc.shape[0])
    28       574    3516540.0   6126.4      0.3      expected_discrepancy[good_indices] = last_smaller_ref_peak[good_indices] / (
    29       287    1936219.0   6746.4      0.1          2 * N[good_indices]
    30                                               )
    31       287     688009.0   2397.2      0.1      M20 = expected_discrepancy / discrepancy
    32       287     164844.0    574.4      0.0      return M20


%lprun -f CandidateOptLoss.gauss_newton_step optimizer.run(q2=q2_obs, n_top_candidates=n_top_candidates)
Total time: 0.852749 s
File: /global/cfs/cdirs/m4064/dwmoreau/MLI/mlindex/optimization/CandidateOptLoss.py
Function: CandidateOptLoss.gauss_newton_step at line 91

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    91                                               def gauss_newton_step(self, xnn):
    92                                                   # q2_pred:       n_entries, n_peaks
    93                                                   # dq2_pred_dxnn: n_entries, n_peaks, xnn_length
    94                                                   # self.q2_obs:   n_peaks
    95        88   83673648.0 950836.9      9.8          q2_pred, dq2_pred_dxnn = self.get_q2_pred(xnn, jac=True)
    96        88    4808818.0  54645.7      0.6          residuals = (q2_pred - self.q2_obs) / self.sigma
    97        88    1802553.0  20483.6      0.2          dlikelihood_dq2_pred = residuals / self.sigma
    98        88   78786711.0 895303.5      9.2          dloss_dxnn = np.sum(dlikelihood_dq2_pred[:, :, np.newaxis] * dq2_pred_dxnn, axis=1)
    99        88  154341323.0 1.75e+06     18.1          term0 = np.matmul(dq2_pred_dxnn[:, :, :, np.newaxis], dq2_pred_dxnn[:, :, np.newaxis, :])
   100        88  189360760.0 2.15e+06     22.2          H = np.sum(self.hessian_prefactor * term0, axis=1)
   101                                                   # Need to ensure H is invertible before inverting.
   102        88  102268762.0 1.16e+06     12.0          invertible = np.linalg.det(H) != 0 # This is the fastest
   103                                                   #invertible = np.linalg.matrix_rank(H, hermitian=True) == self.uc_length
   104                                                   #invertible = np.isfinite(np.linalg.cond(H)) # This is slow
   105        88     819921.0   9317.3      0.1          delta_gn = np.zeros((self.n_entries, self.uc_length))
   106        88      44593.0    506.7      0.0          try:
   107       264   34960971.0 132427.9      4.1              delta_gn[invertible] = -np.matmul(
   108        88  195388064.0 2.22e+06     22.9                  np.linalg.inv(H[invertible]),
   109        88    6363655.0  72314.3      0.7                  dloss_dxnn[invertible, :, np.newaxis]
   110        88      65948.0    749.4      0.0                  )[:, :, 0]
   111                                                   except np.linalg.LinAlgError as e:
   112                                                       print(f'GAUSS-NEWTON INVERSION FAILED: {e}')
   113        88      63448.0    721.0      0.0          return delta_gn


%lprun -f OptimizerManager.redistribute_xnn optimizer.run(q2=q2_obs, n_top_candidates=n_top_candidates)
Total time: 0.26935 s
File: /global/cfs/cdirs/m4064/dwmoreau/MLI/mlindex/optimization/Optimizer.py
Function: OptimizerManager.redistribute_xnn at line 884


%lprun -f Candidates.correct_off_by_two optimizer.run(q2=q2_obs, n_top_candidates=n_top_candidates)

%lprun -f Candidates.assign_extinction_group optimizer.run(q2=q2_obs, n_top_candidates=n_top_candidates)
