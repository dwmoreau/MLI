import os
os.environ["KERAS_BACKEND"] = "torch"
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
#import keras

from Indexing import Indexing


if __name__ == '__main__':
    broadening_tag = '1.5'
    data_params = {
        'tag': f'tetragonal_{broadening_tag}',
        'base_directory': '/Users/DWMoreau/MLI',
        'groupspec_file_name': 'GroupSpec_tetragonal.xlsx',
        'groupspec_sheet': 'groups_v3',
        'load_from_tag': True,
        'augment': True,
        'hkl_ref_length': 750,
        'n_peaks': 20,
        'lattice_system': 'tetragonal',
        'n_max_group': 1000000,
        'broadening_tag': broadening_tag,
        }

    aug_params = {
        'tag': f'tetragonal_{broadening_tag}',
        'max_augmentation': 25,
        'augment_method': 'pca',
        'augment_shift': 0.2,
        'n_per_volume': 200,
        }

    template_group_params = {
        'tag': f'tetragonal_{broadening_tag}',
        'load_from_tag': False,
        'templates_per_dominant_zone_bin': 2000,
        'parallelization': 'multiprocessing',
        'n_processes': 4,
        'max_depth': 16,
        'min_samples_leaf': 10,
        'max_leaf_nodes': 1000,
        'l2_regularization': 0,
        'n_entries_train': 2000,
        'n_instances_train': 10000000,
        'n_peaks_template': 20,
        'n_peaks_calibration': 20,
        'roc_file_name': '/Users/DWMoreau/MLI/figures/data/radius_of_convergence_drop17_iter100_sampQ2_!!.npy',
        'grid_search': None,
        #'grid_search':
        #    {
        #    'max_leaf_nodes': [1000, 1500, 2000],
        #    'max_depth': [20, 30, 40],
        #    'min_samples_leaf': [8, 16],
        #    },
        }

    template_params = {
        'tP': template_group_params,
        'tI': template_group_params,
        }

    reg_group_params = {
        'tag': f'tetragonal_{broadening_tag}',
        'load_from_tag': False,
        'n_estimators': 100,
        'min_samples_leaf': 8,
        'max_depth': 16,
        'subsample': 0.75,
        }

    reg_params = {
        'tP_0_00': reg_group_params,
        'tP_1_00': reg_group_params,
        'tP_0_01': reg_group_params,
        'tP_1_01': reg_group_params,
        'tI_0_00': reg_group_params,
        'tI_1_00': reg_group_params,
        'tI_0_01': reg_group_params,
        'tI_1_01': reg_group_params,
        }

    pitf_group_params = {
        'tag': f'tetragonal_{broadening_tag}',
        'load_from_tag': False,
        'peak_length': 20,
        'extraction_peak_length': 6,
        'filter_length': 3,
        'n_volumes': 100,
        'n_filters': 400,
        'initial_layers': [400, 200, 100],
        'final_layers': [1000, 600, 300, 100, 50],
        'l1_regularization': 0.00002,
        'base_line_layers': [1000, 600, 300, 100, 50],
        'base_line_dropout_rate': 0.0,
        'learning_rate': 0.0001,
        'epochs': 30,
        'batch_size': 64,
        'loss_type': 'log_cosh',
        'augment': True,
        'model_type': 'metric',
        'sigma': 0.03,
        'calibration_params': {
            'layers': 3,
            'dropout_rate': 0.1,
            'epsilon_pds': 0.1,
            'epochs': 20,
            'learning_rate': 0.0002,
            'augment': True,
            'batch_size': 64,
            },
        }

    pitf_group_params_load = {
        'tag': f'tetragonal_{broadening_tag}',
        'load_from_tag': True,
        }

    pitf_params = {
        'tP_0_00': pitf_group_params_load,
        'tP_1_00': pitf_group_params,
        'tP_0_01': pitf_group_params,
        'tP_1_01': pitf_group_params,
        'tI_0_00': pitf_group_params,
        'tI_1_00': pitf_group_params,
        'tI_0_01': pitf_group_params,
        'tI_1_01': pitf_group_params,
        }
    random_params_bl = {
        'tag': f'tetragonal_{broadening_tag}',
        'load_from_tag': False,
        'grid_search': {
            'n_estimators': [200],
            'min_samples_leaf': [8],
            'max_depth': [8],
            'subsample': [0.75],
            }
        }
    random_params = {
        'tI': random_params_bl,
        'tP': random_params_bl,
        }

    indexer = Indexing(
        aug_params=aug_params, 
        data_params=data_params,
        reg_params=reg_params, 
        template_params=template_params,
        pitf_params=pitf_params,
        random_params=random_params,
        seed=12345, 
        )
    if data_params['load_from_tag']:
        indexer.load_data_from_tag(load_augmented=True, load_train=True)
    else:
        indexer.load_data()
    indexer.setup_pitf('training')
    indexer.setup_random()
    indexer.setup_miller_index_templates()
    indexer.setup_regression()
    indexer.inferences_regression()
    indexer.evaluate_regression()
