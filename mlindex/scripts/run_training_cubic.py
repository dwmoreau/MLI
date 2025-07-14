import os
os.environ["KERAS_BACKEND"] = "torch"

from mlindex.model_training.Wrapper import Wrapper


if __name__ == '__main__':
    broadening_tag = '1'
    data_params = {
        'tag': f'cubic_{broadening_tag}',
        'base_directory': '/global/cfs/cdirs/m4064/dwmoreau/MLI/',
        'groupspec_file_name': 'GroupSpec_cubic.xlsx',
        'groupspec_sheet': 'groups_v0',
        'load_from_tag': False,
        'augment': True,
        'hkl_ref_length': 100,
        'n_peaks': 10,
        'lattice_system': 'cubic',
        'n_max_group': 100000,
        'broadening_tag': broadening_tag,
        }

    aug_params = {
        'tag': f'cubic_{broadening_tag}',
        'max_augmentation': 25,
        'median_augmentation': 10,
        'augment_method': 'random',
        'augment_shift': 0.2,
        'n_per_volume': None,
        }

    template_group_params = {
        'tag': f'cubic_{broadening_tag}',
        'load_from_tag': False,
        'templates_per_dominant_zone_bin': 2000,
        'calibrate': True,
        'parallelization': 'multiprocessing',
        'n_processes': 2,
        'max_depth': 16,
        'min_samples_leaf': 8,
        'max_leaf_nodes': 400,
        'l2_regularization': 0,
        'n_entries_train': 5000,
        'n_instances_train': 10000000,
        'n_peaks_template': 10,
        'n_peaks_calibration': 10,
        'roc_file_name': '/Users/DWMoreau/MLI/figures/data/radius_of_convergence_drop7_iter100_sampQ2_!!.npy',
        'grid_search': None,
        #'grid_search':
        #    {
        #    'max_leaf_nodes': [200, 400, 800, 1600],
        #    'max_depth': [8, 16],
        #    'min_samples_leaf': [8, 16, 32],
        #    },
        }

    template_params = {
        'cP': template_group_params,
        'cI': template_group_params,
        'cF': template_group_params,
        }

    rf_group_params = {
        'tag': f'cubic_{broadening_tag}',
        'load_from_tag': False,
        'n_estimators': 50,
        'min_samples_leaf': 2,
        'max_depth': 15,
        'subsample': 0.75,
        }

    rf_group_params_load = {
        'tag': f'cubic_{broadening_tag}',
        'load_from_tag': True,
        }
    rf_params = {
        'cP_0': rf_group_params,
        'cI_0': rf_group_params,
        'cF_0': rf_group_params,
        }
    
    integral_filter_group_params = {
        'tag': f'cubic_{broadening_tag}',
        'load_from_tag': False,
        'peak_length': 10,
        'extraction_peak_length': 6,
        'filter_length': 3,
        'n_volumes': 150,
        'n_filters': 400,
        'initial_layers': [400, 200, 100],
        'final_layers': [600, 300, 100, 50],
        'l1_regularization': 0.00001,
        'base_line_layers': [600, 300, 100, 50],
        'base_line_dropout_rate': 0.0,
        'learning_rate': 0.0001,
        'epochs': 30,
        'batch_size': 64,
        'loss_type': 'log_cosh',
        'augment': True,
        'model_type': 'metric',
        'calibration_params': {
            'layers': 3,
            'l1_regularization': 0.0,
            'epsilon_pds': 0.1,
            'epochs': 20,
            'learning_rate': 0.0002,
            'augment': True,
            'batch_size': 64,
            },
        }

    integral_filter_group_params_load = {
        'tag': f'cubic_{broadening_tag}',
        'load_from_tag': True,
        }
    integral_filter_params = {
        'cP_0': integral_filter_group_params,
        'cI_0': integral_filter_group_params,
        'cF_0': integral_filter_group_params,
        }

    random_params_bl = {
        'tag': f'cubic_{broadening_tag}',
        'load_from_tag': False,
        'grid_search': {
            'n_estimators': [50],
            'min_samples_leaf': [2],
            'max_depth': [7],
            'subsample': [0.75],
            }
        }
    random_params = {
        'cF': random_params_bl,
        'cI': random_params_bl,
        'cP': random_params_bl,
        }

    indexer = Wrapper(
        aug_params=aug_params, 
        data_params=data_params,
        rf_params=rf_params,
        template_params=template_params,
        integral_filter_params=integral_filter_params,
        random_params=random_params,
        seed=12345,
        )
    if data_params['load_from_tag']:
        indexer.load_data_from_tag(load_augmented=True, load_train=True, load_bravais_lattice='all')
    else:
        indexer.load_data()
    indexer.setup_random()
    #indexer.setup_miller_index_templates()
    #indexer.setup_pitf('training')
    #indexer.setup_regression('training')
    #indexer.inferences_regression()
    #indexer.evaluate_regression()
