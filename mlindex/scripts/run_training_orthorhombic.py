import os
os.environ["KERAS_BACKEND"] = "torch"
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
import keras
from mlindex.model_training.Wrapper import Wrapper


if __name__ == '__main__':
    broadening_tag = '1'
    data_params = {
        'tag': f'orthorhombic_{broadening_tag}',
        'base_directory': '/global/cfs/cdirs/m4064/dwmoreau/MLI/',
        'groupspec_file_name': 'GroupSpec_orthorhombic.xlsx',
        'groupspec_sheet': 'Groups_V6',
        'load_from_tag': True,
        'augment': True,
        'hkl_ref_length': 750,
        'n_peaks': 20,
        'lattice_system': 'orthorhombic',
        'n_max_group': 1000000,
        'broadening_tag': broadening_tag
        }

    aug_params = {
        'tag': f'orthorhombic_{broadening_tag}',
        'max_augmentation': 25,
        'median_augmentation': 10,
        'augment_method': 'pca',
        'augment_shift': 0.2,
        'n_per_volume': 200,
        }

    template_group_params = {
        'tag': f'orthorhombic_{broadening_tag}',
        'load_from_tag': False,
        'templates_per_dominant_zone_bin': 2000,
        'parallelization': 'multiprocessing',
        'n_processes': 120,
        'max_depth': 20,
        'min_samples_leaf': 100,
        'max_leaf_nodes': 2000,
        'l2_regularization': 0,
        'n_entries_train': 10000,
        'n_instances_train': 100000000,
        'n_peaks_template': 20,
        'n_peaks_calibration': 20,
        'max_distance': 0.03,
        'roc_file_name': '/global/cfs/cdirs/m4064/dwmoreau/MLI/mlindex/characterization/roc/data/!!_roc_peaks20_drop16_iter100_sampQ2.npy',
        'grid_search': None,
        'load_training_data': True,
        }
    template_group_params_load = {
        'tag': f'orthorhombic_{broadening_tag}',
        'load_from_tag': True,
    }

    template_params = {
        'oC': template_group_params_load,
        'oF': template_group_params_load,
        'oI': template_group_params_load,
        'oP': template_group_params,
        }

    rf_group_params_load = {
        'tag': f'orthorhombic_{broadening_tag}',
        'load_from_tag': True,
        }
    rf_group_params_small_load = rf_group_params_load

    rf_group_params = {
        'tag': f'orthorhombic_{broadening_tag}',
        'load_from_tag': False,
        'n_estimators': 50,
        'min_samples_leaf': 60,
        'max_depth': 10,
        'subsample': 0.5,
        'n_dominant_zone_bins': 10,
        'n_jobs': 50,
        }

    rf_params = {
        'oC_0_00': rf_group_params,
        'oC_1_00': rf_group_params,
        'oC_2_00': rf_group_params,
        'oF_0_00': rf_group_params,
        'oF_0_01': rf_group_params,
        'oI_0_00': rf_group_params,
        'oP_0_00': rf_group_params,
        'oP_0_01': rf_group_params,
        'oP_0_02': rf_group_params,
        'oP_0_03': rf_group_params,
        }

    integral_filter_group_params = {
        'tag': f'orthorhombic_{broadening_tag}',
        'load_from_tag': False,
        'peak_length': 20,
        'extraction_peak_length': 10,
        'n_volumes': 100,
        'n_filters': 1000,
        'd_model': 512,
        'n_heads': 8,
        'layers': [1000, 600, 300, 100, 50],
        'l1_regularization': 0.00005,
        'base_line_layers': [1000, 600, 300, 100, 50],
        'base_line_dropout_rate': 0.0,
        'learning_rate': 0.0001,
        'epochs': 30,
        'batch_size': 64,
        'loss_type': 'log_cosh',
        'augment': True,
        'model_type': 'metric',
        'calibration_params': {
            'layers': 3,
            'epsilon_pds': 0.1,
            'epochs': 40,
            'learning_rate': 0.0002,
            'augment': True,
            'batch_size': 64,
            'n_heads': 5,
            },
        }


    integral_filter_group_params_load = {
        'tag': f'orthorhombic_{broadening_tag}',
        'load_from_tag': True,
        }

    integral_filter_params = {
        'oC_0_00': integral_filter_group_params,
        'oC_1_00': integral_filter_group_params,
        'oC_2_00': integral_filter_group_params,
        'oF_0_00': integral_filter_group_params,
        'oF_0_01': integral_filter_group_params,
        'oI_0_00': integral_filter_group_params,
        'oP_0_00': integral_filter_group_params,
        'oP_0_01': integral_filter_group_params,
        'oP_0_02': integral_filter_group_params,
        'oP_0_03': integral_filter_group_params,
        }

    random_params_bl0 = {
        'tag': f'orthorhombic_{broadening_tag}',
        'load_from_tag': False,
        'grid_search': None,
        'n_estimators': 100,
        'min_samples_leaf': 10,
        'max_depth': 10,
        'subsample': 0.5,
        }
    random_params_bl1 = {
        'tag': f'orthorhombic_{broadening_tag}',
        'load_from_tag': False,
        'grid_search': None,
        'n_estimators': 100,
        'min_samples_leaf': 10,
        'max_depth': 16,
        'subsample': 0.5,
        }
    random_params_bl_load = {
        'tag': f'orthorhombic_{broadening_tag}',
        'load_from_tag': True,
    }

    random_params = {
        'oC': random_params_bl0,
        'oF': random_params_bl0,
        'oI': random_params_bl0,
        'oP': random_params_bl1,
        }

    wrapper = Wrapper(
        aug_params=aug_params, 
        data_params=data_params,
        rf_params=rf_params, 
        template_params=template_params,
        integral_filter_params=integral_filter_params,
        random_params=random_params,
        seed=12345, 
        )
    if data_params['load_from_tag']:
        wrapper.load_data_from_tag(load_augmented=True, load_train=True)
    else:
        wrapper.load_data()
    #wrapper.setup_random()
    #wrapper.setup_miller_index_templates()
    #wrapper.setup_random_forest()
    #wrapper.inferences_random_forest()
    #wrapper.evaluate_random_forest()
    wrapper.setup_integral_filter('training')
