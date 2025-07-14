import os
os.environ["KERAS_BACKEND"] = "torch"
from Indexing import Indexing


if __name__ == '__main__':
    broadening_tag = '1'
    data_params = {
        'tag': f'triclinic_{broadening_tag}',
        'base_directory': '/Users/DWMoreau/MLI',
        'groupspec_file_name': 'GroupSpec_triclinic.xlsx',
        'groupspec_sheet': 'groups_v0',
        'load_from_tag': True,
        'augment': True,
        'hkl_ref_length': 500,
        'n_peaks': 20,
        'lattice_system': 'triclinic',
        'n_max_group': 1000000,
        'broadening_tag': broadening_tag,
        }

    aug_params = {
        'tag': f'triclinic_{broadening_tag}',
        'max_augmentation': 10,
        'augment_method': 'pca',
        'augment_shift': 0.2,
        'n_per_volume': 1000,
        }

    template_group_params = {
        'tag': f'triclinic_{broadening_tag}',
        'load_from_tag': False,
        'templates_per_dominant_zone_bin': 2000,
        'calibrate': True,
        'parallelization': 'multiprocessing',
        'n_processes': 4,
        'max_depth': 12,
        'max_leaf_nodes': 1000,
        'min_samples_leaf': 10,
        'n_peaks_template': 20,
        'n_peaks_calibration': 20,
        'n_entries_train': 2000,
        'n_instances_train': 1000000,
        'roc_file_name': '/Users/DWMoreau/MLI/figures/data/radius_of_convergence_drop12_iter100_sampQ2_!!.npy',
        'grid_search': None,
        #'grid_search':
        #    {
        #    'max_leaf_nodes': [1000, 1500, 2000],
        #    'max_depth': [20, 30, 40],
        #    'min_samples_leaf': [8, 16],
        #    },
        }

    template_params = {
        'aP': template_group_params,
        }

    reg_group_params = {
        'tag': f'triclinic_{broadening_tag}',
        'load_from_tag': False,
        'n_estimators': 100,
        'min_samples_leaf': 8,
        'max_depth': 10,
        'subsample': 0.75,
        'n_dominant_zone_bins': 5,
        }

    reg_params = {
        'aP_00': reg_group_params,
        }

    pitf_group_params = {
        'tag': f'triclinic_{broadening_tag}',
        'load_from_tag': False,
        'peak_length': 20,
        'extraction_peak_length': 8,
        'filter_length': 3,
        'n_volumes': 100,
        'n_filters': 400,
        'n_volumes_depth': [256,  64,  16],
        'n_filters_depth': [200, 200, 200],
        'initial_layers': [400, 200, 100],
        'final_layers': [1000, 600, 300, 100, 50],
        'l1_regularization': 0.00005,
        'base_line_layers': [1000, 600, 300, 100, 50],
        'base_line_dropout_rate': 0.05,
        'learning_rate': 0.0001,
        'epochs': 20,
        'batch_size': 64,
        'loss_type': 'log_cosh',
        'augment': True,
        'model_type': 'metric',
        'sigma': 0.03,
        'calibration_params': {
            'layers': 3,
            'epsilon_pds': 0.1,
            'epochs': 20,
            'learning_rate': 0.0002,
            'augment': True,
            'batch_size': 64,
            },
        }
    pitf_params = {
        'aP_00': pitf_group_params,
        }

    random_params_bl = {
        'tag': f'triclinic_{broadening_tag}',
        'load_from_tag': False,
        'grid_search': {
            'n_estimators': [500],
            'min_samples_leaf': [5],
            'max_depth': [10],
            'subsample': [0.75],
            }
        }
    random_params = {
        'aP': random_params_bl,
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
    #indexer.setup_random()
    #indexer.setup_pitf('training')
    #indexer.setup_miller_index_templates()
    indexer.setup_regression('training')
    indexer.inferences_regression()
    indexer.evaluate_regression()
