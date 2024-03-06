import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import tensorflow as tf

from Networks import hkl_model_builder_mlp
from Utilities import PairwiseDifferenceCalculator
from Utilities import read_params
from Utilities import write_params


class Assigner:
    def __init__(self, data_params, model_params, hkl_ref, uc_scaler, angle_scale, q2_scaler, save_to):
        self.model_params = model_params
        self.model_params['n_uc_params'] = len(data_params['y_indices'])
        self.model_params['n_points'] = data_params['n_points']
        self.model_params['hkl_ref_length'] = data_params['hkl_ref_length']
        model_params_defaults = {
            'n_filters': 4,
            'kernel_size': [5, 20],
            'layers': [200, 200, 200, 100],
            'epsilon': 0.001,
            'dropout_rate': 0.25,
            'output_activation': 'softmax',
            'epochs': 10,
            'learning_rate': 0.002,
            'batch_size': 256,
            'epsilon_pds': 0.01,
            'perturb_std': None,
            }
        for key in model_params_defaults.keys():
            if key not in self.model_params.keys():
                self.model_params[key] = model_params_defaults[key]
        self.save_to = save_to
        self.pairwise_difference_calculation = PairwiseDifferenceCalculator(
            lattice_system=data_params['lattice_system'],
            hkl_ref=hkl_ref,
            tensorflow=True,
            q2_scaler=q2_scaler,
            uc_scaler=uc_scaler,
            angle_scale=angle_scale,
            )

    def save(self):
        write_params(self.model_params, f'{self.save_to}/assignment_params_{self.model_params["tag"]}.csv')
        self.model.save_weights(f'{self.save_to}/assignment_weights_{self.model_params["tag"]}.h5')

    def load_from_tag(self, tag, mode):
        params = read_params(f'{self.save_to}/assignment_params_{self.model_params["tag"]}.csv')
        params_keys = [
            'tag',
            'train_on',
            'batch_size',
            'epochs',
            'layers',
            'learning_rate',
            'epsilon',
            'dropout_rate',
            'output_activation',
            'kernel_size',
            'n_filters',
            'hkl_ref_length',
            'n_points',
            'n_uc_params',
            ]
        self.model_params = dict.fromkeys(params_keys)
        self.model_params['tag'] = params['tag']
        self.model_params['train_on'] = params['train_on']
        self.model_params['n_points'] = int(params['n_points'])
        if params['perturb_std'] == '':
            self.model_params['perturb_std'] = None
        else:
            self.model_params['perturb_std'] = float(params['perturb_std'])
        self.model_params['epsilon_pds'] = float(params['epsilon_pds'])
        self.model_params['n_uc_params'] = int(params['n_uc_params'])
        self.model_params['n_filters'] = int(params['n_filters'])
        self.model_params['batch_size'] = int(params['batch_size'])
        self.model_params['epochs'] = int(params['epochs'])
        self.model_params['hkl_ref_length'] = int(params['hkl_ref_length'])
        self.model_params['layers'] = np.array(
            params['layers'].split('[')[1].split(']')[0].split(','),
            dtype=int
            )
        self.model_params['kernel_size'] = np.array(
            params['kernel_size'].split('[')[1].split(']')[0].split(','),
            dtype=int
            )
        self.model_params['learning_rate'] = float(params['learning_rate'])
        self.model_params['epsilon'] = float(params['epsilon'])
        self.model_params['dropout_rate'] = float(params['dropout_rate'])
        self.model_params['output_activation'] = params['output_activation']

        self.build_model(mode=mode)
        self.model.load_weights(
            filepath=f'{self.save_to}/assignment_weights_{self.model_params["tag"]}.h5',
            by_name=True
            )
        self.compile_model()

    def reload_model(self):
        self.build_model()
        self.model.load_weights(
            filepath=f'{self.save_to}/assignment_weights_{self.model_params["tag"]}.h5',
            by_name=True
            )

    def get_initial_assign_biases(self, data):
        def bias_init_tf_jac(b, f):
            f_est = np.exp(b) / np.sum(np.exp(b))

            term0 = np.zeros((b.size, f.size))
            term0[np.arange(b.size), np.arange(b.size)] = f_est
            term1 = f_est[:, np.newaxis] * f_est[np.newaxis]
            df_est_db = term0 - term1

            loss = 1/2 * np.sum((f - f_est)**2)
            dloss_db = -np.sum((f - f_est) * df_est_db, axis=1)
            return loss, dloss_db

        # bias initialization
        # batch_size x 10 x 100
        hkl_labels = np.stack(data['hkl_labels'])  # n_data x n_points
        frequencies = np.zeros((self.model_params['n_points'], self.model_params['hkl_ref_length']))
        bins = np.arange(0, self.model_params['hkl_ref_length'] + 1) - 0.5
        for index in range(self.model_params['n_points']):
            frequencies[index], _ = np.histogram(hkl_labels[:, index], bins=bins, density=True)

        bias_init = np.zeros((self.model_params['n_points'], self.model_params['hkl_ref_length']))
        for index in range(self.model_params['n_points']):
            results = scipy.optimize.minimize(
                bias_init_tf_jac,
                x0=frequencies[index],
                args=(frequencies[index]),
                method='L-BFGS-B',
                jac=True
                )
            bias_init[index] = results.x

        #fig, axes = plt.subplots(1, 1, figsize=(10, 4))
        #axes.plot(results.x)
        #plt.show()
        #print(results)
        return bias_init

    def build_model(self, mode, data=None):
        inputs = {
            'unit_cell_scaled': tf.keras.Input(
                shape=(self.model_params['n_uc_params']),
                name='unit_cell_scaled',
                dtype=tf.float32,
                ),
            'q2_scaled': tf.keras.Input(
                shape=(self.model_params['n_points']),
                name='q2_scaled',
                dtype=tf.float32,
                ),
            }
        self.model = tf.keras.Model(inputs, self.model_builder(inputs, mode))
        #self.model.summary()
        """
        # This sets the biases to the initial distribution.
        # This helps with very early training but benefits aren't worth the time spent in this loop.
        if not data is None:
            bias_init = self.get_initial_assign_biases(data[data['train']])
            for index in range(self.model_params['n_points']):
                weights, biases = self.model.get_layer(f'hkl_softmaxes_{index}').get_weights()
                self.model.get_layer(f'hkl_softmaxes_{index}').set_weights([weights, bias_init[index]])
        """

    def model_builder(self, inputs, mode):
        if self.model_params['perturb_std'] is None:
            print(f'Building assignment without perturbations {self.model_params["tag"]}')
            unit_cell_scaled = inputs['unit_cell_scaled']
        elif mode != 'training':
            print(f'Building assignment without perturbations {self.model_params["tag"]}')
            unit_cell_scaled = inputs['unit_cell_scaled']
        else:
            print(f'Building assignment with perturbations {self.model_params["tag"]}')
            unit_cell_scaled = tf.keras.layers.GaussianNoise(self.model_params['perturb_std'])(
                inputs['unit_cell_scaled'], training=True
                )
        pairwise_differences_scaled = self.pairwise_difference_calculation.get_pairwise_differences_from_uc_scaled(
            unit_cell_scaled, inputs['q2_scaled']
            )
        pds_inv = self.transform_pairwise_differences(pairwise_differences_scaled, tensorflow=True)
        hkl_softmaxes = hkl_model_builder_mlp(pds_inv, 'softmaxes', self.model_params)
        return hkl_softmaxes

    def compile_model(self):
        optimizer = tf.optimizers.legacy.Adam(self.model_params['learning_rate'])
        loss_functions = {'hkl_softmaxes': tf.keras.losses.SparseCategoricalCrossentropy()}
        loss_metrics = {'hkl_softmaxes': 'accuracy'}
        loss_weights = {'hkl_softmaxes': 1}
        self.model.compile(
            optimizer=optimizer, 
            loss=loss_functions,
            loss_weights=loss_weights,
            metrics=loss_metrics,
            )

    def transform_pairwise_differences(self, pairwise_differences_scaled, tensorflow):
        if tensorflow:
            abs_func = tf.math.abs
        else:
            abs_func = np.abs
        epsilon = self.model_params['epsilon_pds']
        return epsilon / (abs_func(pairwise_differences_scaled) + epsilon)

    def fit_model(self, data, unit_cell_scaled_key, y_indices):
        self.build_model(mode='training', data=data)
        self.compile_model()

        train_data = data[data['train']]
        val_data = data[~data['train']]

        if y_indices is None:
            unit_cell_scaled = np.stack(data[unit_cell_scaled_key])
        else:
            unit_cell_scaled = np.stack(data[unit_cell_scaled_key])[:, y_indices]

        train_inputs = {
            'unit_cell_scaled': unit_cell_scaled[data['train']],
            'q2_scaled': np.stack(train_data['q2_scaled']),
            }
        val_inputs = {
            'unit_cell_scaled': unit_cell_scaled[~data['train']],
            'q2_scaled': np.stack(val_data['q2_scaled']),
            }
        train_true = {'hkl_softmaxes': np.stack(train_data['hkl_labels'])}
        val_true = {'hkl_softmaxes': np.stack(val_data['hkl_labels'])}
        print(f'\n Starting assign model training: {self.model_params["tag"]} {self.model_params["perturb_std"]}')
        self.fit_history = self.model.fit(
            x=train_inputs,
            y=train_true,
            epochs=self.model_params['epochs'],
            shuffle=True,
            batch_size=self.model_params['batch_size'], 
            validation_data=(val_inputs, val_true),
            callbacks=None,
            )
        self.save()

        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
        axes[0].plot(
            np.arange(self.model_params['epochs']),
            self.fit_history.history['loss'],
            marker='.', label='Training', color=colors[0],
            )
        axes[0].plot(
            np.arange(self.model_params['epochs']),
            self.fit_history.history['val_loss'],
            marker='v', label='Validation', color=colors[1],
            )
        axes[1].plot(
            np.arange(self.model_params['epochs']),
            self.fit_history.history['accuracy'],
            marker='.', label='Training', color=colors[0],
            )
        axes[1].plot(
            np.arange(self.model_params['epochs']),
            self.fit_history.history['val_accuracy'],
            marker='v', label='Validation', color=colors[1],
            )
        axes[0].set_title('HKL Assignment')
        axes[0].set_ylabel('Loss')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        for col in range(2):
            axes[col].set_xlabel('Epoch')
        fig.tight_layout()
        fig.savefig(f'{self.save_to}/assignment_training_loss_{self.model_params["tag"]}.png')
        plt.close()

    def do_predictions(self, data, unit_cell_scaled_key, y_indices, reload_model=True, batch_size=256):
        if reload_model:
            self.build_model(mode='evaluate')
            self.model.load_weights(
                filepath=f'{self.save_to}/assignment_weights_{self.model_params["tag"]}.h5',
                by_name=True
                )
            self.compile_model()
        if y_indices is None:
            inputs = {
                'unit_cell_scaled': np.stack(data[unit_cell_scaled_key]),
                'q2_scaled': np.stack(data['q2_scaled'])
                }
        else:
            inputs = {
                'unit_cell_scaled': np.stack(data[unit_cell_scaled_key])[:, y_indices],
                'q2_scaled': np.stack(data['q2_scaled'])
                }
        print(f'\n Inference for Miller Index Assignments')
        n_batchs = len(data) // batch_size
        left_over = len(data) % batch_size

        softmaxes = np.zeros((len(data), self.model_params['n_points'], self.model_params['hkl_ref_length']))
        for batch_index in range(n_batchs + 1):
            start = batch_index * batch_size
            if batch_index == n_batchs:
                batch_unit_cell_scaled = np.zeros((batch_size, self.model_params['n_uc_params']))
                batch_q2_scaled = np.zeros((batch_size, self.model_params['n_points']))
                batch_unit_cell_scaled[:left_over] = inputs['unit_cell_scaled'][start: start + left_over]
                batch_unit_cell_scaled[left_over:] = inputs['unit_cell_scaled'][0]
                batch_q2_scaled[:left_over] = inputs['q2_scaled'][start: start + left_over]
                batch_q2_scaled[left_over:] = inputs['q2_scaled'][0]
            else:
                end = (batch_index + 1) * batch_size
                batch_unit_cell_scaled = inputs['unit_cell_scaled'][start: end]
                batch_q2_scaled = inputs['q2_scaled'][start: end]
            batch_inputs = {
                'unit_cell_scaled': batch_unit_cell_scaled,
                'q2_scaled': batch_q2_scaled
                }
            softmaxes_batch = self.model.predict_on_batch(batch_inputs)
            if batch_index == n_batchs:
                softmaxes[start: start + left_over] = softmaxes_batch[:left_over]
            else:
                softmaxes[start: end] = softmaxes_batch
        return softmaxes

    def evaluate(self, data, bravais_lattices, unit_cell_scaled_key, y_indices, perturb_std):
        for bravais_lattice in bravais_lattices:
            if bravais_lattice == 'All':
                bl_data = data
            else:
                bl_data = data[data['bravais_lattice'] == bravais_lattice]
            if y_indices is None:
                unit_cell_scaled = np.stack(bl_data[unit_cell_scaled_key])
            else:
                unit_cell_scaled = np.stack(bl_data[unit_cell_scaled_key])[:, y_indices]

            if perturb_std is not None:
                noise = np.random.normal(loc=0, scale=perturb_std, size=unit_cell_scaled.shape)
                unit_cell_scaled += noise
            pairwise_differences_scaled = self.pairwise_difference_calculation.get_pairwise_differences_from_uc_scaled(
                unit_cell_scaled, np.stack(bl_data['q2_scaled'])
                )
            bl_pds_inv = self.transform_pairwise_differences(pairwise_differences_scaled, tensorflow=False)
            labels_closest = np.argmax(bl_pds_inv, axis=2)
            labels_true = np.stack(bl_data['hkl_labels'])
            labels_true_train = np.stack(bl_data[bl_data['train']]['hkl_labels'])
            labels_pred_train = np.stack(bl_data[bl_data['train']]['hkl_labels_pred'])
            labels_true_val = np.stack(bl_data[~bl_data['train']]['hkl_labels'])
            labels_pred_val = np.stack(bl_data[~bl_data['train']]['hkl_labels_pred'])

            # correct shape: n_entries, n_peaks
            correct_pred_train = labels_true_train == labels_pred_train
            correct_pred_val = labels_true_val == labels_pred_val
            correct_closest = labels_true == labels_closest
            accuracy_pred_train = correct_pred_train.sum() / correct_pred_train.size
            accuracy_pred_val = correct_pred_val.sum() / correct_pred_val.size
            accuracy_closest = correct_closest.sum() / correct_closest.size
            # accuracy for each entry
            accuracy_entry_train = correct_pred_train.sum(axis=1) / self.model_params['n_points']
            accuracy_entry_val = correct_pred_val.sum(axis=1) / self.model_params['n_points']
            accuracy_entry_closest = correct_closest.sum(axis=1) / self.model_params['n_points']
            # accuracy per peak position
            accuracy_peak_position_train = correct_pred_train.sum(axis=0) / correct_pred_train.shape[0]
            accuracy_peak_position_val = correct_pred_val.sum(axis=0) / correct_pred_val.shape[0]
            accuracy_peak_position_closest = correct_closest.sum(axis=0) / bl_data.shape[0]

            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            bins = (np.arange(self.model_params['n_points'] + 2) - 0.5) / self.model_params['n_points']
            centers = (bins[1:] + bins[:-1]) / 2
            dbin = bins[1] - bins[0]
            hist_train, _ = np.histogram(accuracy_entry_train, bins=bins, density=True)
            hist_val, _ = np.histogram(accuracy_entry_val, bins=bins, density=True)
            hist_closest, _ = np.histogram(accuracy_entry_closest, bins=bins, density=True)
            axes[0].bar(centers, hist_train, width=dbin, label='Predicted: Training')
            axes[0].bar(centers, hist_val, width=dbin, alpha=0.5, label='Predicted: Validation')
            axes[0].bar(centers, hist_closest, width=dbin, alpha=0.5, label='Closest')
            axes[1].bar(
                np.arange(self.model_params['n_points']), accuracy_peak_position_train,
                width=1, label='Predicted: Training'
                )
            axes[1].bar(
                np.arange(self.model_params['n_points']), accuracy_peak_position_val,
                width=1, alpha=0.5, label='Predicted: Validation'
                )
            axes[1].bar(
                np.arange(self.model_params['n_points']), accuracy_peak_position_closest,
                width=1, alpha=0.5, label='Closest'
                )

            axes[1].legend(frameon=False)
            axes[0].set_title(f'Predicted accuracy: {accuracy_pred_train:0.3f}/{accuracy_pred_val:0.3f}\nClosest accuracy: {accuracy_closest:0.3f}')

            axes[0].set_xlabel('Accuracy')
            axes[1].set_xlabel('Peak Position')
            axes[0].set_ylabel('Entry Accuracy')
            axes[1].set_ylabel('Peak Accuracy')
            axes[1].set_ylim([0, 1])
            fig.tight_layout()
            fig.savefig(f'{self.save_to}/{bravais_lattice}_assignment_{self.model_params["tag"]}.png')
            plt.close()    

    def calibrate(self, data):
        def calibration_plots(softmaxes, n_points, n_bins=25):
            N = softmaxes.shape[0]
            y_pred = softmaxes.argmax(axis=2)
            p_pred = np.zeros((N, n_points))
            metrics = np.zeros((n_bins, 4))
            ece = 0
            for entry_index in range(N):
                for point_index in range(n_points):
                    p_pred[entry_index, point_index] = softmaxes[
                        entry_index,
                        point_index,
                        y_pred[entry_index, point_index]
                        ]

            bins = np.linspace(p_pred.min(), p_pred.max(), n_bins + 1)
            centers = (bins[1:] + bins[:-1]) / 2
            metrics[:, 0] = centers
            for bin_index in range(n_bins):
                indices = np.logical_and(
                    p_pred >= bins[bin_index],
                    p_pred < bins[bin_index + 1],
                    )
                if np.sum(indices) > 0:
                    p_pred_bin = p_pred[indices]
                    y_pred_bin = y_pred[indices]
                    y_true_bin = y_true[indices]
                    metrics[bin_index, 1] = np.sum(y_pred_bin == y_true_bin) / y_true_bin.size
                    metrics[bin_index, 2] = p_pred_bin.mean()
                    metrics[bin_index, 3] = p_pred_bin.std()
                    prefactor = indices.sum() / indices.size
                    ece += prefactor * np.abs(metrics[bin_index, 2] - metrics[bin_index, 1])
            return metrics, ece

        softmaxes = np.stack(data['hkl_softmaxes'])
        y_true = np.stack(data['hkl_labels'])

        metrics, ece = calibration_plots(softmaxes, self.model_params['n_points'])
        fig, axes = plt.subplots(1, 1, figsize=(5, 3))
        axes.plot([0, 1], [0, 1], linestyle='dotted', color=[0, 0, 0])
        axes.set_xlabel('Confidence')
        axes.errorbar(metrics[:, 2], metrics[:, 1], yerr=metrics[:, 3], marker='.')

        axes.set_ylabel('Accuracy')
        axes.set_title(f'Unscaled\nExpected Confidence Error: {ece:0.4f}')
        axes.set_title(f'Expected Confidence Error: {ece:0.4f}')
        fig.tight_layout()
        fig.savefig(f'{self.save_to}/assignment_calibration_{self.model_params["tag"]}.png')
        plt.close()
