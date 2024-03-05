import copy
import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf

from Networks import mlp_model_builder


class Classifier:
    def __init__(self, data_params, model_params, save_to):
        self.save_to = save_to
        self.model_params = model_params
        if self.model_params['load_from_tag']:
            self.load_from_tag(self.model_params['tag'])
        else:
            self.model_params['n_points'] = data_params['n_points']
            self.model_params['bravais_lattices'] = data_params['bravais_lattices']
            self.model_params['kernel_initializer'] = None
            self.model_params['bias_initializer'] = None
            model_params_defaults = {
                'batch_size': 64,
                'dropout_rate': 0.0,
                'epochs': 20,
                'epsilon': 0.001,
                'layers': [60, 60, 60, 60],
                'learning_rate': 0.0003,
                'output_activation': 'softmax',
                'output_name': 'bl_softmax',
                }
            for key in model_params_defaults.keys():
                if key not in self.model_params.keys():
                    self.model_params[key] = model_params_defaults[key]

    def save(self):
        model_params = copy.deepcopy(self.model_params)
        model_params.pop('kernel_initializer')
        model_params.pop('bias_initializer')
        with open(f'{self.save_to}/class_params_{self.model_params["tag"]}.csv', 'w') as output_file:
            writer = csv.DictWriter(output_file, fieldnames=model_params.keys())
            writer.writeheader()
            writer.writerow(model_params)
        self.model.save_weights(f'{self.save_to}/class_weights_{self.model_params["tag"]}.h5')

    def load_from_tag(self, tag):
        with open(f'{self.save_to}/class_params_{self.model_params["tag"]}.csv', 'r') as params_file:
            reader = csv.DictReader(params_file)
            for row in reader:
                params = row
        params_keys = [
            'tag',
            'n_points',
            'batch_size',
            'epochs',
            'n_outputs',
            'layers',
            'learning_rate',
            'epsilon',
            'dropout_rate',
            'bravais_lattices',
            'output_activation',
            'kernel_initializer',
            'bias_initializer',
            ]
        self.model_params = dict.fromkeys(params_keys)
        self.model_params['tag'] = params['tag']
        self.model_params['n_points'] = int(params['n_points'])
        self.model_params['batch_size'] = int(params['batch_size'])
        self.model_params['epochs'] = int(params['epochs'])
        self.model_params['n_outputs'] = int(params['n_outputs'])
        self.model_params['layers'] = np.array(
            params['layers'].split('[')[1].split(']')[0].split(','),
            dtype=int
            )
        self.model_params['learning_rate'] = float(params['learning_rate'])
        self.model_params['epsilon'] = float(params['epsilon'])
        self.model_params['dropout_rate'] = float(params['dropout_rate'])
        self.model_params['bravais_lattices'] = params['bravais_lattices'].split('[')[1].split(']')[0].split(',')
        self.model_params['output_activation'] = params['output_activation']
        self.model_params['kernel_initializer'] = None
        self.model_params['bias_initializer'] = None

        self.build_model()
        self.model.load_weights(
            filepath=f'{self.save_to}/class_weights_{self.model_params["tag"]}.h5',
            by_name=True
            )
        self.compile_model()

    def do_classification(self, data):
        self.build_model()
        self.compile_model()
        self.fit_model(data)
        self.save()

    def get_initial_bl_class_biases(self, data):
        # bias initialization
        n = len(self.model_params['bravais_lattices'])
        bl_labels = np.array(data[data['train']]['bravais_lattice_label']) # n_data x n_points
        bins = np.arange(0, n + 1) - 0.5
        frequencies, _ = np.histogram(bl_labels, bins=bins, density=True)

        bias_init = np.zeros(n)
        def bias_init_tf(b, f):
            f_est = np.exp(b) / np.sum(np.exp(b))
            return np.sum((f - f_est)**2)
        results = minimize(
            bias_init_tf,
            x0=np.zeros(n),
            args=frequencies,
            method='BFGS'
            )
        bias_init = results.x
        #fig, axes = plt.subplots(2, 1, figsize=(6, 5))
        #axes[0].plot(bias_init)
        #axes[1].plot(np.exp(bias_init) / np.sum(np.exp(bias_init)))
        #plt.show()
        #print(results)
        return bias_init

    def build_model(self):
        inputs = {
            'q2_scaled': tf.keras.Input(
                shape=self.model_params['n_points'],
                name='input_points',
                dtype=tf.float32,
                ),
            }
        self.model = tf.keras.Model(inputs, self.model_builder(inputs))

    def model_builder(self, inputs):
        bl_softmax = mlp_model_builder(
            inputs['q2_scaled'],
            tag='bl_class',
            model_params=self.model_params,
            output_name='bl_softmax'
            )
        return bl_softmax

    def compile_model(self):
        optimizer = tf.optimizers.legacy.Adam(self.model_params['learning_rate'])
        loss_functions = {'bl_softmax': tf.keras.losses.SparseCategoricalCrossentropy()}
        loss_metrics = {'bl_softmax': 'accuracy'}
        self.model.compile(
            optimizer=optimizer, 
            loss=loss_functions,
            metrics=loss_metrics
            )

    def fit_model(self, data, initialize_biases=True):
        train = data[data['train']]
        val = data[~data['train']]
        train_inputs = {'q2_scaled': np.stack(train['q2_scaled'])}
        val_inputs = {'q2_scaled': np.stack(val['q2_scaled'])}
        train_true = {'bl_softmax': np.stack(train['bravais_lattice_label'])}
        val_true = {'bl_softmax': np.stack(val['bravais_lattice_label'])}

        if initialize_biases:
            bias_init = self.get_initial_bl_class_biases(train)
            weights, biases = self.model.get_layer('bl_softmax').get_weights()
            self.model.get_layer('bl_softmax').set_weights([weights, bias_init])

        self.fit_history = self.model.fit(
            x=train_inputs,
            y=train_true,
            epochs=self.model_params['epochs'],
            shuffle=True,
            batch_size=self.model_params['batch_size'], 
            validation_data=(val_inputs, val_true),
            callbacks=None,
            sample_weight=None,
            )

        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].plot(
            self.fit_history.history['loss'],
            marker='.', label='Training', color=colors[0],
            )
        axes[1].plot(
            self.fit_history.history['accuracy'],
            marker='.', label='Training', color=colors[0],
            )
        axes[0].plot(
            self.fit_history.history['val_loss'],
            marker='v', label='Validation', color=colors[1],
            )
        axes[1].plot(
            self.fit_history.history['val_accuracy'],
            marker='v', label='Validation', color=colors[1],
            )
        axes[0].legend(frameon=False)
        axes[0].set_ylabel('BL Class loss')
        axes[1].set_ylabel('BL Class accuracy')
        fig.tight_layout()
        fig.savefig(f'{self.save_to}/bl_class_training_loss_{self.model_params["tag"]}.png')
        plt.close()

    def do_predictions(self, data):
        inputs = {'q2_scaled': np.stack(data['q2_scaled'])}
        softmaxes = self.model.predict(inputs, batch_size=2048)
        return softmaxes

    def evaluate(self, data):
        def class_calibration_check(softmaxes, y_true, n_bins=25):
            N = y_true.shape[0]
            y_pred = softmaxes.argmax(axis=1)
            p_pred = np.zeros(N)
            metrics = np.zeros((n_bins, 4))
            ece = 0
            for entry_index in range(N):
                p_pred[entry_index] = softmaxes[entry_index, y_pred[entry_index]]

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

        print('\nPredicting Bravais Lattice classification for calibration check')
        data = data[~data['augmented']]
        softmaxes = self.do_predictions(data)
        bravais_lattice_label_pred = np.argmax(softmaxes, axis=1)

        train = data[data['train']]
        val = data[~data['train']]
        fig, axes = plt.subplots(1, 3, figsize=(10, 5), sharex=True, sharey=True)
        cm_all = confusion_matrix(
            y_true=data['bravais_lattice_label'], 
            y_pred=bravais_lattice_label_pred,
            normalize='true',
            )
        cm_train = confusion_matrix(
            y_true=train['bravais_lattice_label'], 
            y_pred=bravais_lattice_label_pred[data['train']],
            normalize='true',
            )
        cm_val = confusion_matrix(
            y_true=val['bravais_lattice_label'], 
            y_pred=bravais_lattice_label_pred[~data['train']],
            normalize='true',
            )
        disp_all = ConfusionMatrixDisplay(
            confusion_matrix=cm_all,
            display_labels=self.model_params['bravais_lattices'],
            )
        disp_train = ConfusionMatrixDisplay(
            confusion_matrix=cm_train,
            display_labels=self.model_params['bravais_lattices'],
            )
        disp_val = ConfusionMatrixDisplay(
            confusion_matrix=cm_val,
            display_labels=self.model_params['bravais_lattices'],
            )

        disp_all.plot(ax=axes[0], xticks_rotation='vertical', colorbar=False, values_format='0.2f')
        disp_train.plot(ax=axes[1], xticks_rotation='vertical', colorbar=False, values_format='0.2f')
        disp_val.plot(ax=axes[2], xticks_rotation='vertical', colorbar=False, values_format='0.2f')
        axes[0].set_title('All Data')
        axes[1].set_title('Training Data')
        axes[2].set_title('Validation Data')
        fig.tight_layout()
        fig.savefig(f'{self.save_to}/bl_confusion_matrix_{self.model_params["tag"]}.png')
        plt.close()

        metrics_train, ece_train = class_calibration_check(
            softmaxes[data['train']],
            train['bravais_lattice_label'],
            )
        metrics_val, ece_val = class_calibration_check(
            softmaxes[~data['train']],
            val['bravais_lattice_label'],
            )
        fig, axes = plt.subplots(1, 1, figsize=(5, 3))
        axes.plot([0, 1], [0, 1], linestyle='dotted', color=[0, 0, 0])
        axes.set_xlabel('Confidence')
        axes.errorbar(
            metrics_train[:, 2], metrics_train[:, 1], yerr=metrics_train[:, 3],
            marker='.', label='Training'
            )
        axes.errorbar(
            metrics_val[:, 2], metrics_val[:, 1], yerr=metrics_val[:, 3],
            marker='.', label='Validation'
            )
        axes.legend()
        axes.set_ylabel('Accuracy')
        axes.set_title(f'Expected Confidence Error (Train/Val): {ece_train:0.4f}/{ece_val:0.4f}')
        fig.tight_layout()
        fig.savefig(f'{self.save_to}/class_probability_calibration_{self.model_params["tag"]}.png')
        plt.close()
