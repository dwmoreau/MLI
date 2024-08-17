"""
New approach:
    x start with xnn ratios & hkl patterns (value & link) and calculate q2_key
        x start with 1 link
        - generalize to random ratio and hkl patterns
    x Get a q2_query, calculate a volume for each ratio & hkl pattern by scaling q2_query & q2_key
        - does accuracy of volume estimate correlate with qk_distance - yes
    x Volume scale q2_query for each ratio & hkl_pattern
    - Calculate distance between q2_query_scaled and q2_keys
        qk_distance = np.linalg.norm(
            1 - q2_queries_scaled[:, np.newaxis, :key_length] / q2_keys[np.newaxis, :, :key_length],
            axis=2
            )**2
    - Convert distance to a probability with softmax
"""
import h5py
import joblib
import matplotlib.pyplot as plt
import numpy as np
import os
# This supresses the tensorflow message on import
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

from Networks import mlp_model_builder
from TargetFunctions import IndexingTargetFunction
from TargetFunctions import LikelihoodLoss
from Utilities import fix_unphysical
from Utilities import get_hkl_matrix
from Utilities import get_unit_cell_from_xnn
from Utilities import get_unit_cell_volume
from Utilities import get_xnn_from_reciprocal_unit_cell
from Utilities import PairwiseDifferenceCalculator
from Utilities import read_params
from Utilities import Q2Calculator
from Utilities import vectorized_resampling
from Utilities import write_params


class CosineAttention(tf.keras.layers.Layer):
    def __init__(self, xnn_ratio_values, hkl_links, key_length, ratio_start, lattice_system, source, **kwargs):
        super().__init__(**kwargs)
        self.lattice_system = lattice_system
        self.key_length = key_length
        self.ratio_start = ratio_start

        if source is None:
            self.n_keys = hkl_links.shape[0]
            self.n_peaks = hkl_links.shape[1]
            similarity_weights = tf.random.normal(
                shape=(self.key_length,), mean=1.0, stddev=0.01, seed=None, dtype=tf.float32
                )
            volume_weights = tf.random.normal(
                shape=(self.n_peaks - self.ratio_start,), mean=1.0, stddev=0.01, seed=None, dtype=tf.float32
                )
            ratio_scale = np.median(xnn_ratio_values)
        else:
            with h5py.File(source, 'r') as h5_file:
                hkl_links = np.array(h5_file['cosine_attention']['hkl_links:0'])
                self.n_keys = hkl_links.shape[0]
                self.n_peaks = hkl_links.shape[1]
                xnn_ratio_values = np.array(h5_file['cosine_attention']['xnn_ratio_values:0'])
                similarity_weights = tf.cast(
                    np.array(h5_file['cosine_attention']['similarity_weights:0']),
                    dtype=tf.float32
                    )
                volume_weights = tf.cast(
                    np.array(h5_file['cosine_attention']['volume_weights:0']),
                    dtype=tf.float32
                    )
                ratio_scale = np.array(h5_file['cosine_attention']['ratio_scale:0'])            

        self.hkl_links = tf.Variable(
            tf.cast(hkl_links, dtype=tf.float32),
            trainable=False,
            name='hkl_links'
            )
        self.xnn_ratio_values = tf.Variable(
            initial_value=tf.cast(xnn_ratio_values, dtype=tf.float32),
            trainable=False,
            name='xnn_ratio_values'
            )
        self.similarity_weights = tf.Variable(
            similarity_weights,
            trainable=False,
            name='similarity_weights'
            )
        self.volume_weights = tf.Variable(
            volume_weights,
            trainable=True,
            name='volume_weights'
            )
        self.ratio_scale = tf.Variable(
            tf.cast(ratio_scale, dtype=tf.float32),
            trainable=False,
            name='ratio_scale'
            )

        self.q2_calculator = Q2Calculator(
            lattice_system=self.lattice_system,
            hkl=hkl_links,
            tensorflow=True,
            representation='xnn'
            )

    def validate_linkage(self, q2_keys):
        # This is purely for checking consistency between values, keys, and links.
        q2_keys = tf.Variable(
            tf.cast(q2_keys[:, :self.key_length], dtype=tf.float32),
            trainable=False,
            dtype=tf.float32,
            shape=(self.n_keys, self.key_length),
            name='keys'
            )
        xnn_values = self.get_xnn_from_ratio(
            self.xnn_ratio_values, lattice_system=self.lattice_system, tensorflow=True
            )

        q2_check = self.q2_calculator.get_q2(xnn_values)[:, :self.key_length]
        if not tf.math.reduce_all(tf.experimental.numpy.isclose(q2_check, q2_keys)):
            print('KEYS AND VALUES ARE NOT PROPERLY LINKED!!!!!!!!!!!!!')
            for i in range(10):
                print(q2_check[i])
                print(q2_keys[i])
                print(xnn_values[i])
                print()

    def link_keys_and_values(self):
        if self.lattice_system == 'tetragonal':
            denominator = self.xnn_ratio_values[:, 0]
        elif self.lattice_system == 'hexagonal':
            denominator = self.xnn_ratio_values[:, 0] * tf.math.sin(np.pi/3)
        else:
            assert False
        xll = 1 / denominator
        xhh = tf.math.sqrt(xll) * self.xnn_ratio_values[:, 0]
        xnn_values = tf.stack((xhh, xll), axis=1)
        return self.q2_calculator.get_q2(xnn_values)

    def call(self, q2_queries, **kwargs):
        q2_keys = self.link_keys_and_values()
        # q2_queries:        batch_size, n_peaks
        # q2_keys:           n_keys,     n_peaks
        # q2_ratio:          batch_size, n_keys, n_peaks
        # reciprocal_volume: batch_size, n_keys
        q2_ratio = (
            q2_queries[:, tf.newaxis, self.ratio_start:] / q2_keys[tf.newaxis, :, self.ratio_start:]
            )**(3/2)
        volume_weights = tf.nn.softmax(self.volume_weights)
        reciprocal_volume = tf.math.reduce_sum(
            volume_weights[tf.newaxis, tf.newaxis] * q2_ratio,
            axis=2,
            name='reciprocal_volume'
            )
        # q2_queries:        batch_size, n_peaks
        # reciprocal_volume: batch_size, n_keys
        # q2_queries_scaled: batch_size, n_keys, n_peaks
        # q2_keys:           n_keys, n_peaks
        q2_queries_scaled = q2_queries[:, tf.newaxis] / reciprocal_volume[:, :, tf.newaxis]**(2/3)

        similarity_weights = tf.nn.softmax(self.similarity_weights)
        query_key_distance = tf.norm(
            similarity_weights * (1 - q2_queries_scaled[:, :, :self.key_length] / q2_keys[tf.newaxis, :, :self.key_length]),
            axis=2
            ) / 0.1

        return query_key_distance, reciprocal_volume

    def ratio_target_function(self, ratio_true, weights):
        # ratio_true: (batch_size)
        # weights:    (batch_size, n_keys)
        difference = ((ratio_true - self.xnn_ratio_values[tf.newaxis, :, 0]) / self.ratio_scale)**2
        return tf.reduce_sum(difference * weights, axis=1)

    def reciprocal_volume_target_function(self, reciprocal_volume_scaled_true, predictions):
        # reciprocal_volume_true: batch_size
        # weights:                batch_size, n_keys
        # reciprocal_volume_pred: batch_size, n_keys
        weights = predictions[:, :, 0]
        reciprocal_volume_scaled_pred = predictions[:, :, 1]
        difference = (reciprocal_volume_scaled_true[:, tf.newaxis] - reciprocal_volume_scaled_pred)**2
        return tf.reduce_sum(difference * weights, axis=1)

    @staticmethod
    def get_ratio_from_xnn(xnn, lattice_system, tensorflow):
        if tensorflow:
            if lattice_system in ['hexagonal', 'tetragonal']:
                return (xnn[:, 0] / tf.math.sqrt(xnn[:, 1]))[:, tf.newaxis]
        else:
            if lattice_system in ['hexagonal', 'tetragonal']:
                return (xnn[:, 0] / np.sqrt(xnn[:, 1]))[:, np.newaxis]

    @staticmethod
    def get_xnn_from_ratio(ratio, lattice_system, tensorflow):
        if tensorflow:
            if lattice_system == 'tetragonal':
                denominator = ratio[:, 0]
            elif lattice_system == 'hexagonal':
                denominator = ratio[:, 0] * tf.math.sin(np.pi/3)
            else:
                assert False
            xll = 1 / denominator
            xhh = tf.math.sqrt(xll) * ratio[:, 0]
            return tf.stack((xhh, xll), axis=1)
        else:
            if lattice_system == 'tetragonal':
                denominator = ratio[:, 0]
            elif lattice_system == 'hexagonal':
                denominator = ratio[:, 0] * np.sin(np.pi/3)
            else:
                assert False
            xll = 1 / denominator
            xhh = np.sqrt(xll) * ratio[:, 0]
            return np.stack((xhh, xll), axis=1)


class PhysicsInformedModel:
    def __init__(self, split_group, data_params, model_params, save_to, seed, q2_scaler, xnn_scaler, hkl_ref):
        self.split_group = split_group
        self.data_params = data_params
        self.model_params = model_params

        self.n_peaks = data_params['n_peaks']
        self.unit_cell_length = data_params['unit_cell_length']
        self.unit_cell_indices = data_params['unit_cell_indices']
        self.save_to = save_to
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)

        self.lattice_system = self.data_params['lattice_system']
        self.q2_scaler = q2_scaler
        self.xnn_scaler = xnn_scaler
        self.hkl_ref = hkl_ref

    def setup(self, data):
        model_params_defaults = {
            'key_length': 6,
            'ratio_start': 0,
            'layers': 2,
            'dropout_rate': 0.05,
            'learning_rate': 0.002,
            'epochs': 100,
            'batch_size': 64,
            }

        for key in model_params_defaults.keys():
            if key not in self.model_params.keys():
                self.model_params[key] = model_params_defaults[key]
        self.key_length = self.model_params['key_length']
        self.build_model(data=data)

    def save(self):
        write_params(self.model_params, f'{self.save_to}/{self.split_group}_pitf_params_{self.model_params["tag"]}.csv')
        self.model.save_weights(f'{self.save_to}/{self.split_group}_pitf_weights_{self.model_params["tag"]}.h5')
        joblib.dump(
            self.volume_scaler,
            f'{self.save_to}/{self.split_group}_pitf_volume_scaler_{self.model_params["tag"]}.bin'
            )

    def load_from_tag(self):
        params = read_params(f'{self.save_to}/{self.split_group}_pitf_params_{self.model_params["tag"]}.csv')
        params_keys = [
            'tag',
            'n_keys',
            'key_length',
            'ratio_start',
            'layers',
            'dropout_rate',
            'learning_rate',
            'epochs',
            'batch_size',
            ]
        self.model_params = dict.fromkeys(params_keys)
        self.model_params['tag'] = params['tag']
        self.model_params['n_keys'] = int(params['n_keys'])
        self.model_params['key_length'] = int(params['key_length'])
        self.model_params['ratio_start'] = int(params['ratio_start'])
        self.model_params['layers'] = int(params['layers'])
        self.model_params['dropout_rate'] = float(params['dropout_rate'])
        self.model_params['learning_rate'] = float(params['learning_rate'])
        self.model_params['epochs'] = int(params['epochs'])
        self.model_params['batch_size'] = int(params['batch_size'])
        self.key_length = self.model_params['key_length']

        self.volume_scaler = joblib.load(f'{self.save_to}/{self.split_group}_pitf_volume_scaler_{self.model_params["tag"]}.bin')

        self.build_model(data=None)
        self.compile_model()
        self.model.load_weights(
            filepath=f'{self.save_to}/{self.split_group}_pitf_weights_{self.model_params["tag"]}.h5',
            by_name=True
            )
        #print(self.AttentionLayer.ratio_scale)
        #print(self.AttentionLayer.similarity_weights)
        #print(self.AttentionLayer.volume_weights)
        #print(self.AttentionLayer.xnn_ratio_values.numpy().sum())
        #print(self.AttentionLayer.hkl_links.numpy().sum())

    def train(self, data):
        train = data[data['train']]
        val = data[~data['train']]

        q2_queries = np.stack(train['q2'])[:, :self.key_length]
        q2_keys = self.AttentionLayer.link_keys_and_values().numpy()
        similarity = np.matmul(q2_queries, q2_keys[:, :self.key_length].T)
        mag_keys = np.linalg.norm(q2_keys[:, :self.key_length], axis=1)
        mag_queries = np.linalg.norm(q2_queries, axis=1)
        similarity /= (mag_queries[:, np.newaxis] * mag_keys[np.newaxis])
        most_similar = similarity.max(axis=1)
        train_indices = most_similar < 0.99999999

        train_inputs = {'q2_queries': np.stack(train['q2'])[train_indices]}
        val_inputs = {'q2_queries': np.stack(val['q2'])}

        train_xnn = np.stack(train['reindexed_xnn'])[train_indices][:, self.data_params['unit_cell_indices']]
        train_ratio = CosineAttention.get_ratio_from_xnn(
            train_xnn, lattice_system=self.lattice_system, tensorflow=False
            )
        val_xnn = np.stack(val['reindexed_xnn'])[:, self.data_params['unit_cell_indices']]
        val_ratio = CosineAttention.get_ratio_from_xnn(
            val_xnn, lattice_system=self.lattice_system, tensorflow=False
            )

        train_reciprocal_unit_cell = np.stack(
            train['reciprocal_reindexed_unit_cell']
            )[train_indices][:, self.unit_cell_indices]
        train_reciprocal_unit_cell_volume = get_unit_cell_volume(
            train_reciprocal_unit_cell,
            partial_unit_cell=True,
            lattice_system=self.data_params['lattice_system']
            )
        train_reciprocal_unit_cell_volume_scaled = \
            (train_reciprocal_unit_cell_volume - self.volume_scaler.mean_[0]) / self.volume_scaler.scale_[0]

        val_reciprocal_unit_cell = np.stack(
            val['reciprocal_reindexed_unit_cell']
            )[:, self.unit_cell_indices]
        val_reciprocal_unit_cell_volume = get_unit_cell_volume(
            val_reciprocal_unit_cell,
            partial_unit_cell=True,
            lattice_system=self.data_params['lattice_system']
            )
        val_reciprocal_unit_cell_volume_scaled = \
            (val_reciprocal_unit_cell_volume - self.volume_scaler.mean_[0]) / self.volume_scaler.scale_[0]

        train_true = {
            'weights': train_ratio,
            'weights__reciprocal_volume_scaled': train_reciprocal_unit_cell_volume_scaled,
            }
        val_true = {
            'weights': val_ratio,
            'weights__reciprocal_volume_scaled': val_reciprocal_unit_cell_volume_scaled,
            }

        print(self.AttentionLayer.similarity_weights)
        print(self.AttentionLayer.volume_weights)
        print(self.AttentionLayer.xnn_ratio_values.numpy().sum())
        self.fit_history = self.model.fit(
            x=train_inputs,
            y=train_true,
            epochs=self.model_params['epochs'],
            shuffle=True,
            batch_size=self.model_params['batch_size'], 
            validation_data=(val_inputs, val_true),
            callbacks=None,
            )
        #print(self.AttentionLayer.ratio_scale)
        print(self.AttentionLayer.similarity_weights)
        print(self.AttentionLayer.volume_weights)
        print(self.AttentionLayer.xnn_ratio_values.numpy().sum())
        #print(self.AttentionLayer.hkl_links.numpy().sum())
        self.save()

        fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
        axes[0].plot(
            self.fit_history.history['weights_loss'], 
            label='Training', marker='.'
            )
        axes[0].plot(
            self.fit_history.history['val_weights_loss'], 
            label='Validation', marker='v'
            )
        axes[1].plot(
            self.fit_history.history['weights__reciprocal_volume_scaled_loss'], 
            label='Training', marker='.'
            )
        axes[1].plot(
            self.fit_history.history['val_weights__reciprocal_volume_scaled_loss'], 
            label='Validation', marker='v'
            )
        axes[0].set_ylabel('Ratio Loss')
        axes[1].set_ylabel('Volume Loss')
        axes[1].set_xlabel('Epoch')
        axes[0].legend()
        fig.tight_layout()
        fig.savefig(f'{self.save_to}/{self.split_group}_pitf_training_loss_{self.model_params["tag"]}.png')
        plt.close()

    def get_keys_values(self, data):
        training_data = data[data['train']]
        q2 = np.stack(training_data['q2'])
        xnn = np.stack(training_data['reindexed_xnn'])[:, self.unit_cell_indices]
        hkl = np.stack(training_data['reindexed_hkl'])
        reciprocal_unit_cell = np.stack(training_data['reciprocal_reindexed_unit_cell'])[:, self.unit_cell_indices]
        reciprocal_unit_cell_volume = get_unit_cell_volume(
            reciprocal_unit_cell, partial_unit_cell=True, lattice_system=self.data_params['lattice_system']
            )
        
        q2_volume_scaled = q2 / reciprocal_unit_cell_volume[:, np.newaxis]**(2/3)
        xnn_volume_scaled = xnn / reciprocal_unit_cell_volume[:, np.newaxis]**(2/3)

        self.volume_scaler = StandardScaler(with_mean=False)
        self.volume_scaler.fit(reciprocal_unit_cell_volume[:, np.newaxis])

        if self.data_params['lattice_system'] in ['tetragonal', 'hexagonal']:
            ratio = CosineAttention.get_ratio_from_xnn(
                xnn, lattice_system=self.data_params['lattice_system'], tensorflow=False
                )[:, 0]
            chunk_size = 10000
            threshold = 0.999

            n_chunks = q2_volume_scaled.shape[0] // chunk_size + 1
            #print(q2_volume_scaled.shape[0], n_chunks)
            xnn_values = []
            q2_keys = []
            hkl_links = []

            sort_indices = np.argsort(ratio)
            q2_volume_scaled = q2_volume_scaled[sort_indices]
            xnn_volume_scaled = xnn_volume_scaled[sort_indices]
            hkl = hkl[sort_indices]

            for chunk_index in range(n_chunks):
                if chunk_index == n_chunks - 1:
                    xnn_chunk = xnn_volume_scaled[chunk_index * chunk_size:]
                    q2_chunk = q2_volume_scaled[chunk_index * chunk_size:]
                    hkl_chunk = hkl[chunk_index * chunk_size:]
                else:
                    xnn_chunk = xnn_volume_scaled[chunk_index * chunk_size: (chunk_index + 1) * chunk_size]
                    q2_chunk = q2_volume_scaled[chunk_index * chunk_size: (chunk_index + 1) * chunk_size]
                    hkl_chunk = hkl[chunk_index * chunk_size: (chunk_index + 1) * chunk_size]
                status = True
                while status:
                    similarity = np.matmul(q2_chunk[:, :self.key_length], q2_chunk[:, :self.key_length].T)
                    mag = np.linalg.norm(q2_chunk[:, :self.key_length], axis=1)
                    similarity /= mag[:, np.newaxis] * mag[np.newaxis]
                    neighbor_array = similarity > threshold
                    neighbor_count = np.sum(neighbor_array, axis=1)
                    if neighbor_count.size > 0 and neighbor_count.max() > 1:
                        highest_density_index = np.argmax(neighbor_count)
                        neighbor_indices = np.where(neighbor_array[highest_density_index])[0]
                        delete_indices = neighbor_indices[
                            np.where(neighbor_indices != highest_density_index)[0]
                            ]
                        xnn_chunk = np.delete(xnn_chunk, delete_indices, axis=0)
                        q2_chunk = np.delete(q2_chunk, delete_indices, axis=0)
                        hkl_chunk = np.delete(hkl_chunk, delete_indices, axis=0)
                    else:
                        status = False
                xnn_values.append(xnn_chunk)
                q2_keys.append(q2_chunk)
                hkl_links.append(hkl_chunk)
                    
            xnn_values = np.row_stack(xnn_values)
            q2_keys = np.row_stack(q2_keys)
            hkl_links = np.row_stack(hkl_links)
            #print(self.q2_keys.shape)
            #print(self.xnn_values.shape)
            #print(self.hkl_links.shape)
        else:
            assert False

        return q2_keys, xnn_values, hkl_links

    def build_model(self, data=None):
        inputs = {
            'q2_queries': tf.keras.Input(
                shape=self.data_params['n_peaks'],
                name='q2_queries',
                dtype=tf.float32,
                )
            }
        if not data is None:
            q2_keys, xnn_values, hkl_links = self.get_keys_values(data)
            xnn_ratio_values = CosineAttention.get_ratio_from_xnn(
                xnn_values, lattice_system=self.data_params['lattice_system'], tensorflow=False
                )
            self.model_params['n_keys'] = hkl_links.shape[0]
            source = None
        else:
            xnn_ratio_values = None
            hkl_links = None
            source = f'{self.save_to}/{self.split_group}_pitf_weights_{self.model_params["tag"]}.h5'
        self.AttentionLayer = CosineAttention(
            xnn_ratio_values,
            hkl_links,
            self.key_length,
            self.model_params['ratio_start'],
            self.lattice_system,
            source
            )
        if not data is None:
            self.AttentionLayer.validate_linkage(q2_keys)
        self.model = tf.keras.Model(inputs, self.model_builder(inputs))
        self.compile_model()
        self.model.summary()

    def model_builder(self, inputs):
        # Calculate the reciprocal volume from the ratio of q2_keys and q2_queries
        # inputs['q2_queries']:        batch_size, n_peaks
        # self.AttentionLayer.q2_keys: n_keys,     n_peaks
        # query_key_distance:          batch_size, n_keys
        # weights:                     batch_size, n_keys
        # reciprocal_volume:           batch_size, n_keys

        query_key_distance, reciprocal_volume = self.AttentionLayer(inputs['q2_queries'])
        reciprocal_volume_scaled = (reciprocal_volume - self.volume_scaler.mean_[0]) / self.volume_scaler.scale_[0]
        weights = tf.keras.layers.Softmax(
            name='weights',
            )(query_key_distance)
        weights__reciprocal_volume_scaled = tf.keras.layers.Concatenate(
            axis=2,
            name='weights__reciprocal_volume_scaled'
            )((
                weights[:, :, tf.newaxis],
                reciprocal_volume_scaled[:, :, tf.newaxis]
                ))
        return [weights, weights__reciprocal_volume_scaled]

    def compile_model(self):
        ####################
        # similarity model #
        ####################
        optimizer = tf.optimizers.legacy.Adam(self.model_params['learning_rate'])
        loss_functions = {
            'weights': self.AttentionLayer.ratio_target_function,
            'weights__reciprocal_volume_scaled': self.AttentionLayer.reciprocal_volume_target_function,
            }
        loss_weights = {
            'weights': 1,
            'weights__reciprocal_volume_scaled': 1,
            }
        self.model.compile(
            optimizer=optimizer, 
            loss=loss_functions,
            loss_weights=loss_weights,
            )

    def evaluate(self, data):
        """
        RMS xnn:
            - most probable
            - top 10
            - top 100
        """

        xnn_true = np.stack(data['reindexed_xnn'])[:, self.data_params['unit_cell_indices']]
        inputs = {'q2_queries': np.stack(data['q2'])}
        outputs = self.model(inputs)
        q2_keys = self.AttentionLayer.link_keys_and_values()

        weights = outputs[1][:, :, 0]
        reciprocal_volume_scaled = outputs[1][:, :, 1]
        reciprocal_volume_pred = reciprocal_volume_scaled*self.volume_scaler.scale_[0] + self.volume_scaler.mean_[0]

        reciprocal_unit_cell = np.stack(
            data['reciprocal_reindexed_unit_cell']
            )[:, self.unit_cell_indices]
        reciprocal_volume_true = get_unit_cell_volume(
            reciprocal_unit_cell,
            partial_unit_cell=True,
            lattice_system=self.data_params['lattice_system']
            )

        xnn_values = self.AttentionLayer.get_xnn_from_ratio(
            self.AttentionLayer.xnn_ratio_values.numpy(),
            lattice_system=self.data_params['lattice_system'],
            tensorflow=False
            )
        print(weights.shape)
        print(reciprocal_volume_scaled.shape)
        print(xnn_values.shape)

        np.save('q2_queries.npy', inputs['q2_queries'])
        np.save('q2_keys.npy', q2_keys)
        np.save('weights.npy', weights)
        np.save('reciprocal_volume_pred.npy', reciprocal_volume_pred)
        np.save('reciprocal_volume_true.npy', reciprocal_volume_true)
        np.save('xnn_values.npy', xnn_values)
        np.save('xnn_true.npy', xnn_true)
