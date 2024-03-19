import tensorflow as tf
import tensorflow_probability as tfp


# Define the prior weight distribution as Normal of mean=0 and stddev=1.
# Note that, in this example, the prior distribution is not trainable,
# as we fix its parameters.
def prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    prior_model = tf.keras.Sequential(
        [
            tfp.layers.DistributionLambda(
                lambda t: tfp.distributions.MultivariateNormalDiag(
                    loc=tf.zeros(n), scale_diag=tf.ones(n)
                )
            )
        ]
    )
    return prior_model


# Define variational posterior weight distribution as multivariate Gaussian.
# Note that the learnable parameters for this distribution are the means,
# variances, and covariances.
def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    posterior_model = tf.keras.Sequential(
        [
            tfp.layers.VariableLayer(
                tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype
            ),
            tfp.layers.MultivariateNormalTriL(n),
        ]
    )
    return posterior_model


def bnn_model_builder(x, tag, model_params, output_name, N_train):
    for index in range(len(model_params['layers'])):
        x = tfp.layers.DenseVariational(
            units=model_params['layers'][index],
            make_prior_fn=prior,
            make_posterior_fn=posterior,
            kl_weight=1/N_train,
            activation='linear',
            name=f'dense_{tag}_{index}',
            )(x)
        x = tf.keras.layers.LayerNormalization(
            epsilon=model_params['epsilon'], 
            name=f'layer_norm_{tag}_{index}'
            )(x)
        x = tf.keras.activations.gelu(x)

    if not output_name is None:
        output = tf.keras.layers.Dense(
            model_params['n_outputs'],
            activation=model_params['output_activation'],
            name=output_name,
            kernel_initializer=model_params['kernel_initializer'],
            bias_initializer=model_params['bias_initializer'],
            )(x)
        return output
    else:
        return x


def mlp_model_builder(x, tag, model_params, output_name):
    for index in range(len(model_params['layers'])):
        x = tf.keras.layers.Dense(
            model_params['layers'][index],
            activation='linear',
            name=f'dense_{tag}_{index}',
            )(x)
        x = tf.keras.layers.LayerNormalization(
            epsilon=model_params['epsilon'], 
            name=f'layer_norm_{tag}_{index}'
            )(x)
        x = tf.keras.activations.gelu(x)
        x = tf.keras.layers.Dropout(
            rate=model_params['dropout_rate'],
            name=f'dropout_{tag}_{index}',
            )(x)
    output = tf.keras.layers.Dense(
        model_params['n_outputs'],
        activation=model_params['output_activation'],
        name=output_name,
        kernel_initializer=model_params['kernel_initializer'],
        bias_initializer=model_params['bias_initializer'],
        )(x)
    return output


def hkl_model_builder_conv2D(x_in, tag, model_params):
    # doing 10 classifications effectively
    # before softmax: batch_size x 10 x 100
    # after softmax: batch_size x 10 x 100
    # y_true: batch_size x 10
    
    # x_class = batch_size x n_points x hkl_ref_length
    x = tf.keras.layers.Conv2D(
        filters=model_params['n_filters'],
        kernel_size=model_params['kernel_size'],
        strides=(1, 1),
        padding='same',
        activation='linear',
        name=f'conv2d_{tag}',
        )(x_in[:, :, :, tf.newaxis])
    x = tf.keras.layers.LayerNormalization(
        epsilon=model_params['epsilon'], 
        name=f'layer_norm_{tag}_conv2d',
        )(x)
    x = tf.keras.activations.gelu(x)
    x = tf.keras.layers.Dropout(
        rate=model_params['dropout_rate'],
        name=f'dropout_{tag}_conv2d'
        )(x)
    # x = batch_size x n_points x hkl_ref_length x filters
    x = tf.keras.layers.Reshape((
       model_params['n_points'],
       model_params['n_filters'] * model_params['hkl_ref_length']
       ))(x)

    for index in range(len(model_params['layers'])):
        x = tf.keras.layers.Dense(
            model_params['layers'][index],
            activation='linear',
            name=f'dense_{tag}_{index}',
            )(x)
        x = tf.keras.layers.LayerNormalization(
            epsilon=model_params['epsilon'], 
            name=f'layer_norm_{tag}_{index}',
            )(x)
        x = tf.keras.activations.gelu(x)
        x = tf.keras.layers.Dropout(
            rate=model_params['dropout_rate'],
            name=f'dropout_{tag}_{index}'
            )(x)
    
    hkl_outs = [None for _ in range(model_params['n_points'])]
    for index in range(model_params['n_points']):
        hkl_outs[index] = tf.keras.layers.Dense(
            units=model_params['hkl_ref_length'],
            activation=model_params['output_activation'],
            name=f'hkl_{tag}_{index}',
            )(x[:, index, :])[:, tf.newaxis, :]

    # hkl_out: n_batch, n_points, hkl_ref_length
    hkl_out = tf.keras.layers.Concatenate(
        axis=1,
        name=f'hkl_{tag}'
        )(hkl_outs)
    return hkl_out


def hkl_model_builder_mlp(x, tag, model_params):
    # doing 10 classifications effectively
    # before softmax: batch_size x 10 x 100
    # after softmax: batch_size x 10 x 100
    # y_true: batch_size x 10
    for index in range(len(model_params['layers'])):
        x = tf.keras.layers.Dense(
            model_params['layers'][index],
            activation='linear',
            name=f'dense_{tag}_{index}',
            )(x)
        x = tf.keras.layers.LayerNormalization(
            epsilon=model_params['epsilon'], 
            name=f'layer_norm_{tag}_{index}',
            )(x)
        x = tf.keras.activations.gelu(x)
        x = tf.keras.layers.Dropout(
            rate=model_params['dropout_rate'],
            name=f'dropout_{tag}_{index}'
            )(x)

    hkl_outs = [None for _ in range(model_params['n_points'])]
    for index in range(model_params['n_points']):
        hkl_outs[index] = tf.keras.layers.Dense(
            units=model_params['hkl_ref_length'],
            activation=model_params['output_activation'],
            name=f'hkl_{tag}_{index}',
            )(x[:, index, :])[:, tf.newaxis, :]

    # hkl_out: n_batch, n_points, hkl_ref_length
    hkl_out = tf.keras.layers.Concatenate(
        axis=1,
        name=f'hkl_{tag}'
        )(hkl_outs)
    return hkl_out



def hkl_model_builder_mlp_flat(x, tag, model_params):
    # doing 10 classifications effectively
    # before softmax: batch_size x 10 x 100
    # after softmax: batch_size x 10 x 100
    # y_true: batch_size x 10
    for index in range(len(model_params['layers'])):
        x = tf.keras.layers.Dense(
            model_params['layers'][index],
            activation='linear',
            name=f'dense_{tag}_{index}',
            )(x)
        x = tf.keras.layers.LayerNormalization(
            epsilon=model_params['epsilon'], 
            name=f'layer_norm_{tag}_{index}',
            )(x)
        x = tf.keras.activations.gelu(x)
        x = tf.keras.layers.Dropout(
            rate=model_params['dropout_rate'],
            name=f'dropout_{tag}_{index}'
            )(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(
        units=model_params['hkl_ref_length'] * model_params['n_points'],
        activation='linear',
        name=f'hkl_{tag}_flat',
        )(x)
    x = tf.keras.layers.Reshape(
        target_shape=(model_params['n_points'], model_params['hkl_ref_length']),
        name=f'hkl_{tag}_logits',
        )(x)

    # hkl_out: n_batch, n_points, hkl_ref_length
    hkl_out = tf.keras.layers.Softmax(
        axis=2,
        name=f'hkl_{tag}'
        )(x)
    return hkl_out


def hkl_model_builder_linear(x, tag, model_params):
    """
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(
        units=model_params['hkl_ref_length'] * model_params['n_points'],
        activation='linear',
        name=f'hkl_{tag}_flat',
        )(x)
    x = tf.keras.layers.Reshape(
        target_shape=(model_params['n_points'], model_params['hkl_ref_length']),
        name=f'hkl_{tag}_logits',
        )(x)
    # hkl_out: n_batch, n_points, hkl_ref_length
    hkl_out = tf.keras.layers.Softmax(
        axis=2,
        name=f'hkl_{tag}'
        )(x)
    """
    hkl_outs = [None for _ in range(model_params['n_points'])]
    for index in range(model_params['n_points']):
        hkl_outs[index] = tf.keras.layers.Dense(
            units=model_params['hkl_ref_length'],
            activation=model_params['output_activation'],
            name=f'hkl_{tag}_{index}',
            )(x[:, index, :])[:, tf.newaxis, :]
    # hkl_out: n_batch, n_points, hkl_ref_length
    hkl_out = tf.keras.layers.Concatenate(
        axis=1,
        name=f'hkl_{tag}'
        )(hkl_outs)
    return hkl_out


def hkl_model_builder_conv2D_flat(x_in, tag, model_params):
    # doing 10 classifications effectively
    # before softmax: batch_size x 10 x 100
    # after softmax: batch_size x 10 x 100
    # y_true: batch_size x 10
    
    # x_class = batch_size x n_points x hkl_ref_length
    x = tf.keras.layers.Conv2D(
        filters=model_params['n_filters'],
        kernel_size=model_params['kernel_size'],
        strides=(1, 1),
        padding='same',
        activation='linear',
        name=f'conv2d_{tag}',
        )(x_in[:, :, :, tf.newaxis])
    x = tf.keras.layers.LayerNormalization(
        epsilon=model_params['epsilon'], 
        name=f'layer_norm_{tag}_conv2d',
        )(x)
    x = tf.keras.activations.gelu(x)
    x = tf.keras.layers.Dropout(
        rate=model_params['dropout_rate'],
        name=f'dropout_{tag}_conv2d'
        )(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(
        units=model_params['hkl_ref_length'] * model_params['n_points'],
        activation='linear',
        name=f'hkl_{tag}_flat',
        )(x)
    x = tf.keras.layers.Reshape(
        target_shape=(model_params['n_points'], model_params['hkl_ref_length']),
        name=f'hkl_{tag}_logits',
        )(x)

    # hkl_out: n_batch, n_points, hkl_ref_length
    hkl_out = tf.keras.layers.Softmax(
        axis=2,
        name=f'hkl_{tag}'
        )(x)
    return hkl_out