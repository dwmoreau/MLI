import tensorflow as tf


def mlp_model_builder(x, tag, model_params, output_name):
    for index in range(len(model_params['layers'])):
        x = tf.keras.layers.Dense(
            model_params['layers'][index],
            activation='linear',
            name=f'dense_{tag}_{index}',
            use_bias=False,
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
        model_params['unit_cell_length'],
        activation=model_params['output_activation'],
        name=output_name,
        kernel_initializer=model_params['kernel_initializer'],
        bias_initializer=model_params['bias_initializer'],
        )(x)
    return output


def hkl_model_builder_additive(x_in, tag, model_params):
    # doing 10 classifications effectively
    # before softmax: batch_size x 10 x 100
    # after softmax: batch_size x 10 x 100
    # y_true: batch_size x 10

    x = tf.keras.layers.Dense(
        model_params['hkl_ref_length'],
        activation='linear',
        name=f'dense_{tag}_0',
        kernel_regularizer=tf.keras.regularizers.OrthogonalRegularizer(
            factor=model_params['Ortho_kernel_reg'],
            mode='rows'
            ),
        use_bias=False,
        )(x_in)
    x = tf.keras.layers.LayerNormalization(
        epsilon=model_params['epsilon'], 
        name=f'layer_norm_{tag}_0',
        )(x)
    x = tf.keras.activations.gelu(x)
    x = tf.keras.layers.Dropout(
        rate=model_params['dropout_rate'],
        name=f'dropout_{tag}_0'
        )(x)

    x = tf.keras.layers.Dense(
        model_params['hkl_ref_length'],
        activation='linear',
        name=f'hkl_output_{tag}',
        kernel_regularizer=tf.keras.regularizers.OrthogonalRegularizer(
            factor=model_params['Ortho_kernel_reg'],
            mode='rows'
            ),
        )(x)

    hkl_out = tf.keras.layers.Softmax(
        axis=2,
        name=f'hkl_{tag}'
        )(x + x_in)
    return hkl_out


def hkl_model_builder(x, tag, model_params):
    # doing 10 classifications effectively
    # before softmax: batch_size x 10 x 100
    # after softmax: batch_size x 10 x 100
    # y_true: batch_size x 10
    for index in range(len(model_params['layers'])):
        x = tf.keras.layers.Dense(
            model_params['layers'][index],
            activation='linear',
            name=f'dense_{tag}_{index}',
            kernel_regularizer=tf.keras.regularizers.OrthogonalRegularizer(
                factor=model_params['Ortho_kernel_reg'],
                mode='rows'
                ),
            use_bias=False,
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

    x = tf.keras.layers.Dense(
        model_params['hkl_ref_length'],
        activation='linear',
        name=f'hkl_output_{tag}',
        kernel_regularizer=tf.keras.regularizers.OrthogonalRegularizer(
            factor=model_params['Ortho_kernel_reg'],
            mode='rows'
            ),
        )(x)
    hkl_out = tf.keras.layers.Softmax(
        axis=2,
        name=f'hkl_{tag}'
        )(x)
    return hkl_out
