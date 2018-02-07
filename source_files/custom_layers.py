import tensorflow as tf
import tensorlayer as tl

def activation_module(layer, activation_fn, leaky_relu_alpha=0.2):

    if activation_fn not in ["ReLU", "PReLU", "Leaky_ReLU", "CReLU", "ELU", "SELU", "tanh", "sigmoid", None]:
        raise Exception("Unknown 'activation_fn': %s" % activation_fn)

    if activation_fn == "ReLU":
        layer.outputs = tf.nn.relu(
            features = layer.outputs,
            name     = 'activation'
        )

    if activation_fn == "PReLU":
        layer = tl.layers.PReluLayer(
            layer  = layer,
            a_init = tf.constant_initializer(0.2),
            name   = 'activation'
        )

    elif activation_fn == "Leaky_ReLU":
        layer.outputs = tf.nn.leaky_relu(
            features = layer.outputs,
            alpha    = leaky_relu_alpha,
            name     = 'activation'
        )

    elif activation_fn == "CReLU":
        layer.outputs = tf.nn.crelu(
            features = layer.outputs,
            name     = 'activation'
        )

    elif activation_fn == "ELU":
        layer.outputs = tf.nn.elu(
            features = layer.outputs,
            name     = 'activation'
        )

    elif activation_fn == "SELU":
        layer.outputs = tf.nn.selu(
            features = layer.outputs,
            name     = 'activation'
        )

    elif activation_fn == "tanh":
        layer.outputs = tf.nn.tanh(
            x        = layer.outputs,
            name     = 'activation'
        )

    elif activation_fn == "sigmoid":
        layer.outputs = tf.nn.sigmoid(
            x        = layer.outputs,
            name     = 'activation'
        )

    return layer


def conv_module(
    input,
    n_out_channel,
    filter_size,
    strides,
    padding,
    conv_init,
    batch_norm_init,
    is_train,
    use_batchnorm=True,
    activation_fn=None
):

    if activation_fn not in ["ReLU", "PReLU", "Leaky_ReLU", "CReLU", "ELU", "SELU", "tanh", "sigmoid", None]:
        raise Exception("Unknown 'activation_fn': %s" % activation_fn)

    # Conv Layer - 1
    layer = tl.layers.Conv2d(
        input,
        n_filter    = n_out_channel,
        filter_size = filter_size,
        strides     = strides,
        padding     = padding,
        act         = tf.identity,
        W_init      = conv_init,
        name        = 'conv2d'
    )

    if use_batchnorm:
        layer = tl.layers.BatchNormLayer(
            layer,
            act        = tf.identity,
            is_train   = is_train,
            gamma_init = batch_norm_init,
            name       = 'batch_norm'
        )

    logits = layer.outputs

    layer = activation_module(layer, activation_fn)

    return layer, logits

def dense_module(
    input,
    n_units,
    dense_init,
    batch_norm_init,
    is_train,
    use_batchnorm  = True,
    flatten_before = False,
    activation_fn  = None
):
    # Flatten: Conv to FC
    if flatten_before:
        layer = tl.layers.FlattenLayer(input, name='flatten')
    else:
        layer = input

    # FC Layer - 5
    layer = tl.layers.DenseLayer(
        layer,
        n_units = n_units,
        act     = tf.identity,
        W_init  = dense_init,
        name    = 'dense'
    )

    if use_batchnorm:
        layer = tl.layers.BatchNormLayer(
            layer,
            act        = tf.identity,
            is_train   = is_train,
            gamma_init = batch_norm_init,
            name       = 'batch_norm'
        )

    logits = layer.outputs

    layer = activation_module(layer, activation_fn)

    return layer, logits