import numpy as np
import tensorlayer as tl
import tensorflow as tf


def cnn_network(inputs):
    # =============================== GAME RULES ===============================
    # 1. Only 2 Layers should be in the networks, no more, no less !
    #    => This is means, there should be no:
    #       ** No activation function/layer
    #       ** No pooling
    #       ** Basically nothing else than convolutions.
    #
    # 2. You are only allowed to use convolution layers (No Dense, No Reshape, No Whatever else)
    #
    #    2.1 Tensorflow:
    #       ** tf.nn.convolution (Hardcore Level)
    #       ** tf.nn.conv2d (Expert Level)
    #       ** tf.contrib.layers.conv2d (Easy Level)
    #
    #    2.2 Tensorlayer:
    #       ** tl.layers.Conv2dLayer (Medium Level)
    #       ** tl.layers.Conv2d (Easy Level)
    #
    # 3. You are free to use whatever settings you want for the convolutions
    # =============================== GAME RULES ===============================

    # implements here the CNN

    # Settings for conv layer 1
    f1 = 7
    s1 = 2
    n_c1 = 20
    pad1 = "SAME"
    act1 = tf.nn.relu

    # Settings for conv layer 2
    f2 = 128
    s2 = 128
    n_c2 = 10
    pad2 = "VALID"
    act2 = tf.nn.tanh

    # tf.contrib.layers.conv2d (Easy Level)
    # """
    conv1 = tf.contrib.layers.conv2d(inputs, n_c1, [f1, f1], stride=s1, padding=pad1, activation_fn=act1)
    conv2 = tf.contrib.layers.conv2d(conv1, n_c2, [f2, f2], stride=s2, padding=pad2, activation_fn=act2)
    # """

    # tf.nn.conv2d (Expert Level)
    """
    W1 = tf.get_variable("W1", [f1, f1, 1, n_c1], initializer=tf.contrib.layers.xavier_initializer())
    W2 = tf.get_variable("W2", [f2, f2, n_c1, n_c2], initializer=tf.contrib.layers.xavier_initializer())
    conv1 = act1(tf.nn.conv2d(inputs, W1, strides=[1, s1, s1, 1], padding=pad1))
    conv2 = act2(tf.nn.conv2d(conv1, W2, strides=[1, s2, s2, 1], padding=pad2))
    """

    # tf.nn.convolution (Hardcore Level)
    """
    W1 = tf.get_variable("W1", [f1, f1, 1, n_c1], initializer=tf.contrib.layers.xavier_initializer())
    W2 = tf.get_variable("W2", [f2, f2, n_c1, n_c2], initializer=tf.contrib.layers.xavier_initializer())
    conv1 = act1(tf.nn.convolution(inputs, W1, strides=[s1, s1], padding=pad1))
    conv2 = act2(tf.nn.convolution(conv1, W2, strides=[s2, s2], padding=pad2))
    """

    # tl.layers.Conv2d (Easy Level)
    """
    X = tl.layers.InputLayer(inputs, name='inputs')
    conv1 = tl.layers.Conv2d(X, n_filter=n_c1, filter_size=(f1, f1), strides=(s1, s1), act=act1, padding=pad1, name='conv1')
    conv2 = tl.layers.Conv2d(conv1, n_filter=n_c2, filter_size=(f2, f2), strides=(s2, s2), act=act2, padding=pad2, name='conv2')
    conv2 = conv2.outputs
    """

    # tl.layers.Conv2dLayer (Medium Level)
    """
    X = tl.layers.InputLayer(inputs, name='inputs')
    conv1 = tl.layers.Conv2dLayer(X, shape=(f1, f1, 1, n_c1), strides=(1, s1, s1, 1), act=act1, padding=pad1, name='conv1')
    conv2 = tl.layers.Conv2dLayer(conv1, shape=(f2, f2, n_c1, n_c2), strides=(1, s2, s2, 1), act=act2, padding=pad2, name='conv2')
    conv2 = conv2.outputs
    """

    #return None  # Return the output of the network
    return conv2


if __name__ == '__main__':

    input_shape = [None, 256, 256, 1]

    input_plh = tf.placeholder(tf.float32, input_shape)
    network = cnn_network(input_plh)

    batch_size = 32

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        random_input = np.random.random([batch_size] + input_shape[1:])

        net_out = sess.run(network, feed_dict={input_plh: random_input})

        if (net_out.shape != (32, 1, 1, 10)):
            raise ValueError("The network does not output the correct shape: %s instead of: %s" %
                             (str(net_out.shape), str((32, 1, 1, 10))))

        if (net_out.max() >= 1):
            raise ValueError("The maximum value of the network should be 1 (excluded) instead of: %f" % net_out.max())

        if (net_out.min() <= -1):
            raise ValueError("The minimum value of the network should be -1 (excluded) instead of: %f" % net_out.min())

        print("\n################################################################################")
        print("# Network Output Mean: %f - Network Output Standard Deviation: %f #" % (net_out.mean(), net_out.std()))
        print("# Congratulation the exercise is finished with success!                        #")
        print("################################################################################")