import numpy as np
import tensorlayer as tl
import tensorflow as tf

def cnn_network(inputs):
    # =============================== GAME RULES ===============================
    # 1. Only 2 Layers should be in the networks, no more, no less !
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

    filter1_shape = (1, 1, 1, 1)
    filter2_shape = (256, 256, 1, 10)

    # 2.1
    if True:
        filter1 = tf.Variable(tf.truncated_normal(filter1_shape, dtype=tf.float32, stddev=0.02))
        filter2 = tf.Variable(tf.truncated_normal(filter2_shape, dtype=tf.float32, stddev=0.02))
        l1 = tf.nn.convolution(inputs, filter=filter1, padding="VALID", name="conv1")
        l2 = tf.nn.convolution(l1, filter=filter2, padding="VALID", name="conv1")
        l2_out = tf.nn.tanh(l2)

    # 2.2
    else:
        il = tl.layers.InputLayer(inputs, name="inputs")
        l1 = tl.layers.Conv2dLayer(il, act=tf.identity, shape=list(filter1_shape), padding="VALID", name="conv1")
        l2 = tl.layers.Conv2dLayer(l1, act=tf.nn.tanh, shape=list(filter2_shape), padding="VALID", name="conv2")
        l2_out = l2.outputs

    return l2_out #Return the output of the network

if __name__ == '__main__':

    input_shape = [None, 256, 256, 1]

    input_plh   = tf.placeholder(tf.float32, input_shape)

    network = cnn_network(input_plh)

    batch_size = 32

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        random_input = np.random.random([batch_size] + input_shape[1:])

        net_out = sess.run(network, feed_dict={input_plh: random_input})

        if (net_out.shape != (32, 1, 1, 10)):
            raise ValueError("The network does not output the correct shape: %s instead of: %s" %
            (str(net_out.shape), str((32, 1, 1, 10))))

        if(net_out.max() >= 1):
            raise ValueError("The maximum value of the network should be 1 (excluded) instead of: %f" % net_out.max())

        if (net_out.min() <= -1):
            raise ValueError("The minimum value of the network should be -1 (excluded) instead of: %f" % net_out.min())

        print("\n################################################################################")
        print("# Network Output Mean: %f - Network Output Standard Deviation: %f #" % (net_out.mean(), net_out.std()))
        print("# Congratulation the exercise is finished with success!                        #")
        print("################################################################################")