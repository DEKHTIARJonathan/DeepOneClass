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

    return None #Return the output of the network

if __name__ == '__main__':

    input_shape = [None, 256, 256, 1]

    input_plh   = tf.placeholder(tf.float32, input_shape)
    network     = cnn_network(input_plh)

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