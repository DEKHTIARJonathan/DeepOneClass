import os

import numpy as np

import tensorflow as tf
import tensorlayer as tl

from tensorflow.contrib.slim.python.slim.nets.inception_v3 import inception_v3
from tensorflow.contrib.slim.python.slim.nets.inception_v3 import inception_v3_arg_scope

slim = tf.contrib.slim

class InceptionV3_Network(object):
    """InceptionV3 Network model.
    """

    def __call__(self, inputs):

        # Input Layers
        net_in = tl.layers.InputLayer(inputs, name='input_layer')

        with slim.arg_scope(inception_v3_arg_scope()):

            network = tl.layers.SlimNetsLayer(
                prev_layer=net_in,
                slim_layer=inception_v3,
                slim_args={
                    'num_classes': 1001,
                    'is_training': False,
                },
                name='InceptionV3'
            )

            self.network = network
            self.network.print_params(False)

            conv_layers = [
                "InceptionV3/InceptionV3/Conv2d_1a_3x3/Relu",
                "InceptionV3/InceptionV3/Conv2d_2a_3x3/Relu",
                "InceptionV3/InceptionV3/Conv2d_2b_3x3/Relu",
                "InceptionV3/InceptionV3/Conv2d_3b_1x1/Relu",
                "InceptionV3/InceptionV3/Conv2d_4a_3x3/Relu",
                "InceptionV3/InceptionV3/Mixed_5b/concat",
                "InceptionV3/InceptionV3/Mixed_5c/concat",
                "InceptionV3/InceptionV3/Mixed_5d/concat",
                "InceptionV3/InceptionV3/Mixed_6a/concat",
                "InceptionV3/InceptionV3/Mixed_6b/concat",
                "InceptionV3/InceptionV3/Mixed_6c/concat",
                "InceptionV3/InceptionV3/Mixed_6d/concat",
                "InceptionV3/InceptionV3/Mixed_6e/concat",
                "InceptionV3/InceptionV3/Mixed_7a/concat",
                "InceptionV3/InceptionV3/Mixed_7b/concat",
                "InceptionV3/InceptionV3/Mixed_7c/concat",
            ]

            conv_outs = [
                tl.layers.get_layers_with_name(
                    self.network,
                    name      = layer_name,
                    printable = False
                )[0]
                for layer_name in conv_layers
            ]

            return network, conv_outs

    def load_pretrained(self, sess, weights_path='weights/inception_v3.ckpt'):

        tf.logging.info("Loading InceptionV3 Net weights ...")

        if not os.path.isfile(weights_path):
            raise FileNotFoundError(
                "Please download inception_v3 ckpt from : https://github.com/tensorflow/models/tree/master/research/slim"
            )

        self.network.print_params(False)

        saver = tf.train.Saver()
        saver.restore(sess, weights_path)

        tf.logging.info("Finished loading InceptionV3 Net weights ...")

###################


if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.DEBUG)

    input_plh = tf.placeholder(tf.float32, [None, 299, 299, 3], name='input_placeholder')

    inception_model    = InceptionV3_Network()
    network, conv_outs = inception_model(input_plh)

    with tf.Session() as sess:
        inception_model.load_pretrained(sess)

        dummy_data = np.random.random([32, 299, 299, 3])

        result = sess.run(conv_outs, feed_dict={input_plh: dummy_data})

        for i, output in enumerate(result):
            print("Result Shape " + str(i + 1) + ":", output.shape)
