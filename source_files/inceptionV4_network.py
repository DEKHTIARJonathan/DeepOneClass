import os

import numpy as np

import tensorflow as tf
import tensorlayer as tl

from tf_slim.inception_v4 import inception_v4
from tf_slim.inception_v4 import inception_v4_arg_scope

slim = tf.contrib.slim


class InceptionV4_Network(object):
    """InceptionV4 Network model.
    """

    def __call__(self, inputs):

        # Input Layers
        net_in = tl.layers.InputLayer(inputs, name='input_layer')

        with slim.arg_scope(inception_v4_arg_scope()):
            network = tl.layers.SlimNetsLayer(
                prev_layer=net_in,
                slim_layer=inception_v4,
                slim_args={
                    'num_classes': 1001,
                    'is_training': False,
                },
                name='inception_net'
            )

            self.network = network
            self.network.print_params(False)

            conv_layers = [
                "InceptionV4/InceptionV4/Conv2d_1a_3x3/Relu",
                "InceptionV4/InceptionV4/Conv2d_2a_3x3/Relu",
                "InceptionV4/InceptionV4/Conv2d_2b_3x3/Relu",
                "InceptionV4/InceptionV4/Mixed_3a/concat",
                "InceptionV4/InceptionV4/Mixed_4a/concat",
                "InceptionV4/InceptionV4/Mixed_5a/concat",
                "InceptionV4/InceptionV4/Mixed_5b/concat",
                "InceptionV4/InceptionV4/Mixed_5c/concat",
                "InceptionV4/InceptionV4/Mixed_5d/concat",
                "InceptionV4/InceptionV4/Mixed_5e/concat",
                "InceptionV4/InceptionV4/Mixed_6a/concat",
                "InceptionV4/InceptionV4/Mixed_6b/concat",
                "InceptionV4/InceptionV4/Mixed_6c/concat",
                "InceptionV4/InceptionV4/Mixed_6d/concat",
                "InceptionV4/InceptionV4/Mixed_6e/concat",
                "InceptionV4/InceptionV4/Mixed_6f/concat",
                "InceptionV4/InceptionV4/Mixed_6g/concat",
                "InceptionV4/InceptionV4/Mixed_6h/concat",
                "InceptionV4/InceptionV4/Mixed_7a/concat",
                "InceptionV4/InceptionV4/Mixed_7b/concat",
                "InceptionV4/InceptionV4/Mixed_7c/concat",
                "InceptionV4/InceptionV4/Mixed_7d/concat"
            ]

            conv_outs = [
                tl.layers.get_layers_with_name(
                    self.network,
                    name=layer_name,
                    printable=False
                )[0]
                for layer_name in conv_layers
            ]

            for layer in conv_outs:
                print(layer)

            return network, conv_outs

    def load_pretrained(self, sess, weights_path='../weights/inception_v4.ckpt'):

        tf.logging.info("Loading InceptionV4 Net weights ...")

        if not os.path.isfile(weights_path):
            raise FileNotFoundError(
                "Please download inception_v4 ckpt from : "
                "https://github.com/tensorflow/models/tree/master/research/slim"
            )

        self.network.print_params(False)

        saver = tf.train.Saver()
        saver.restore(sess, weights_path)

        tf.logging.info("Finished loading InceptionV4 Net weights ...")


###################


if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.DEBUG)

    input_plh = tf.placeholder(tf.float32, [None, 299, 299, 3], name='input_placeholder')

    inception_model = InceptionV4_Network()
    network, conv_outs = inception_model(input_plh)

    with tf.Session() as sess:
        inception_model.load_pretrained(sess)

        dummy_data = np.random.random([32, 299, 299, 3])

        result = sess.run(conv_outs, feed_dict={input_plh: dummy_data})

        for i, output in enumerate(result):
            print("Result Shape " + str(i + 1) + ":", output.shape)

        import numpy as np
        from skimage.transform import resize
        from imageio import imread

        test_image_path = 'data/laska.jpg'

        img_raw = imread(test_image_path)
        img_resized = resize(img_raw, (299, 299), mode='reflect') * 255

        net_out_probs = sess.run(network.outputs, feed_dict={input_plh: [img_resized]})[0]

        most_likely_classID = np.argmax(net_out_probs)
        max_prob = net_out_probs[most_likely_classID]

        # noinspection PyStringFormat
        print("InceptionV4 Network: Most Likely Class: %d with a probability of: %.5f - Output Shape: %s" % (
            most_likely_classID,
            max_prob,
            net_out_probs.shape
        ))

        print("Type:", type(most_likely_classID))
