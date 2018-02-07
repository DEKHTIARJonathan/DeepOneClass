import sys
import unittest

from unittest.util import strclass

parent_dir = "\\".join(sys.path[0].split("\\")[:-1])
sys.path.append(parent_dir)

import numpy as np
import tensorflow as tf

from source_files.model import OneClassCNN

class OneClassCNNTest(unittest.TestCase):

    @classmethod
    def __str__(cls):
        return strclass(cls)

    @classmethod
    def __unicode__(cls):
        return strclass(cls)

    @classmethod
    def __repr__(cls):
        return strclass(cls)

    @classmethod
    def setUpClass(cls):

        with tf.Session() as sess:
            model = OneClassCNN(sess)

            random_data  = np.random.random([64, 256, 256, 1])

            cls.model_output, cls.model_logits = sess.run(
                [
                    model.eval_model.outputs,
                    model.eval_logits
                ],
                feed_dict = {model.input_plh: random_data}
            )

    @classmethod
    def tearDownClass(cls):
        if tf.logging.get_verbosity() == tf.logging.DEBUG:
            print("\n\n###########################")

        tf.logging.debug("OneClassCNN:output => Shape: %s - Mean: %e - Std: %f - Min: %f - Max: %f" % (
            cls.model_output.shape,
            cls.model_output.mean(),
            cls.model_output.std(),
            cls.model_output.min(),
            cls.model_output.max()
        ))

        tf.logging.debug("OneClassCNN:logits => Shape: %s - Mean: %e - Std: %f - Min: %f - Max: %f" % (
            cls.model_logits.shape,
            cls.model_logits.mean(),
            cls.model_logits.std(),
            cls.model_logits.min(),
            cls.model_logits.max()
        ))


    def test_shape_last_conv_layer_logits(self):
        assert (self.model_output.shape == (64, 1, 1, 1))


    def test_min_output(self):
        assert (self.model_output.min() >= 0)


    def test_max_output(self):
        assert (self.model_output.max() <= 1)



if __name__ == '__main__':
    # tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.set_verbosity(tf.logging.DEBUG)

    unittest.main()