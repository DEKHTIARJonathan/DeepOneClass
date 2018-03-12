import os
import sys
import unittest

from skimage.transform import resize
from imageio import imread

from unittest.util import strclass


parent_dir = "\\".join(sys.path[0].split("\\")[:-1])
sys.path.append(parent_dir)

import numpy as np
import tensorflow as tf

from source_files.model import OneClassCNN

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

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

            #random_data  = np.random.random([64, 256, 256, 1])
            #random_data = np.random.random([4, 256, 256, 1])

            test_image_path = 'data/laska.jpg'

            if os.getcwd().split(os.sep)[-1] != "tests":
                test_image_path = 'tests/' + test_image_path

            img_raw = imread(test_image_path)
            img_resized = resize(img_raw, (256, 256), mode='reflect') * 255

            img_prepared = rgb2gray(img_resized) # Shape: (256, 256)

            img_prepared = np.expand_dims(img_prepared, axis=2) # Shape: (256, 256, 1)
            img_prepared = np.expand_dims(img_prepared, axis=0) # Shape: (1, 256, 256, 1)

            cls.vgg_out = sess.run(model.vgg_net.outputs, feed_dict = {model.input_plh: img_prepared})

    @classmethod
    def tearDownClass(cls):
        if tf.logging.get_verbosity() == tf.logging.DEBUG:
            print("\n\n###########################")

        tf.logging.debug("OneClassCNN:output => Shape: %s - Mean: %e - Std: %f - Min: %f - Max: %f" % (
            cls.vgg_out.shape,
            cls.vgg_out.mean(),
            cls.vgg_out.std(),
            cls.vgg_out.min(),
            cls.vgg_out.max()
        ))

    def test_shape_last_conv_layer_logits(self):
        self.assertEqual(self.vgg_out.shape, (1, 7, 7, 512))


    def test_min_output(self):
        self.assertGreaterEqual(self.vgg_out.min(), 0)


    def test_max_output(self):
        # Test that the model is correctly loaded. Input is fixed, output should be fixed.
        self.assertAlmostEqual (self.vgg_out.max(), 542.8046875)


if __name__ == '__main__':
    # tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.set_verbosity(tf.logging.DEBUG)

    unittest.main()