import os
import sys
import unittest

from unittest.util import strclass

import numpy as np
import tensorflow as tf
import tensorlayer as tl

from source_files.naive_svdd import SvddLayer

class NaiveSvddTest(unittest.TestCase):

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

        cls.in_dims = 10
        cls.in_size = 100
        cls.map = 'rbf'
        cls.rffm_dims = 200
        cls.rffm_std = 25
        cls.c = 1

        inpt_plh = tf.placeholder(tf.float32, shape=(None, cls.in_dims), name="X")
        inpt = tl.layers.InputLayer(inpt_plh)
        cls.model = SvddLayer(inpt, c=cls.c, map=cls.map, rffm_dims=cls.rffm_dims, rffm_stddev=cls.rffm_std)
        outpt = cls.model

        x_test = np.random.multivariate_normal(mean=np.ones(cls.in_dims), cov=np.eye(cls.in_dims), size=cls.in_size).astype(np.float32)


        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            cls.svdd_out, cls.svdd_radius, cls.svdd_center = sess.run([outpt.outputs, outpt._radius, outpt._center], feed_dict = {inpt_plh: x_test})

    @classmethod
    def tearDownClass(cls):
        if tf.logging.get_verbosity() == tf.logging.DEBUG:
            print("\n\n###########################")

        tf.logging.debug("SvddNaive: Input dim: %d - Mapping: %s - RFFM dim: %d - RFFM std: %f - C: %f" % (
            cls.in_dims,
            cls.map,
            cls.rffm_dims,
            cls.rffm_std,
            cls.c
        ))


        tf.logging.debug("SvddNaive:output => Shape: %s - Mean: %e - Std: %f - Min: %f - Max: %f" % (
            cls.svdd_out.shape,
            cls.svdd_out.mean(),
            cls.svdd_out.std(),
            cls.svdd_out.min(),
            cls.svdd_out.max()
        ))

        tf.logging.debug("SvddNaive:radius => Shape: %s - Mean: %e - Std: %f - Min: %f - Max: %f" % (
            cls.svdd_radius.shape,
            cls.svdd_radius.mean(),
            cls.svdd_radius.std(),
            cls.svdd_radius.min(),
            cls.svdd_radius.max()
        ))

        tf.logging.debug("SvddNaive:center => Shape: %s - Mean: %e - Std: %f - Min: %f - Max: %f" % (
            cls.svdd_center.shape,
            cls.svdd_center.mean(),
            cls.svdd_center.std(),
            cls.svdd_center.min(),
            cls.svdd_center.max()
        ))

    def test_shape_output(self):
        self.assertEqual(self.svdd_out.shape, (self.in_size,))


    def test_min_output(self):
        self.assertGreaterEqual(self.svdd_out.min(), -1)


    def test_max_output(self):
        self.assertLessEqual(self.svdd_out.max(), 1)


    def test_positive_radius(self):
        self.assertGreater(self.svdd_radius, 0)


    def test_center_shape(self):
        if self.map == 'linear':
            self.assertEqual(self.svdd_center.shape, (self.in_dims,))
        else:
            self.assertEqual(self.svdd_center.shape, (self.rffm_dims,))


if __name__ == '__main__':
    # tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.set_verbosity(tf.logging.DEBUG)

    unittest.main()