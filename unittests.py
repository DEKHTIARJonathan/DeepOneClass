import sys
import unittest

import tensorflow as tf
import tensorlayer as tl

sys.path.append("tests")

runner = unittest.TextTestRunner(verbosity=2)

if __name__ == '__main__':

    #tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.set_verbosity(tf.logging.DEBUG)

    for test_module in []:

        tf.logging.info("Running tests for: %s ..." % test_module)

        test_suite = unittest.TestLoader().loadTestsFromTestCase(test_module)
        runner.run(test_suite)

        tf.reset_default_graph()
        tl.layers.clear_layers_name()