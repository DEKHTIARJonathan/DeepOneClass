#!/usr/bin/env python

import tensorflow as tf

from pathlib import Path

from data_utils import train_cnn_input_fn, train_input_fn, run_dataset_through_network
from vgg_network import VGG_Network
from estimator_svdd_naive import _LoadPreTrainedWeightsVGG
from estimator_svdd_naive import OCClassifier as SVDDClassifier


FLAGS = tf.flags.FLAGS

# Inputs params
tf.flags.DEFINE_integer('class_nbr', 6,
                        lower_bound=0, upper_bound=6,
                        help='DAGM class to train on.')
tf.flags.DEFINE_enum('mode', 'cached', ['direct', 'cached'],
                     help='''The input pipeline mode, either direct ent to end '''
                          '''forward pass, or through cached CNN output.''')
tf.flags.DEFINE_integer('target_width', 224,
                        lower_bound=0,
                        help='End width after transforming images.')

# Model params
tf.flags.DEFINE_enum('kernel', 'linear', ['linear', 'rbf', 'rffm'],
                     help='Approximation of the SVM kernel.')
tf.flags.DEFINE_float('learning_rate', 0.1,
                      lower_bound=0,
                      help='Starting value for the learning rate.')
tf.flags.DEFINE_float('c', 3,
                      lower_bound=0,
                      help='C parameter for the hardness of the margin.')

# Train params
tf.flags.DEFINE_string('cnn_output_dir', str(Path(__file__).parent / '../tmp/cnn_output/VGG16'),
                       help='Where the cached files are located, if using the cached mode')
tf.flags.DEFINE_integer('batch_size', 64,
                        lower_bound=1,
                        help='Batch size.')
tf.flags.DEFINE_integer('epochs', 100,
                        lower_bound=0,
                        help='Number of epochs.')
tf.flags.DEFINE_string('train_dir', str(Path(__file__).parent / '../tmp/estimator'),
                       help='Where to write events and checkpoints.')


def main(argv=None):
    train_hooks = []

    if FLAGS.mode == 'cached':
        input_fn_train = lambda: train_cnn_input_fn(FLAGS.class_nbr, FLAGS.cnn_output_dir)\
                                 .repeat(FLAGS.epochs).batch(FLAGS.batch_size)
    else:
        vgg_net = VGG_Network(include_FC_head=False)

        input_fn_train = lambda: run_dataset_through_network(
                                     train_input_fn(FLAGS.class_nbr, FLAGS.target_width) \
                                         .repeat(FLAGS.epochs).batch(FLAGS.batch_size),
                                     vgg_net
                                 )

        train_hooks.append(_LoadPreTrainedWeightsVGG(vgg_net))

    tf.logging.info('Creating the classifier\n\n')
    classifier = SVDDClassifier(
        c=FLAGS.c,
        kernel=FLAGS.kernel,
        learning_rate=FLAGS.learning_rate,
        model_dir=FLAGS.train_dir,
    )

    tf.logging.info('Training the classifier\n\n')
    classifier.train(
        input_fn=input_fn_train,
        hooks=train_hooks
    )


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
