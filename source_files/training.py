#!/usr/bin/env python

import tensorflow as tf

from data_utils import train_cnn_input_fn, train_input_fn, run_dataset_through_network
from data_utils import _LoadPreTrainedWeights
from vgg_network import VGG_Network
from estimator_svdd import SVDDClassifier as SVDDClassifier
from flags import FLAGS

def main(argv=None):
    train_hooks = []

    if FLAGS.mode == 'cached':
        input_fn_train = lambda: train_cnn_input_fn(FLAGS.class_nbr, FLAGS.cnn_output_dir)\
                                 .repeat(FLAGS.epochs).batch(FLAGS.batch_size)
    else:
        vgg_net = VGG_Network(include_FC_head=False)

        input_fn_train = lambda: run_dataset_through_network(
                                     train_input_fn(FLAGS.class_nbr, FLAGS.target_width)\
                                         .repeat(FLAGS.epochs).batch(FLAGS.batch_size),
                                     vgg_net
                                 )

        train_hooks.append(_LoadPreTrainedWeights(vgg_net))

    tf.logging.info('Creating the classifier\n\n')
    classifier = SVDDClassifier(
        c=FLAGS.c,
        kernel=FLAGS.kernel,
        learning_rate=FLAGS.learning_rate,
        model_dir=FLAGS.model_dir,
    )

    tf.logging.info('Validating the classifier\n\n')
    classifier.train(
        input_fn=input_fn_train,
        hooks=train_hooks
    )


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
