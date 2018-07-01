#!/usr/bin/env python

import tensorflow as tf

from data_utils import train_cached_features_dataset, train_img_dataset, run_dataset_through_network
from data_utils import _LoadPreTrainedWeights
from vgg_network import VGG_Network
from estimator_svdd import SVDDClassifier
from estimator_ocsvm import OCSVMClassifier
from flags import FLAGS

def main(argv=None):
    train_hooks = []

    if FLAGS.mode == 'cached':
        def input_fn_train():
            with tf.name_scope('input_dataset'):
                dataset = train_cached_features_dataset(FLAGS.class_nbr, FLAGS.cnn_output_dir, FLAGS.cnn_out_dims)
                dataset = dataset.repeat(FLAGS.epochs).batch(FLAGS.batch_size)
                return dataset

    else:
        vgg_net = VGG_Network(include_FC_head=False)

        def input_fn_train():
            with tf.name_scope('input_dataset'):
                dataset = train_img_dataset(FLAGS.class_nbr, FLAGS.target_width)
                dataset = dataset.repeat(FLAGS.epochs).batch(FLAGS.batch_size)
                dataset = run_dataset_through_network(dataset, vgg_net)
                return dataset

        train_hooks.append(_LoadPreTrainedWeights(vgg_net))

    tf.logging.info('Creating the classifier\n\n')
    if FLAGS.type == "ocsvm":
        classifier = OCSVMClassifier(
            c=FLAGS.c,
            kernel=FLAGS.kernel,
            learning_rate=FLAGS.learning_rate,
            model_dir=FLAGS.model_dir
        )
    else:
        classifier = SVDDClassifier(
            c=FLAGS.c,
            kernel=FLAGS.kernel,
            learning_rate=FLAGS.learning_rate,
            model_dir=FLAGS.model_dir,
            rffm_dims=FLAGS.rffm_dims,
            rffm_stddev=FLAGS.rffm_stddev
        )

    tf.logging.info('Training the classifier\n\n')
    classifier.train(
        input_fn=input_fn_train,
        hooks=train_hooks
    )


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
