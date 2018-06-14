#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import pandas as pd

import itertools
from matplotlib import pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from estimator_svdd import SVDDClassifier as SVDDClassifier
from vgg_network import VGG_Network
from data_utils import test_cnn_input_fn, test_input_fn, run_dataset_through_network
from data_utils import _LoadPreTrainedWeights
from flags import FLAGS

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm[0].sum() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def evaluation_summary(y_true, y_pred, plot_cm=False):
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    if plot_cm:
        plot_confusion_matrix(cm, classes=["Outlier", "Normal"],
                              normalize=False, title='Confusion matrix')

    return pd.Series({
        "Accuracy": acc,
        "Precision": pre,
        "Recall": rec,
        "f1-score": f1
    })

def main(argv=None):
    train_hooks = []

    if FLAGS.mode == 'cached':
        input_fn_test = lambda: test_cnn_input_fn(FLAGS.class_nbr, FLAGS.cnn_output_dir)\
                                 .batch(FLAGS.batch_size)
    else:
        vgg_net = VGG_Network(include_FC_head=False)

        input_fn_test = lambda: run_dataset_through_network(
                                     test_input_fn(FLAGS.class_nbr, FLAGS.target_width)\
                                         .batch(FLAGS.batch_size),
                                     vgg_net
                                 )

        train_hooks.append(_LoadPreTrainedWeights(vgg_net))

    tf.logging.info('Creating the classifier\n\n')
    classifier = SVDDClassifier(
        c=FLAGS.c,
        kernel=FLAGS.kernel,
        model_dir=FLAGS.model_dir,
    )

    tf.logging.info('Predicting with the classifier\n\n')
    predictions = classifier.predict(
        input_fn=input_fn_test,
        hooks=train_hooks
    )

    predictions = list(predictions)
    test_predicted_scores = np.asarray(list(map(lambda p: p["predicted_scores"], predictions))).astype(np.int32)
    test_predicted_classes = np.asarray(list(map(lambda p: p["predicted_classes"], predictions))).astype(np.int32)

    y_test = []
    input_fn = test_cnn_input_fn(FLAGS.class_nbr, FLAGS.cnn_output_dir).batch(1)
    input_fn = input_fn.make_one_shot_iterator().get_next()
    sess = tf.Session()
    while True:
        try:
            data = sess.run(input_fn)
            y_test.append(data[1][0])
        except tf.errors.OutOfRangeError:
            break
    y_test = np.asarray(y_test)
    tf.logging.debug(y_test.shape)

    evaluation_summary(y_test, test_predicted_classes, plot_cm=True)


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()