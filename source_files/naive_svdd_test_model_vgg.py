import numpy as np
import tensorflow as tf

import pandas
import os

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from naive_svdd import SvddLayer
from vgg_network import VGG_Network

from data_utils import *


import matplotlib.pyplot as plt

dataset_number  = 6
target_w        = 224
input_shape     = [None, target_w, target_w, 3]
batch_size      = 10
# n_epoch         = 25
n_epoch         = 10
kernel          = 'linear'
rffm_dims       = 200
rffm_stddev     = 25
c               = 1
seed            = 3
data_dir        = '../data/DAGM 2007 - Splitted'

np.random.seed(seed)

tf.logging.set_verbosity('DEBUG')


with tf.name_scope('Datasets'):

    train_dataset = get_train_dataset(
        class_nbr=dataset_number,
        data_dir=data_dir,
        target_w=target_w
    )\
        .apply(center_dataset_values)

    test_dataset = get_test_dataset(
        class_nbr=dataset_number,
        data_dir=data_dir,
        target_w=target_w
    )\
        .apply(center_dataset_values)

def create_model(inputs):

    with tf.name_scope('VGG'):
        vgg_model = VGG_Network(include_FC_head=False)
        vgg_network, _ = vgg_model(inputs)

    with tf.name_scope('OCSVM'):
        network = SvddLayer(
            vgg_network,
            c=c,
            map=kernel,
            rffm_dims=rffm_dims,
            rffm_stddev=rffm_stddev,
            freeze_before=True
        )

    return network, vgg_model

with tf.Session() as sess:

    with tf.name_scope('placeholders'):
        input_plh = tf.placeholder(tf.float32, shape=input_shape, name="X")

    network, vgg_model = create_model(input_plh)

    sess.run(tf.global_variables_initializer())
    vgg_model.load_pretrained(sess)

    try:
        network.train(sess, input_plh, train_dataset, n_epoch=n_epoch, batch_size=batch_size)

    except KeyboardInterrupt: # allows to stop training with ctrl-C
        pass

    eval_y, _ = network.predict(sess, input_plh, test_dataset, batch_size=batch_size)

    test_df = pandas.read_csv('../data/DAGM 2007 - Splitted/{}/test_files.csv'
                                    .format(dataset_number))
    test_images_y = test_df["is_healthy"]

    n = len(test_images_y)
    acc = accuracy_score(test_images_y, eval_y)
    cm = confusion_matrix(test_images_y, eval_y) / n
    print("======== Results ========")
    print("Accuracy: %f" % acc)
    print("Confusion Matrix:")
    print(cm)

    true_normal   = test_df[(test_images_y > 0) & (eval_y > 0)]["target"]
    false_defect = test_df[(test_images_y > 0) & (eval_y < 0)]["target"]
    false_normal  = test_df[(test_images_y < 0) & (eval_y > 0)]["target"]
    true_defect  = test_df[(test_images_y < 0) & (eval_y < 0)]["target"]

    if len(true_normal) > 0:
        fn = os.path.join(data_dir, str(dataset_number), 'test', true_normal.values[0].split('\\')[1])
        img = sess.run(load_and_transf_img(fn, target_w))
        plt.subplot(221)
        plt.imshow(
            img.astype(np.uint8),
            cmap='gray',
            interpolation=None
        )
        plt.axis('off')
        plt.title("True normal - {}".format(cm[1, 1]))

    if len(false_defect) > 0:
        fn = os.path.join(data_dir, str(dataset_number), 'test', false_defect.values[0].split('\\')[1])
        img = sess.run(load_and_transf_img(fn, target_w))
        print(img)
        plt.subplot(222)
        plt.imshow(img.astype(np.uint8),
                   cmap='gray',
                   interpolation=None
                   )
        plt.axis('off')
        plt.title("False defect - {}".format(cm[1, 0]))

    if len(false_normal) > 0:
        fn = os.path.join(data_dir, str(dataset_number), 'test', false_normal.values[0].split('\\')[1])
        img = sess.run(load_and_transf_img(fn, target_w))
        plt.subplot(223)
        plt.imshow(img.astype(np.uint8),
                   cmap='gray',
                   interpolation=None
        )
        plt.axis('off')
        plt.title("False normal - {}".format(cm[0, 1]))

    if len(true_defect) > 0:
        fn = os.path.join(data_dir, str(dataset_number), 'test', true_defect.values[0].split('\\')[1])
        img = sess.run(load_and_transf_img(fn, target_w))
        plt.subplot(224)
        plt.imshow(img.astype(np.uint8),
                   cmap='gray',
                   interpolation=None
                   )
        plt.axis('off')
        plt.title("True defect - {}".format(cm[0, 0]))

    plt.show()