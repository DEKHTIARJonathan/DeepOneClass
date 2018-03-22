import numpy as np
import tensorflow as tf

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from naive_svdd import SvddLayer
from vgg_network import VGG_Network

from data_utils import get_train_val_imgs

import matplotlib.pyplot as plt

dataset_number  = 6
target_w        = 224
input_shape     = [None, target_w, target_w, 3]
batch_size      = 10
# n_epoch         = 25
n_epoch         = 1
kernel          = "linear"
rffm_dims       = 200
rffm_stddev     = 25
c               = 1
seed            = 3

np.random.seed(seed)


train_images, test_images, test_images_y = get_train_val_imgs(
    dataset_number,
    '../data',
    target_w
)

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
        network.train(sess, input_plh, train_images, n_epoch=n_epoch, batch_size=batch_size)

    except KeyboardInterrupt: # allows to stop training with ctrl-C
        pass

    eval_y, _ = network.predict(sess, input_plh, test_images, batch_size=batch_size)

    n = len(test_images_y)
    acc = accuracy_score(test_images_y, eval_y)
    cm = confusion_matrix(test_images_y, eval_y) / n
    print("======== Results ========")
    print("Accuracy: %f" % acc)
    print("Confusion Matrix:")
    print(cm)

    true_normal   = test_images[(test_images_y > 0) & (eval_y > 0)]
    false_defect = test_images[(test_images_y > 0) & (eval_y < 0)]
    false_normal  = test_images[(test_images_y < 0) & (eval_y > 0)]
    true_defect  = test_images[(test_images_y < 0) & (eval_y < 0)]

    if len(true_normal) > 0:
        plt.subplot(221)
        plt.imshow(
            true_normal[0][:, :, 0],
            vmin=0,
            vmax=1,
            cmap='gray',
            interpolation=None
        )
        plt.axis('off')
        plt.title("True normal - {}".format(cm[1, 1]))

    if len(false_defect) > 0:
        plt.subplot(222)
        plt.imshow(false_defect[0][:, :, 0],
                   vmin=0,
                   vmax=1,
                   cmap='gray',
                   interpolation=None
                   )
        plt.axis('off')
        plt.title("False default - {}".format(cm[1, 0]))

    if len(false_normal) > 0:
        plt.subplot(223)
        plt.imshow(false_normal[0][:, :, 0],
                   vmin=0,
                   vmax=1,
                   cmap='gray',
                   interpolation=None
        )
        plt.axis('off')
        plt.title("False normal - {}".format(cm[0, 1]))

    if len(true_defect) > 0:
        plt.subplot(224)
        plt.imshow(true_defect[0][:, :, 0],
                   vmin=0,
                   vmax=1,
                   cmap='gray',
                   interpolation=None
                   )
        plt.axis('off')
        plt.title("True default - {}".format(cm[0, 0]))

    plt.show()