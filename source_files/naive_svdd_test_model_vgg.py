
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from naive_svdd import SvddLayer
from sklearn.metrics import accuracy_score, confusion_matrix
from vgg_network import VGG_Network
from data_utils import get_train_val_imgs

np.random.seed(3)

dataset_number  = 6
target_w        = 224
input_shape     = [None, target_w, target_w, 3]
batch_size      = 10
n_epoch         = 25
kernel          = "linear"
rffm_dims       = 200
rffm_stddev     = 25
c               = 1


train_images, test_images, test_images_y = get_train_val_imgs(dataset_number, '../data', target_w)


input_plh = tf.placeholder(tf.float32, shape=input_shape, name="X")

with tf.Session() as sess:

    model = VGG_Network(include_FC_head=False)
    network, _ = model(input_plh)

    layer = SvddLayer(network, c=c, map=kernel, rffm_dims=rffm_dims, rffm_stddev=rffm_stddev, freeze_before=True)

    sess.run(tf.global_variables_initializer())
    model.load_pretrained(sess)

    try:
        layer.train(sess, input_plh, train_images, n_epoch=n_epoch, batch_size=batch_size)
    except KeyboardInterrupt:
        pass

    eval_y, _ = layer.predict(sess, input_plh, test_images, batch_size=batch_size)

    n = len(test_images_y)
    acc = accuracy_score(test_images_y, eval_y)
    cm = confusion_matrix(test_images_y, eval_y) / n
    print("======== Results ========")
    print("Accuracy: %f" % acc)
    print("Confusion Matrix:")
    print(cm)

    true_normal = test_images[(test_images_y > 0) & (eval_y > 0)]
    false_default = test_images[(test_images_y > 0) & (eval_y < 0)]
    false_normal = test_images[(test_images_y < 0) & (eval_y > 0)]
    true_default = test_images[(test_images_y < 0) & (eval_y < 0)]

    if len(true_normal) > 0:
        plt.subplot(221)
        plt.imshow(true_normal[0][:, :, 0], cmap='gray')
        plt.axis('off')
        plt.title("True normal - {}".format(cm[1, 1]))

    if len(false_default) > 0:
        plt.subplot(222)
        plt.imshow(false_default[0][:, :, 0], cmap='gray')
        plt.axis('off')
        plt.title("False default - {}".format(cm[1, 0]))

    if len(false_normal) > 0:
        plt.subplot(223)
        plt.imshow(false_normal[0][:, :, 0], cmap='gray')
        plt.axis('off')
        plt.title("False normal - {}".format(cm[0, 1]))

    if len(true_default) > 0:
        plt.subplot(224)
        plt.imshow(true_default[0][:, :, 0], cmap='gray')
        plt.axis('off')
        plt.title("True default - {}".format(cm[0, 0]))

    plt.show()