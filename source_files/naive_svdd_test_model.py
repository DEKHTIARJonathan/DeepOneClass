import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorlayer as tl
from model import OneClassCNN
from naive_svdd import SvddLayer
from sklearn.metrics import accuracy_score, confusion_matrix
from image_ops import get_image

np.random.seed(3)

dataset_number  = 6
n_max_normal    = 100                                # number max of normal images to train/test on
n_max_def       = 50                                 # number max of def images to test on
target_w        = 256
input_shape     = [None, target_w, target_w, 1]
batch_size      = 10
n_epoch         = 1000
kernel          = "rbf"
rffm_dims       = 200
rffm_stddev     = 25

data_6_path             = os.path.join("..", "data", "Class{}".format(dataset_number))
data_6_path_files       = glob(os.path.join(data_6_path, "*.png"))[:n_max_normal]

data_6_def_path         = os.path.join("..", "data", "Class{}_def".format(dataset_number))
data_6_def_path_files   = glob(os.path.join(data_6_def_path, "*.png"))[:n_max_def]

np.random.shuffle(data_6_path_files)

l                 = int(len(data_6_path_files) / 2)
train_filenames   = data_6_path_files[:l]
test_filenames    = np.concatenate((data_6_path_files[l:], data_6_def_path_files), axis=0)

# Train set (only normal images)
train_images = np.array(list(map(lambda f: get_image(f, target_w=target_w), train_filenames)))
assert(train_images.shape == (len(train_filenames), target_w, target_w, 1))

# Test set (both normal and def images)
test_images     = np.array(list(map(lambda f: get_image(f, target_w=target_w), test_filenames)))
test_images_y   = np.concatenate((np.ones(l), -1 * np.ones(len(test_filenames) - l)), axis=0)
assert(test_images.shape == (len(test_filenames), target_w, target_w, 1))
assert(test_images.shape[0] == test_images_y.shape[0])

input_plh = tf.placeholder(tf.float32, shape=input_shape, name="X")

with tf.Session() as sess:

    layer, _ = OneClassCNN(sess)._get_model(input_plh, reuse=True)
    layer = tl.layers.FlattenLayer(layer, name="flatten")
    layer = SvddLayer(layer, map=kernel, rffm_dims=rffm_dims, rffm_stddev=rffm_stddev)

    sess.run(tf.global_variables_initializer())

    layer.train(sess, input_plh, train_images, n_epoch=n_epoch, batch_size=batch_size)
    eval_y = sess.run(layer.outputs, feed_dict={input_plh: test_images})

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