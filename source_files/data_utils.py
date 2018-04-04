import os

import tensorflow as tf

from functools import partial

# Genrator for filenames as strings
def str_filenames_gen(dir):
    for fn in os.listdir(dir):
        yield os.path.join(dir, str(fn))


# Image target_w x target_w x 3 as floats 0 - 255
def load_and_transf_img(filename, target_w):
    file = tf.read_file(filename)
    img = tf.image.decode_png(
        contents=file,
        channels=3,
        dtype=tf.uint8
    )
    img = tf.image.resize_images(
        images=img,
        size=[target_w, target_w],
        method=tf.image.ResizeMethod.BILINEAR
    )
    return img

# Dataset of
def get_dataset(dir, target_w):
    dataset = tf.data.Dataset.from_generator(
        generator=partial(str_filenames_gen, dir),
        output_types=tf.string,
    )\
        .map(lambda fn: load_and_transf_img(fn, target_w))

    tf.logging.debug(' Created a dataset from directory %s' % dir)
    tf.logging.debug('     Output shapes : %s' % str(dataset.output_shapes))
    tf.logging.debug('     Output types : %s\n' % str(dataset.output_types))


    return dataset

def get_train_dataset(class_nbr, data_dir, target_w):
    return get_dataset(
        dir=os.path.join(data_dir, str(class_nbr), 'train'),
        target_w=target_w
    )

def get_test_dataset(class_nbr, data_dir, target_w):
    return get_dataset(
        dir=os.path.join(data_dir, str(class_nbr), 'test'),
        target_w=target_w
    )

def center_dataset_values(dataset, min_v=0, max_v=255):
    return dataset.map(lambda img: img - (max_v - min_v) / 2)

def scale_dataset_values(dataset, max_v=255):
    return dataset.map(lambda img: img / max_v)

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    sess = tf.Session()
    img = sess.run(load_and_transf_img('../data/DAGM 2007 - Splitted/6/test/001.png', 224))
    print(img.astype(np.uint8))
    plt.imshow(img.astype(np.uint8))
    plt.show()

