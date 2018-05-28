import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from functools import partial


def center_dataset_values(dataset, min_v=0, max_v=255):
    return dataset.map(lambda img: img - (max_v - min_v) / 2)


def scale_dataset_values(dataset, max_v=255):
    return dataset.map(lambda img: img / max_v)


def str_filenames_gen(dir):
    """Genrator for filenames as strings"""
    for fn in glob.glob(os.path.join(dir, "*.png")):
        yield os.path.join(dir, str(fn))


def load_and_transf_img(filename, target_w):
    """Image target_w x target_w x 3 as floats 0 - 255"""
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


def get_dataset(dir, target_w):
    dataset = tf.data.Dataset.from_generator(
        generator=partial(str_filenames_gen, dir),
        output_types=tf.string,
    )
    dataset = dataset.map(lambda fn: (fn, load_and_transf_img(fn, target_w)))

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


def train_input_fn(class_nbr, target_w, batch_size, keep_fn=False):
    """Return iterator on train dataset"""
    dataset = get_train_dataset(class_nbr, "../data/DAGM 2007 - Splitted", target_w)
    if not keep_fn:
        dataset = dataset.map(lambda fn, img: img)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


def test_input_fn(class_nbr, target_w, batch_size, keep_fn=False):
    """Return iterator on test dataset"""
    dataset = get_test_dataset(class_nbr, "../data/DAGM 2007 - Splitted", target_w)
    if not keep_fn:
        dataset = dataset.map(lambda fn, img: img)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


def train_cnn_input_fn(class_nbr, cnn_output_dir, batch_size, keep_fn=False):
    """Return iterator on train dataset"""
    dataset = tf.data.Dataset.from_generator(
        generator=partial(str_filenames_gen, "../data/DAGM 2007 - Splitted"),
        output_types=tf.string,
    )
    dataset = dataset.map(lambda path: tf.py_func(lambda p: (p, os.path.basename(p)), [path], [tf.string, tf.string]))
    dataset = dataset.map(lambda path, fn: tf.py_func(lambda p, f, cnn, cl: (p, f, np.load("{}/{}/train/{}".format(
        cnn,
        cl,
        f
    ))), [path, fn, cnn_output_dir, class_nbr], [tf.string, tf.string, tf.float32]))
    if not keep_fn:
        dataset = dataset.map(lambda path, fn, img: img)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


if __name__ == '__main__':
    sess = tf.Session()
    img = sess.run(load_and_transf_img('../data/DAGM 2007 - Splitted/6/test/001.png', 224))
    print(img.astype(np.uint8))
    plt.imshow(img.astype(np.uint8))
    plt.show()

