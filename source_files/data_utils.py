import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from functools import partial

class _LoadPreTrainedWeights(tf.train.SessionRunHook):
    def __init__(self, model, weights_path='../weights/vgg16_weights.npz'):
        self._model = model
        self._weights_path = weights_path

    def after_create_session(self, session, coord):
        session.run(self._ops)
        tf.logging.info("model pretrained weights are assigned")

    def begin(self):
        self._ops = self._model.get_ops_load_pretrained(self._weights_path)


############################################
#
# Images transformation
#
############################################

def center_dataset_values(dataset, min_v=0, max_v=255):
    return dataset.map(lambda img: img - (max_v - min_v) / 2)


def scale_dataset_values(dataset, max_v=255):
    return dataset.map(lambda img: img / max_v)


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


############################################
#
# Images generator
#
############################################

def filename_generator(dir):
    """Generator filenames"""
    return glob.iglob(os.path.join(dir, "*.png"))


def csv_generator(csv_path, class_nbr):
    """Generator csv files"""
    df = pd.read_csv(csv_path, sep=",")
    for index, row in df.iterrows():
        img_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                '..', 'data/DAGM 2007 - Splitted', str(class_nbr), row['target'].replace("\\", "/"))
        label = row['is_healthy']
        yield (img_path, label)


############################################
#
# Base Tensorflow Datasets
#
############################################

# Generator from filenames

def get_fn_dataset(dir):
    """Return Tensorflow Dataset from filename generator"""
    dataset = tf.data.Dataset.from_generator(
        generator=partial(filename_generator, dir),
        output_types=tf.string,
    )

    return dataset


def get_fn_train_dataset(class_nbr, data_dir):
    return get_fn_dataset(
        dir=os.path.join(data_dir, str(class_nbr), 'train')
    )


def get_fn_test_dataset(class_nbr, data_dir):
    return get_fn_dataset(
        dir=os.path.join(data_dir, str(class_nbr), 'test')
    )

# Generator from csv files

def get_csv_dataset(csv_path, class_nbr):
    """Return Tensorflow Dataset from csv generator"""
    dataset = tf.data.Dataset.from_generator(
        generator=lambda: csv_generator(csv_path, class_nbr),
        output_types=(tf.string, tf.int32),
    )

    return dataset


############################################
#
# Pipelines for images
#
############################################

def _img_dataset(class_nbr, target_w, type="train", keep_label=False):
    """Return ready Dataset to turn into iterator"""

    # Get filenames
    csv_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            "..", "data/DAGM 2007 - Splitted", str(class_nbr), "{}_files.csv".format(type))
    dataset = get_csv_dataset(csv_path, class_nbr)

    # Open and transform original images
    dataset = dataset.map(lambda fn, label: (load_and_transf_img(fn, target_w), label))

    if not keep_label:
        dataset = dataset.map(lambda img, label: img)

    tf.logging.debug(' Created a {} dataset from csv file {}'.format(type, csv_path))
    tf.logging.debug('     Output shapes : %s' % str(dataset.output_shapes))
    tf.logging.debug('     Output types : %s\n' % str(dataset.output_types))

    return dataset


def train_img_dataset(class_nbr, target_w):
    """Return Dataset on train dataset: get_next() -> ([image * batch_size])"""
    return _img_dataset(class_nbr, target_w, type="train", keep_label=False)


def test_img_dataset(class_nbr, target_w):
    """Return Dataset on train dataset: get_next() -> ([image * batch_size], [label * batch_size])"""
    return _img_dataset(class_nbr, target_w, type="test", keep_label=True)


############################################
#
# Pipelines for already encoded images
#
############################################

def _cached_features_dataset(class_nbr, cnn_output_dir, type="train", keep_label=False):
    """Return ready Dataset to turn into iterator"""

    # Get filenames path
    csv_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            "..", "data/DAGM 2007 - Splitted", str(class_nbr), "{}_files.csv".format(type))
    dataset = get_csv_dataset(csv_path, class_nbr)

    # Get exact filename
    dataset = dataset.map(lambda path, label: tf.py_func(
        lambda p, l: (os.path.basename(p), l),
        [path, label],
        [tf.string, tf.int32]
    ))

    # Open encoded image
    dataset = dataset.map(lambda fn, label: tf.py_func(
        lambda cnn, cl, t, fn, l: (np.load("{}/{}/{}/{}.npy".format(
            cnn.decode('utf-8'),
            cl,
            t.decode('utf-8'),
            fn.decode('utf-8')
        )), l),
        [cnn_output_dir, class_nbr, type, fn, label],
        [tf.float32, tf.int32]
    ))

    # Read shape of read .npy files
    first_file = glob.glob(os.path.join(cnn_output_dir, str(class_nbr), type, "*.npy"))[0]
    shape = np.load(first_file).shape
    dataset = dataset.map(lambda img, label: (tf.reshape(img, shape), label))

    if not keep_label:
        dataset = dataset.map(lambda img, label: img)

    tf.logging.debug(' Created a {} dataset from csv file {}'.format(type, csv_path))
    tf.logging.debug('     Output shapes : %s' % str(dataset.output_shapes))
    tf.logging.debug('     Output types : %s\n' % str(dataset.output_types))

    return dataset


def train_cached_features_dataset(class_nbr, cnn_output_dir):
    """Return Dataset on train dataset: get_next() -> ([image * batch_size])"""
    return _cached_features_dataset(
        class_nbr,
        cnn_output_dir,
        type="train",
        keep_label=False
    )


def test_cached_features_dataset(class_nbr, cnn_output_dir):
    """Return Dataset on test dataset: get_next() -> ([image * batch_size], [label * batch_size])"""
    return _cached_features_dataset(
        class_nbr,
        cnn_output_dir,
        type="test",
        keep_label=True,
    )


############################################
#
# Pipelines for full direct pass through CNN
#
############################################

def run_dataset_through_network(dataset, network, reuse=False):
    return dataset.map(lambda img: (network(img, reuse=reuse)).outputs)



############################################
#
# Sanity checks
#
############################################

if __name__ == '__main__':

    tf.logging.set_verbosity(tf.logging.DEBUG)

    CLASS_NBR = 6
    CNN_OUTPUT_DIR = os.path.join("..", "tmp", "cnn_output", "VGG16")
    TARGET_W = 224

    with tf.Session() as sess:

        print("load_and_transf_img")
        img = sess.run(load_and_transf_img(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                        '../data/DAGM 2007 - Splitted/6/test/001.png'), 224))
        assert isinstance(img, np.ndarray)

        print("get_csv_dataset")
        dataset = get_csv_dataset(
            os.path.join(os.path.dirname(os.path.realpath(__file__)),
                         '..', 'data/DAGM 2007 - Splitted', str(CLASS_NBR), 'train_files.csv'),
            CLASS_NBR
        ).batch(1)
        res = dataset.make_one_shot_iterator().get_next()
        assert len(res) == 2

        print("train_img_dataset")
        dataset = train_img_dataset(CLASS_NBR, TARGET_W)
        dataset = dataset.batch(1)
        iterator = dataset.make_one_shot_iterator().get_next()
        t = sess.run(iterator)
        assert len(t) == 1
        assert isinstance(img, np.ndarray)

        print("test_img_dataset")
        dataset = test_img_dataset(CLASS_NBR, TARGET_W)
        dataset = dataset.batch(1)
        iterator = dataset.make_one_shot_iterator().get_next()
        t = sess.run(iterator)
        assert len(t) == 2
        assert isinstance(t[0], np.ndarray)
        assert isinstance(t[1], np.ndarray)

        print("train_cached_features_dataset")
        dataset = train_cached_features_dataset(CLASS_NBR, CNN_OUTPUT_DIR)
        dataset = dataset.batch(2)
        iterator = dataset.make_one_shot_iterator().get_next()
        t = sess.run(iterator)

        print("test_cached_features_dataset")
        dataset = test_cached_features_dataset(CLASS_NBR, CNN_OUTPUT_DIR)
        dataset = dataset.batch(2)
        iterator = dataset.make_one_shot_iterator().get_next()
        t = sess.run(iterator)

    print("Sanity checks: OK")
