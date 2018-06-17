import os
import numpy as np
import tensorflow as tf

from flags import FLAGS
from vgg_network import VGG_Network
from data_utils import train_img_dataset, test_img_dataset, get_csv_dataset

# Two ways to pass through the tensorlayer network
# tl.utils.predict(...)
# sess.run(network.outputs, feed_dict={input_plh: images})


def run_images_in_cnn(iterator_op, input_plh, network_outputs, output_path, hooks=None):
    """
    Passes all data from a batched dataset into a network and saves predictions in output_dir

    :param iterator_op: iterator.get_next() operation (the dataset has to be encoded by batch)
    :param input_plh: placeholder of the CNN input
    :param network_outputs: output tensor of the CNN (.outputs of last tensorlayer layer)
    :param output_path: where the .tfrecord files are saved
    :param hooks: list of functions to be called after the session is created and the
     variables initialized. The session is passed as a parameter

    :return: the total number of processed images
    """

    with tf.Session() as sess, tf.python_io.TFRecordWriter(output_path) as writer:
        sess.run(tf.global_variables_initializer())

        if hooks:
            for hook in hooks:
                hook(sess)

        i = 0
        try:
            while True:
                # Consume input iterator (returns a batch)
                [paths, labels, images] = sess.run(iterator_op)
                filenames = list(map(lambda p: os.path.dirname(p.decode('utf-8')), paths))

                predictions = sess.run(network_outputs, feed_dict={input_plh: images})
                predictions = predictions.astype(np.float32)

                # Save prediction
                i += len(filenames)

                if output_path:
                    for filename, prediction, label in zip(filenames, predictions, labels):
                        example = tf.train.Example(
                            features=tf.train.Features(
                                feature={
                                    'filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename.encode('ascii')])),
                                    'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(prediction.tostring())])),
                                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                                    'ndims': tf.train.Feature(int64_list=tf.train.Int64List(value=[prediction.shape[0]]))
                                }))

                        writer.write(example.SerializeToString())

                if i % 100 == 0:
                    tf.logging.debug("Image {} has been passed through CNN".format(i))

        except tf.errors.OutOfRangeError:
            tf.logging.info("Finished iterating over the dataset (%d elements)" % i)

    return i


def main(argv=None):
    input_shape = [None, FLAGS.target_width, FLAGS.target_width, 3]

    if not os.path.exists(FLAGS.cnn_output_dir):
        os.makedirs(FLAGS.cnn_output_dir)

    # Network
    input_plh = tf.placeholder(tf.float32, shape=input_shape, name="X")
    vgg_model = VGG_Network(include_FC_head=False)
    vgg_network = vgg_model(input_plh)
    network_outputs = vgg_network.outputs

    hook_load_pretrained = lambda sess: vgg_model.load_pretrained(sess)

    # Train Dataset
    train_csv_path = os.path.join(FLAGS.data_dir, str(FLAGS.class_nbr), "{}_files.csv".format("train"))
    input_fn_images = train_img_dataset(FLAGS.class_nbr, FLAGS.target_width)
    input_fn_filenames_labels = get_csv_dataset(train_csv_path, FLAGS.class_nbr)
    input_fn = tf.data.Dataset.zip((input_fn_filenames_labels, input_fn_images)).map(lambda fl, i: (fl[0], fl[1], i)).batch(FLAGS.batch_size)
    iterator_op = input_fn.make_one_shot_iterator().get_next()

    # Target directory
    vgg16_train_output = os.path.join(FLAGS.cnn_output_dir, str(FLAGS.class_nbr), 'train.tfrecord')
    if not os.path.exists(os.path.dirname(vgg16_train_output)):
        os.makedirs(os.path.dirname(vgg16_train_output))

    run_images_in_cnn(
        iterator_op,
        input_plh,
        network_outputs,
        vgg16_train_output,
        hooks=[hook_load_pretrained]
    )

    # Test Dataset
    test_csv_path = os.path.join(FLAGS.data_dir, str(FLAGS.class_nbr), "{}_files.csv".format("test"))
    input_fn_images = test_img_dataset(FLAGS.class_nbr, FLAGS.target_width).map(lambda img, label: img)
    input_fn_filenames_labels = get_csv_dataset(test_csv_path, FLAGS.class_nbr)
    input_fn = tf.data.Dataset.zip((input_fn_filenames_labels, input_fn_images)).map(lambda fl, i: (fl[0], fl[1], i)).batch(FLAGS.batch_size)
    iterator_op = input_fn.make_one_shot_iterator().get_next()

    # Target directory
    vgg16_test_output = os.path.join(FLAGS.cnn_output_dir, str(FLAGS.class_nbr), 'test.tfrecord')
    if not os.path.exists(os.path.dirname(vgg16_test_output)):
        os.makedirs(os.path.dirname(vgg16_test_output))

    run_images_in_cnn(
        iterator_op,
        input_plh,
        network_outputs,
        vgg16_test_output,
        hooks=[hook_load_pretrained]
    )

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.DEBUG)
    tf.app.run()