import os
import numpy as np
import tensorflow as tf

# Two ways to pass through the tensorlayer network
# tl.utils.predict(...)
# sess.run(network.outputs, feed_dict={input_plh: images})


def run_images_in_cnn(iterator_op, input_plh, network_outputs, output_dir, hooks=None):
    """
    Passes all data from a batched dataset into a network and saves predictions in output_dir

    :param iterator_op: iterator.get_next() operation (the dataset has to be encoded by batch)
    :param input_plh: placeholder of the CNN input
    :param network_outputs: output tensor of the CNN (.outputs of last tensorlayer layer)
    :param output_dir: where the .npy files are saved
    :param hooks: list of functions to be called after the session is created and the
     variables initialized. The session is passed as a parameter

    :return: the total number of processed images
    """


    tf.logging.info("Running images through the network and saving results in %s" % output_dir)
    if output_dir:
        tf.logging.info("Results will be saved to '%s'" % output_dir)
    else:
        tf.logging.warn("Results will not be saved")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        if hooks:
            for hook in hooks:
                hook(sess)

        i = 0
        try:
            while True:
                # Consume input iterator (returns a batch)
                path_and_images = sess.run(iterator_op)
                filenames = list(map(lambda path: os.path.basename(path.decode('utf-8')), path_and_images[0]))
                images = path_and_images[1]

                predictions = sess.run(network_outputs, feed_dict={input_plh: images})

                # Save prediction
                i += len(filenames)
                if output_dir:
                    for filename, prediction in zip(filenames, predictions):
                        np.save(os.path.join(output_dir, "{}.npy".format(filename)), prediction)

                if i % 100 == 0:
                    tf.logging.debug("Image {} has been passed through CNN".format(i))

        except tf.errors.OutOfRangeError:
            tf.logging.info("Finished iterating over the dataset (%d elements)" % i)

    return i


if __name__ == "__main__":

    # TODO: utiliser les flags, et enregistrer les resultats
    from vgg_network import VGG_Network
    from data_utils import train_img_dataset, get_csv_dataset

    tf.logging.set_verbosity(tf.logging.DEBUG)

    class_nbr = 6
    target_w = 224
    input_shape = [None, target_w, target_w, 3]
    batch_size = 2

    # Network
    input_plh = tf.placeholder(tf.float32, shape=input_shape, name="X")
    vgg_model = VGG_Network(include_FC_head=False)
    vgg_network = vgg_model(input_plh)
    network_outputs = vgg_network.outputs

    # Dataset iterator get_next
    train_csv_path = os.path.join("..", "data/DAGM 2007 - Splitted", str(class_nbr), "{}_files.csv".format("train"))
    input_fn_images = train_img_dataset(class_nbr, target_w)
    input_fn_filenames = get_csv_dataset(train_csv_path, class_nbr).map(lambda fn, label: fn)
    input_fn = tf.data.Dataset.zip((input_fn_filenames, input_fn_images)).batch(batch_size)
    iterator_op = input_fn.make_one_shot_iterator().get_next()

    hook_load_pretrained = lambda sess: vgg_model.load_pretrained(sess)

    # output_dir can be None (predictions not saved)
    run_images_in_cnn(
        iterator_op,
        input_plh,
        network_outputs,
        None,
        hooks=[hook_load_pretrained]
    )

