import os
import numpy as np
import tensorflow as tf
import tensorlayer as tl

# Two ways to pass through the tensorlayer network
# tl.utils.predict(...)
# sess.run(network.outputs, feed_dict={input_plh: images})


def run_images_in_cnn(iterator_op, input_plh, network_output, output_dir, hooks=None):
    """
    Passes all data from a batched dataset into a network and saves predictions in output_dir

    :param iterator_op: iterator.get_next() operation (the dataset has to be by batch)
    :param input_plh: placeholder of the CNN input
    :param network_output: output tensor of the CNN
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
                fn_and_images = sess.run(iterator_op)
                images = fn_and_images[1]
                filenames = list(map(lambda fn: os.path.basename(fn), fn_and_images[0]))

                predictions = sess.run(network_output, feed_dict={input_plh: images})

                # Save prediction
                i += len(filenames)
                if output_dir:
                    for filename, prediction in zip(filenames, predictions):
                        np.save(os.path.join(output_dir, "{}.npy".format(filename.decode('utf-8'))), prediction)

                if i % 100 == 0:
                    tf.logging.debug("Image {} has been passed through CNN".format(i))

        except tf.errors.OutOfRangeError:
            tf.logging.info("Finished iterating over the dataset (%d elements)" % i)

    return i


if __name__ == "__main__":

    from vgg_network import VGG_Network
    from data_utils import train_input_fn

    tf.logging.set_verbosity(tf.logging.DEBUG)

    class_nbr = 6
    target_w = 224
    input_shape = [None, target_w, target_w, 3]
    batch_size = 2

    # Network
    input_plh = tf.placeholder(tf.float32, shape=input_shape, name="X")
    vgg_model = VGG_Network(include_FC_head=False)
    vgg_network, _ = vgg_model(input_plh)
    network = vgg_network

    # Dataset iterator get_next
    iter_op = train_input_fn(class_nbr, target_w, batch_size, keep_fn=True)

    hook_load_pretrained = lambda sess: vgg_model.load_pretrained(sess)

    # output_dir can be None (predictions not saved)
    run_images_in_cnn(iter_op, input_plh, network.outputs, None, hooks=[hook_load_pretrained])

