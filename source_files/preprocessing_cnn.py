import os
import numpy as np
import tensorflow as tf
import tensorlayer as tl

# Two ways to pass through the tensorlayer network
# tl.utils.predict(...)
# sess.run(network.outputs, feed_dict={input_plh: images})

def run_images_in_cnn(input_fn, input_plh, network, output_dir, hooks=[], verbose=True):
    """Pass all data from dataset (input_fn) into CNN (network) and save predictions in output_dir"""

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        if len(hooks) > 0:
            for hook in hooks:
                hook(sess)

        i = 0
        while True:
            try:
                # Consume input iterator
                fn_and_images = sess.run(input_fn)
                images = fn_and_images[1]
                filenames = list(map(lambda fn: os.path.basename(fn), fn_and_images[0]))

                # Pass data into the network
                predictions = sess.run(network.outputs, feed_dict={input_plh: images})

                # Save prediction
                i = i + len(filenames)
                if output_dir:
                    for filename, prediction in zip(filenames, predictions):
                        np.save(os.path.join(output_dir, "{}.npy".format(filename.decode('utf-8'))), prediction)

                if verbose and i % 100 == 0:
                    tf.logging.info("Image {} has been passed through CNN".format(i))

            except tf.errors.OutOfRangeError:
                break

    return i


if __name__ == "__main__":

    from vgg_network import VGG_Network
    from data_utils import train_input_fn

    class_nbr = 6
    target_w = 224
    input_shape = [None, target_w, target_w, 3]
    batch_size = 2

    # Network
    input_plh = tf.placeholder(tf.float32, shape=input_shape, name="X")
    vgg_model = VGG_Network(include_FC_head=False)
    vgg_network, _ = vgg_model(input_plh)
    network = vgg_network

    # Input function (Dataset)
    input_fn = train_input_fn(class_nbr, target_w, batch_size, keep_fn=True)

    # Hooks
    hook_load_pretrained = lambda sess: vgg_model.load_pretrained(sess)

    # output_dir can be None (predictions not saved)
    run_images_in_cnn(input_fn, input_plh, network, None, hooks=[hook_load_pretrained])