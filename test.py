import tensorflow as tf
import tensorlayer as tl

from source_files.vgg_network import VGG_Network

if __name__ == "__main__":

    vgg_net = VGG_Network(include_FC_head=False)

    a = tf.placeholder(tf.float32, [None, 224, 224, 3])
    b = tf.placeholder(tf.float32, [None, 224, 224, 3])

    out_a = vgg_net(a, reuse=False)
    out_b = vgg_net(b, reuse=True)