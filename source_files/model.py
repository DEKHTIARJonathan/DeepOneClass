import os
import tensorflow as tf
import tensorlayer as tl

from .custom_layers import conv_module

class OneClassCNN(object):

    def __init__(self, session, model_name="model", ckpt_dir="checkpoints", samples_dir="samples", log_dir="logs"):

        #############################################
        ##### ===== Saving the Parameters ===== #####
        #############################################

        self.sess       = session
        self.model_name = model_name

        #######################################################################
        #                      CREATING THE DIRECTORIES                       #
        #######################################################################

        self.ckpt_dir    = os.path.join(ckpt_dir, self.model_name)
        self.samples_dir = os.path.join(samples_dir, self.model_name)
        self.log_dir     = os.path.join(log_dir, self.model_name)

        tl.files.exists_or_mkdir(ckpt_dir)
        tl.files.exists_or_mkdir(samples_dir)
        tl.files.exists_or_mkdir(log_dir)

        tl.files.exists_or_mkdir(self.ckpt_dir)
        tl.files.exists_or_mkdir(self.samples_dir)
        tl.files.exists_or_mkdir(self.log_dir)

        #############################################
        ##### ===== PlaceHolders Creation ===== #####
        #############################################

        with tf.name_scope("placeholders"):
            self.input_plh         = tf.placeholder(tf.float32, shape = [None, 256, 256, 1], name="input_plh")

        #############################################
        ##### ======== Model Creation ========= #####
        #############################################

        self.training_model, self.training_logits = self._get_model(self.input_plh)

        self.eval_model,     self.eval_logits     = self._get_model(self.input_plh, is_train=False, reuse=True)

        #############################################
        ##### ====== Model Initialization ===== #####
        #############################################

        self.training_model.print_params(False)

        tl.layers.initialize_global_variables(self.sess)



        ################################################################################################################
        ################################################################################################################
        ################################################################################################################
        ################################################################################################################


    def _get_model(self, inputs, is_train=True, reuse=False):

        xavier_initilizer  = tf.contrib.layers.xavier_initializer(uniform=True)
        normal_initializer = tf.random_normal_initializer(mean=1., stddev=0.02)

        with tf.variable_scope("cnn_model", reuse=reuse):
            tl.layers.set_name_reuse(reuse)

            # Input Layers
            layer = tl.layers.InputLayer(inputs, name='input')

            ### Input Shape: [None, 256, 256, 1]

            with tf.variable_scope('h1', reuse=reuse):
                tl.layers.set_name_reuse(reuse)

                layer, _ = conv_module(
                    input           = layer,
                    n_out_channel   = 32,
                    filter_size     = (5, 5),
                    strides         = (2, 2),
                    padding         = "SAME",
                    conv_init       = xavier_initilizer,
                    batch_norm_init = normal_initializer,
                    is_train        = is_train,
                    use_batchnorm   = True,
                    activation_fn   = "PReLU"
                )

                tf.logging.debug("H1 Layer Shape: %s " % layer.outputs.shape)
                ### Output Shape: [None, 128, 128, 32]

            with tf.variable_scope('h2', reuse=reuse):
                tl.layers.set_name_reuse(reuse)

                layer, _ = conv_module(
                    input           = layer,
                    n_out_channel   = 64,
                    filter_size     = (5, 5),
                    strides         = (2, 2),
                    padding         = "SAME",
                    conv_init       = xavier_initilizer,
                    batch_norm_init = normal_initializer,
                    is_train        = is_train,
                    use_batchnorm   = True,
                    activation_fn   = "PReLU"
                )

                tf.logging.debug("H2 Layer Shape: %s " % layer.outputs.shape)
                ### Output Shape: [None, 64, 64, 64]

            with tf.variable_scope('h3', reuse=reuse):
                tl.layers.set_name_reuse(reuse)

                layer, _ = conv_module(
                    input           = layer,
                    n_out_channel   = 128,
                    filter_size     = (5, 5),
                    strides         = (2, 2),
                    padding         = "SAME",
                    conv_init       = xavier_initilizer,
                    batch_norm_init = normal_initializer,
                    is_train        = is_train,
                    use_batchnorm   = True,
                    activation_fn   = "PReLU"
                )

                tf.logging.debug("H3 Layer Shape: %s " % layer.outputs.shape)
                ### Output Shape: [None, 32, 32, 128]

            with tf.variable_scope('h4', reuse=reuse):
                tl.layers.set_name_reuse(reuse)

                layer, _ = conv_module(
                    input           = layer,
                    n_out_channel   = 256,
                    filter_size     = (5, 5),
                    strides         = (2, 2),
                    padding         = "SAME",
                    conv_init       = xavier_initilizer,
                    batch_norm_init = normal_initializer,
                    is_train        = is_train,
                    use_batchnorm   = True,
                    activation_fn   = "PReLU"
                )

                tf.logging.debug("H4 Layer Shape: %s " % layer.outputs.shape)
                ### Output Shape: [None, 16, 16, 256]

            with tf.variable_scope('h5', reuse=reuse):
                tl.layers.set_name_reuse(reuse)

                layer, _ = conv_module(
                    input           = layer,
                    n_out_channel   = 1,
                    filter_size     = (16, 16),
                    strides         = (2, 2),
                    padding         = "VALID",
                    conv_init       = xavier_initilizer,
                    batch_norm_init = normal_initializer,
                    is_train        = is_train,
                    use_batchnorm   = True,
                    activation_fn   = "sigmoid"
                )

                tf.logging.debug("H5 Layer Shape: %s " % layer.outputs.shape)
                ### Output Shape: [None, 1, 1, 1]

                logits = _

        return layer, logits


