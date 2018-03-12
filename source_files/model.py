import os
import tensorflow as tf
import tensorlayer as tl

from source_files.vgg_network import VGG_Network

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

        with tf.name_scope("image_processing"):
            self.resized_input = tf.image.resize_images(
                images = self.input_plh,
                size   = (224, 224),
                method = tf.image.ResizeMethod.BILINEAR
            )

            self.resized_input = tf.image.grayscale_to_rgb(images=self.resized_input)

        #############################################
        ##### ======== Model Creation ========= #####
        #############################################

        self.vgg_model = VGG_Network(include_FC_head=False, flatten_output=False)

        self.vgg_net, _ = self.vgg_model(
            inputs = self.resized_input,
            reuse  = False
        )

        # print("VGG Output Size:", self.vgg_net.outputs.get_shape()) (None, 7, 7, 512)

        #############################################
        ##### ====== Model Initialization ===== #####
        #############################################

        self.vgg_net.print_params(False)

        tl.layers.initialize_global_variables(self.sess)

        self.vgg_model.load_pretrained(self.sess) # Load pretrained model


