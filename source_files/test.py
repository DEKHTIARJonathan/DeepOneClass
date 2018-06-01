import tensorflow as tf
from vgg_network import VGG_Network
from estimator_svdd_naive import naive_svdd_model_fn, _LoadPreTrainedWeightsVGG, OCClassifier
from data_utils import train_cnn_direct_input_fn

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.DEBUG)

    classifier = OCClassifier(
        frac_err=0.0001,
        n_inputs=5,
        kernel="linear",
        learning_rate=1,
        model_dir=None
    )

    OCClassifier()

    net = VGG_Network(include_FC_head=False)

    classifier.train(
        input_fn=lambda: train_cnn_direct_input_fn(class_nbr=6,
                                                   data_dir='../data/sanity_check',
                                                   batch_size=1,
                                                   network=net),
        hooks=[_LoadPreTrainedWeightsVGG(net)],
        max_steps=1000
    )