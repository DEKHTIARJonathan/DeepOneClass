import tensorflow as tf
from vgg_network import VGG_Network
from estimator_svdd_naive import naive_svdd_model_fn, _LoadPreTrainedWeightsVGG
from data_utils import train_cnn_direct_input_fn

if __name__ == "__main__":
    OCClassifier = tf.estimator.Estimator(
        model_fn=naive_svdd_model_fn,
        params={
            "frac_err": 50,
            "n_inputs": 1000,
            "kernel": "linear",
            "rffm_dims": 200,
            "rffm_stddev": 25,
            "learning_rate": 0.1,
            "input_size": 25088
        },
        model_dir='.'
    )

    net = VGG_Network(include_FC_head=False)

    OCClassifier.train(
        input_fn=lambda: train_cnn_direct_input_fn(6, '../data/DAGM 2007 - Splitted', 3, net),
        hooks=[_LoadPreTrainedWeightsVGG(net)]
    )