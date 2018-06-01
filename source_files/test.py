import tensorflow as tf
from vgg_network import VGG_Network
from estimator_svdd_naive import naive_svdd_model_fn, _LoadPreTrainedWeightsVGG, OCClassifier
from data_utils import *
import numpy as np

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.DEBUG)

    classifier = OCClassifier(
        frac_err=0.5,
        n_inputs=5,
        kernel="linear",
        learning_rate=1,
        model_dir='estimator-ckpt'
    )

    OCClassifier()

    net1 = VGG_Network(include_FC_head=False)

    def get_dataset(net, reuse=False):
        dataset = get_train_dataset(6, '../data/sanity_check', 224).batch(1)
        dataset = run_dataset_trough_network(dataset, net, reuse)
        return dataset

    classifier.train(
        input_fn=lambda: get_dataset(net1, None).repeat().batch(1),
        hooks=[_LoadPreTrainedWeightsVGG(net1)],
        max_steps=100
    )
    #
    # out = classifier.predict(
    #     input_fn=lambda: get_dataset(net1, tf.AUTO_REUSE).batch(1),
    #     hooks=[_LoadPreTrainedWeightsVGG(net1)]
    # )
    #
    # predictions = np.asarray(list(map(lambda p: p["predicted_classes"], out))).astype(np.int32)
    # print(predictions)