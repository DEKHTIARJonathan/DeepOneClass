import tensorflow as tf
from vgg_network import VGG_Network
from estimator_svdd_naive import naive_svdd_model_fn, _LoadPreTrainedWeightsVGG, OCClassifier
from data_utils import *
import numpy as np

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.DEBUG)

    classifier = OCClassifier(
        c=5,
        kernel="linear",
        rffm_dims=10,
        rffm_stddev=5,
        learning_rate=0.1,
        model_dir=None
    )

    net1 = VGG_Network(include_FC_head=False)

    def get_dataset(net, reuse=False):
        dataset = get_train_dataset(6, '../data/DAGM 2007 - Splitted', 224).batch(10)
        dataset = run_dataset_trough_network(dataset, net, reuse)
        return dataset

    classifier.train(
        input_fn=lambda: get_dataset(net1, False).repeat(),
        hooks=[_LoadPreTrainedWeightsVGG(net1)],
        max_steps=1000
    )

    # out = classifier.predict(
    #     input_fn=lambda: get_dataset(net1, False),
    #     hooks=[_LoadPreTrainedWeightsVGG(net1)]
    # )
    #
    # predictions = np.asarray(list(map(lambda p: p["predicted_classes"], out))).astype(np.int32)
    # print(predictions)

    print('Center : {}'.format(classifier.get_variable_value("Center")))
    print('Radius : {}'.format(classifier.get_variable_value("Radius")))