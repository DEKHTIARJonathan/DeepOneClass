import tensorflow as tf
from vgg_network import VGG_Network
from estimator_svdd import _LoadPreTrainedWeightsVGG, OCClassifier
from data_utils import *
import numpy as np

class_nbr = 6
batch_size = 2

def get_train_dataset(net, reuse=False):
    dataset = train_input_fn(class_nbr, 224).batch(batch_size)
    dataset = run_dataset_through_network(dataset, net, reuse)
    return dataset.repeat()

def get_test_dataset(net, reuse=False):
    dataset = test_input_fn(class_nbr, 224).batch(batch_size)
    dataset = dataset.map(lambda img, label: img)
    dataset = run_dataset_through_network(dataset, net, reuse)
    return dataset

# Direct image -> CNN -> estimator
if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.DEBUG)

    classifier = OCClassifier(
        c=3,
        kernel="linear",
        learning_rate=1,
        model_dir=None
    )

    net = VGG_Network(include_FC_head=False)

    classifier.train(
        input_fn=lambda: get_train_dataset(net, False),
        hooks=[_LoadPreTrainedWeightsVGG(net)],
        steps=400
    )

    out = classifier.predict(
        input_fn=lambda: get_test_dataset(net, False),
        hooks=[_LoadPreTrainedWeightsVGG(net)]
    )

    predictions = np.asarray(list(map(lambda p: p["predicted_classes"], out))).astype(np.int32)
    print(predictions)
    print(np.mean(predictions))
    print(np.sum(predictions))
    print(classifier.get_variable_value("Radius"))