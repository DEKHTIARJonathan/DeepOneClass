import tensorflow as tf
from vgg_network import VGG_Network
from estimator_svdd_naive import _LoadPreTrainedWeightsVGG, OCClassifier
from data_utils import *
import numpy as np

class_nbr = 6
batch_size = 2

# Cached .npy -> estimator
if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.DEBUG)

    classifier = OCClassifier(
        c=3,
        kernel="linear",
        learning_rate=1,
        model_dir=None
    )

    classifier.train(
        input_fn=lambda: train_cnn_input_fn(class_nbr, '../tmp/cnn_output/VGG16').batch(batch_size).repeat(),
        steps=400
    )

    out = classifier.predict(
        input_fn=lambda: test_cnn_input_fn(6, '../tmp/cnn_output/VGG16').batch(batch_size),
    )

    predictions = np.asarray(list(map(lambda p: p["predicted_classes"], out))).astype(np.int32)
    print(predictions)
    print(np.mean(predictions))
    print(np.sum(predictions))
    print(classifier.get_variable_value("Radius"))