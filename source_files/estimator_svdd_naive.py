import tensorflow as tf
from vgg_network import VGG_Network
import numpy as np

class _LoadPreTrainedWeightsVGG(tf.train.SessionRunHook):
    def __init__(self, vgg_model):
        self._vgg_model = vgg_model

    def after_create_session(self, session, coord):
        session.run(self.ops)
        tf.logging.info("vgg_model pretrained weights are assigned")

    def begin(self):
        self.ops = self._vgg_model.get_ops_load_pretrained('weights/vgg16_weights.npz')

def naive_svdd_model_fn(features, labels, mode, params):
    """
    Naive SVDD
    :param features: data passed to fit function (train/predict/eval)
    :param labels: labels passed (used only for evaluate)
    :param mode: train/predict/eval
    :param params: dict of additional params
    :return: tf.estimator.EstimatorSpec
    """

    # Few details on inputs
    # print(tf.shape(features)) # 4 because -> (?, 256, 256, 3)
    # print(features) # Tensor("IteratorGetNext:0", shape=(?, 256, 256, 3), [...])
    # print(labels) # None if iterator only returns single image and not a tuple

    #
    # with tf.name_scope("VGG"):
    #     vgg_model = VGG_Network(include_FC_head=False)
    #     vgg_network, _ = vgg_model(features)
    #     features_map = vgg_network.outputs
    #     features_map_size = int(features_map.get_shape()[1])

    if "input_size" in params and params["input_size"]:
        input_size = params["input_size"]
    elif features.get_shape().dims is not None:
        input_size = features.get_shape().as_list()[1]
    else:
        raise Exception("Input size is unknown, either not given via params or features shape is None")

    features.set_shape((None, input_size))

    if params["kernel"] == "linear":
        out_size = input_size
        mapped_inputs = features
    elif params["kernel"] in ["rffm", "rbf"]:
        out_size = params["rffm_dims"] if "rffm_dims" in params else input_size
        kernel_mapper = tf.contrib.kernel_methods.RandomFourierFeatureMapper(
            input_dim=input_size,
            output_dim=out_size,
            stddev=params["rffm_stddev"],
            name="rffm"
        )
        mapped_inputs = kernel_mapper.map(features)
    else:
        raise ValueError("Map function {} not implemented.".format(params["kernel"]))

    # Model variables
    R = tf.Variable(tf.random_normal([], mean=10), dtype=tf.float32, name="Radius")
    a = tf.Variable(tf.random_normal([out_size], mean=5), dtype=tf.float32, name="Center")

    C = tf.constant(params["c"], dtype=tf.float32)

    # Loss
    constraint = tf.square(R) - tf.square(tf.norm(mapped_inputs - a, axis=1))
    loss = tf.square(R) - C * tf.reduce_sum(tf.minimum(constraint, 0.0))
    loss = tf.reduce_sum(loss)
    tf.summary.scalar('loss', loss)

    predicted_classes = tf.sign(constraint)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'predicted_scores': constraint,
            'predicted_classes': predicted_classes
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    elif mode == tf.estimator.ModeKeys.EVAL:
        # Compute evaluation metrics.
        accuracy = tf.metrics.accuracy(labels=labels,
                                       predictions=predicted_classes,
                                       name='acc_op')
        metrics = {'accuracy': accuracy}
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    elif mode == tf.estimator.ModeKeys.TRAIN:
        lr = params["learning_rate"] if "learning_rate" in params else 0.1
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(loss, var_list=[R, a], global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    else:
        tf.logging.error("Mode not recognized: {}".format(mode))


class OCClassifier(tf.estimator.Estimator):
    def __init__(self,
                 c=2.0,
                 kernel="linear",
                 rffm_dims=None,
                 rffm_stddev=None,
                 learning_rate=0.1,
                 input_size=None,
                 *args, **kwargs):
        """
        :param frac_err: Fraction of the inputs that are defective
        :param n_inputs: Approximation of the total number of input vectors
        :param kernel: Mapping function used: linear or rbf
        :param rffm_dims: If rbf kernel, specify the output dimensions of the map
        :param rffm_stddev: Stddev for the rbf map
        :param learning_rate: Learning rate of the optimizer
        :param input_size: Number of dimensions of the input vectors
        """

        assert kernel in ["linear", "rffm", "rbf"]
        assert kernel not in ["rffm", "rbf"] or (rffm_dims is not None and rffm_stddev is not None)
        assert kernel not in ["rffm", "rbf"] or (rffm_dims > 0 and rffm_stddev > 0)
        assert input_size is None or input_size > 0
        assert c > 0


        super(OCClassifier, self).__init__(
            model_fn=naive_svdd_model_fn,
            params={
               "c": c,
               "kernel": kernel,
               "rffm_dims": rffm_dims,
               "rffm_stddev": rffm_stddev,
               "learning_rate": learning_rate,
               "input_size": input_size
            },
            *args,
            **kwargs
        )
