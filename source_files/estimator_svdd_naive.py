import tensorflow as tf
from vgg_network import VGG_Network

class _LoadPreTrainedWeightsVGG(tf.train.SessionRunHook):

    def __init__(self, vgg_model):
        self._vgg_model = vgg_model

    def after_create_session(self, session, coord):
        tf.logging.debug("vgg_model pretrained weights are to be loaded")
        self._vgg_model.load_pretrained(session)

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

    if "input_size" in params:
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
    frac_err = params["frac_err"]
    n_inputs = params["n_inputs"]
    C = tf.constant(1.0 / (n_inputs * frac_err), dtype=tf.float32)

    # Loss
    constraint = tf.square(R) - tf.square(tf.norm(mapped_inputs - a, axis=1))
    loss = tf.square(R) - C * tf.reduce_sum(tf.minimum(constraint, 0.0))
    loss = tf.reduce_sum(loss)
    tf.summary.scalar('loss', loss)

    if mode in [tf.estimator.ModeKeys.PREDICT, tf.estimator.ModeKeys.EVAL]:
        # Compute predictions.
        predicted_classes = tf.sign(tf.square(R) - tf.square(tf.norm(mapped_inputs - a, axis=1)))

    # Prediction
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'predicted_distance': tf.square(tf.norm(mapped_inputs - a, axis=1)),
            'predicted_classes': predicted_classes,
            'mapped_inputs': mapped_inputs
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Evaluate
    if mode == tf.estimator.ModeKeys.EVAL:
        # Compute evaluation metrics.
        accuracy = tf.metrics.accuracy(labels=labels,
                                       predictions=predicted_classes,
                                       name='acc_op')
        metrics = {'accuracy': accuracy}
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    # Train
    assert mode == tf.estimator.ModeKeys.TRAIN
    lr = params["learning_rate"] if "learning_rate" in params else 0.1
    optimizer = tf.train.AdamOptimizer(lr)
    train_op = optimizer.minimize(loss, var_list=[R, a], global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
