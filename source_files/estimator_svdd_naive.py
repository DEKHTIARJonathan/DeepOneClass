import tensorflow as tf
from data_utils import train_input_fn, test_input_fn, train_input_fn_random
from vgg_network import VGG_Network

tf.set_random_seed(1)
tf.logging.set_verbosity("DEBUG")

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
    #
    # with tf.name_scope("VGG"):
    #     vgg_model = VGG_Network(include_FC_head=False)
    #     vgg_network, _ = vgg_model(features)
    #     features_map = vgg_network.outputs
    #     features_map_size = int(features_map.get_shape()[1])

    input_size = params["input_size"]
    print(tf.shape(features))

    with tf.name_scope("SVDD"):
        # kernel
        if params["kernel"] == "linear":
            out_size = input_size
            mapped_inputs = features
        elif params["kernel"] in ["rffm", "rbf"]:
            out_size = params["rffm_dims"] if "rffm_dims" in params else input_size # todo: check not None
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
        inputs_nbr = params["inputs_nbr"] # number of data
        C = tf.constant(1.0 / (inputs_nbr * frac_err), dtype=tf.float32)

        # Loss
        constraint = tf.square(R) - tf.square(tf.norm(mapped_inputs - a, axis=1))
        loss = tf.square(R) - C * tf.reduce_sum(tf.minimum(constraint, 0.0))
        loss = tf.reduce_sum(loss)

    # Prediction
    if mode == tf.estimator.ModeKeys.PREDICT:
        predicted_classes = tf.sign(tf.square(R) - tf.square(tf.norm(mapped_inputs - a, axis=1)))
        predictions = {
            'predicted_classes': predicted_classes
        }
        prediction_hooks = []#[_LoadPreTrainedWeightsVGG(vgg_model)]
        return tf.estimator.EstimatorSpec(mode,
                                          predictions=predictions,
                                          prediction_hooks=prediction_hooks)

    """
    # Evaluate
    if mode == tf.estimator.ModeKeys.EVAL:
        # Compute evaluation metrics.
        accuracy = tf.metrics.accuracy(labels=labels,
                                       predictions=predicted_classes,
                                       name='acc_op')
        metrics = {'accuracy': accuracy}
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)
    """

    # Train
    assert mode == tf.estimator.ModeKeys.TRAIN
    optimizer = tf.train.AdamOptimizer(0.1)
    train_op = optimizer.minimize(loss, var_list=[R, a])
    training_hooks = []#[_LoadPreTrainedWeightsVGG(vgg_model)]
    return tf.estimator.EstimatorSpec(mode,
                                      loss=loss,
                                      train_op=train_op,
                                      training_chief_hooks=training_hooks)

if __name__ == "__main__":

    target_w        = 256
    batch_size      = 1
    train_steps     = 1
    model_dir       = "../tmp/estimator_svdd_naive"

    """
    OCClassifier = tf.estimator.Estimator(
        model_fn=naive_svdd_model_fn,
        params={
            "frac_err": 0.1,
            "inputs_nbr": 100,
            "input_size": 100,
        },
        model_dir=model_dir
    )"""

    OCClassifier = tf.estimator.Estimator(
        model_fn=naive_svdd_model_fn,
        params={
            "frac_err": 0.1,
            "inputs_nbr": 100,
            "input_size": 10,
            "kernel": "linear",
            "rffm_dims": 200,
            "rffm_stddev": 25,
        },
        model_dir=model_dir
    )

    OCClassifier.train(
            input_fn=lambda: train_input_fn_random(batch_size),
            steps=train_steps
    )

    """
    predictions = OCClassifier.predict(
            input_fn=lambda: test_input_fn(6, target_w, batch_size))

    for pred_dict in predictions:
        tf.logging.info(pred_dict['predicted_classes'])
    """

    """
    OCClassifier.evaluate(
        input_fn=lambda: train_input_fn(6, target_w, batch_size),
        steps=train_steps
    )
    """