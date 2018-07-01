import tensorflow as tf

def naive_ocsvm_model_fn(features, labels, mode, params):
    """
    Naive OCSVM
    :param features: data passed to fit function (train/predict/eval)
    :param labels: labels passed (used only for evaluate)
    :param mode: train/predict/eval
    :param params: dict of additional params
    :return: tf.estimator.EstimatorSpec
    """

    if "input_size" in params and params["input_size"]:
        input_size = params["input_size"]
    elif features.get_shape().dims is not None:
        input_size = features.get_shape().as_list()[1]
    else:
        raise Exception("Input size is unknown, either not given via params or features shape is None")

    features.set_shape((None, input_size))

    with tf.variable_scope('OCSVM'):
        if params["kernel"] == "linear":
            out_size = input_size
            mapped_inputs = features
        else:
            raise ValueError("Map function {} not implemented.".format(params["kernel"]))

        # Model variables
        ro = tf.Variable(tf.random_normal([], mean=0.1), dtype=tf.float32, name="ro")
        w = tf.Variable(tf.random_normal([out_size], mean=0.1), dtype=tf.float32, name="w")

        # Inputs and constants
        C = tf.constant(params["c"], dtype=tf.float32, name="C")

        # Loss
        constraint = tf.matmul(mapped_inputs, tf.expand_dims(w, -1)) - ro
        loss = 0.5 * tf.square(tf.norm(w)) - ro - C * tf.reduce_mean(tf.minimum(constraint, 0.0))

        predicted_classes = tf.sign(constraint)

        tf.summary.scalar('loss', loss)
        tf.summary.scalar('ro_offset', ro)
        tf.summary.scalar('w_norm', tf.norm(w))

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

            grads = optimizer.compute_gradients(loss, [ro, w])
            train_op = optimizer.apply_gradients(grads, global_step=tf.train.get_global_step())

            tf.summary.scalar('grad_ro', grads[0][0])
            tf.summary.histogram('grad_w', grads[1][0])



            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

        else:
            tf.logging.error("Mode not recognized: {}".format(mode))


class OCSVMClassifier(tf.estimator.Estimator):
    def __init__(self,
                 c=2.0,
                 kernel="linear",
                 learning_rate=0.1,
                 input_size=None,
                 *args, **kwargs):
        """
        :param c: Regularization parameter
        :param kernel: Mapping function used: linear
        :param learning_rate: Learning rate of the optimizer
        :param input_size: Number of dimensions of the input vectors
        """

        assert kernel in ["linear"]
        assert input_size is None or input_size > 0
        assert c > 0

        super(OCSVMClassifier, self).__init__(
            model_fn=naive_ocsvm_model_fn,
            params={
               "c": c,
               "kernel": kernel,
               "learning_rate": learning_rate,
               "input_size": input_size
            },
            *args,
            **kwargs
        )
