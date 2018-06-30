import tensorflow as tf

def naive_svdd_model_fn(features, labels, mode, params):
    """
    Naive SVDD
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

    with tf.variable_scope('SVDD'):
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
        loss = tf.square(R) - C * tf.reduce_mean(tf.minimum(constraint, 0.0))

        predicted_classes = tf.sign(constraint)

        tf.summary.scalar('loss', loss)
        tf.summary.scalar('radius', R)
        tf.summary.scalar('center_norm', tf.norm(a))

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
            lr = params["learning_rate"] if "q" in params else 0.1
            optimizer = tf.train.AdamOptimizer(lr)

            grads = optimizer.compute_gradients(loss, [R, a])
            train_op = optimizer.apply_gradients(grads, global_step=tf.train.get_global_step())

            tf.summary.scalar('grad_radius', grads[0][0])
            tf.summary.histogram('grad_center', grads[1][0])



            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

        else:
            tf.logging.error("Mode not recognized: {}".format(mode))


class SVDDClassifier(tf.estimator.Estimator):
    def __init__(self,
                 c=2.0,
                 kernel="linear",
                 rffm_dims=None,
                 rffm_stddev=None,
                 learning_rate=0.1,
                 input_size=None,
                 *args, **kwargs):
        """
        :param c: Regularization parameter
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

        super(SVDDClassifier, self).__init__(
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
