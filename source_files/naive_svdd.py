"""
Naive implementation of the primal form of the SVDD with an explicit feature map as the kernel
"""
import tensorflow as tf
import tensorlayer as tl
import time
import numpy as np

class SvddLayer(tl.layers.Layer):
    def __init__(self, layer=None, name='svdd_layer', c=1, map="linear", rffm_dims=None, rffm_stddev=10.0, freeze_before=False):

        tl.layers.Layer.__init__(self, name=name)
        self.inputs = layer.outputs

        in_dims = self.inputs.get_shape().as_list()[1]

        mapped_inputs = self.inputs

        if map == "linear":
            out_dims = in_dims
        elif map in ["rffm", "rbf"]:
            out_dims = rffm_dims if rffm_dims else in_dims
            kernel_mapper = tf.contrib.kernel_methods.RandomFourierFeatureMapper(
                input_dim=in_dims,
                output_dim=out_dims,
                stddev=rffm_stddev,
                name="rffm"
            )
            mapped_inputs = kernel_mapper.map(self.inputs)
        else:
            raise ValueError("Map function {} not implemented.".format(map))


        with tf.variable_scope(name):
            self._radius = tf.Variable(tf.random_normal([], mean=10.0), dtype=tf.float32, name="radius")
            self._center = tf.Variable(tf.random_normal([out_dims], mean=10.0), dtype=tf.float32, name="center")
            self._c = tf.constant(c, dtype=tf.float32, name="C")

        self.outputs = tf.sign(tf.square(self._radius) - tf.square(tf.norm(mapped_inputs - self._center, axis=1)))

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)

        self.all_layers.extend([self.outputs])
        self.all_params.extend([self._radius, self._center])


        constraint = tf.square(self._radius) - tf.square(tf.norm(mapped_inputs - self._center, axis=1))
        self.cost = tf.square(self._radius) - self._c * tf.reduce_sum(tf.minimum(constraint, 0.0))
        self.cost = tf.reduce_sum(self.cost)

        if freeze_before:
            var_list = [self._radius, self._center]
        else:
            var_list = self.all_params


        self.train_op = tf.train.AdamOptimizer(0.1).minimize(self.cost, var_list=var_list)

    def train(self, sess, x, X_train, n_epoch=5000, batch_size=100, print_freq=1):

        print("     [*] %s start training" % self.name)
        print("     batch_size: %d" % batch_size)
        print("     n_epoch: %d" % n_epoch)

        for epoch in range(n_epoch):
            start_time = time.time()
            n_batch = 0
            train_loss = 0

            for X_train_a, _ in tl.iterate.minibatches(X_train, X_train, batch_size, shuffle=True):
                feed_dict = {x: X_train_a}
                _, err, r = sess.run([self.train_op, self.cost, self._radius], feed_dict=feed_dict)
                train_loss += err
                n_batch += 1

            if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
                print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
                if n_batch > 0:
                    print("   train loss: %f" % (train_loss / n_batch))
                    print("   radius: %f" % r)

    def predict(self, sess, x, X_predict, batch_size=100):
        print("     [*] %s start predict" % self.name)
        print("     batch_size: %d" % batch_size)

        predict_loss = 0
        predict_y = np.array([])

        for X_predict_a, _ in tl.iterate.minibatches(X_predict, X_predict, batch_size, shuffle=False):
            feed_dict = {x: X_predict_a}
            y, err = sess.run([self.outputs, self.cost], feed_dict=feed_dict)
            predict_y = np.concatenate((predict_y, y), axis=0)
            predict_loss += err

        return predict_y, predict_loss







if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    inpt_plh = tf.placeholder(tf.float32, shape=(None, 2), name="X")
    inpt = tl.layers.InputLayer(inpt_plh)
    outpt = SvddLayer(inpt, map='rbf', rffm_dims=200, rffm_stddev=25)

    x_train = np.random.multivariate_normal(mean=[1., 1.], cov=np.eye(2), size=100).astype(np.float32)
    x_eval = np.vstack([
        np.random.multivariate_normal(mean=[1., 1.], cov=np.eye(2), size=950).astype(np.float32),
        np.random.multivariate_normal(mean=[10., 10.], cov=np.eye(2), size=50).astype(np.float32)
    ])

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    res = sess.run(outpt.outputs, feed_dict={inpt_plh: x_train})

    assert(res.shape == (100,))

    outpt.train(sess, inpt_plh, x_train, n_epoch=10000)

    eval_y = sess.run(outpt.outputs, feed_dict={inpt_plh: x_eval})

    fig = plt.gcf()
    ax = fig.gca()

    s1p = plt.plot(x_eval[eval_y > 0, 0], x_eval[eval_y > 0, 1], 'x', c='blue')
    s1n = plt.plot(x_eval[eval_y < 0, 0], x_eval[eval_y < 0, 1], 'x', c='red')
    s1t = plt.plot(x_train[:, 0], x_train[:, 1], 'x', c='green')
    plt.axis('equal')
    plt.show()

    print("Radius: %d" % sess.run(outpt._radius, feed_dict={inpt_plh: x_eval}))




