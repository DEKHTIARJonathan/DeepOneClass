import tensorflow as tf
import sys

class NaiveSVDDLinear(object):
    def __init__(self, dims=2, c=1, mean_r=10, mean_a=10):
        self.R = tf.Variable(tf.random_normal([], mean=mean_r), dtype=tf.float32, name="Radius")
        self.a = tf.Variable(tf.random_normal([dims], mean=mean_a), dtype=tf.float32, name="Center")
        self.C = tf.constant(c, dtype=tf.float32)
        self.X = tf.placeholder(tf.float32, shape=(None, dims), name="X")

    def _get_loss(self, inputs):
        constraint = tf.square(self.R) - tf.square(tf.norm(inputs - self.a, axis=1))
        loss = tf.square(self.R) - self.C * tf.reduce_sum(tf.minimum(constraint, 0.0))
        loss = tf.reduce_sum(loss)
        return loss

    def train(self, sess, inputs, epochs=5000, optimizer=tf.train.AdamOptimizer(0.1)):
        loss = self._get_loss(inputs)
        train = optimizer.minimize(loss)

        sess.run(tf.global_variables_initializer())
        for e in range(epochs):
            _, l, R1, a1 = sess.run([train, loss, self.R, self.a], feed_dict={self.X: inputs})

            towrite = "\r{0:4.1f} %, Loss: {1:7.4f}, R: {2:7.4f}, a:".format(e / epochs * 100, l, R1) + str(a1)
            sys.stdout.write(towrite)
            sys.stdout.flush()

        return l, R1, a1


    def eval(self, sess, inputs):
        eval = tf.sign(tf.square(self.R) - tf.square(tf.norm(inputs - self.a, axis=1)))
        return sess.run(eval)


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    num_samples = 100
    frac_err = 0.0005

    x_train = np.random.multivariate_normal(mean=[1., 1.], cov=np.eye(2), size=num_samples).astype(np.float32)
    x_eval = np.vstack([
        np.random.multivariate_normal(mean=[1., 1.], cov=np.eye(2), size=950).astype(np.float32),
        np.random.multivariate_normal(mean=[10., 10.], cov=np.eye(2), size=50).astype(np.float32)
    ])

    svdd = NaiveSVDDLinear(c=1.0/(frac_err*num_samples), dims=2)

    with tf.Session() as sess:
        loss, R, a = svdd.train(sess, x_train, epochs=10000)
        eval_y = svdd.eval(sess, x_eval)
        print(eval_y)


    fig = plt.gcf()
    ax = fig.gca()

    s1p = plt.plot(x_eval[eval_y > 0, 0], x_eval[eval_y > 0, 1], 'x', c='blue')
    s1n = plt.plot(x_eval[eval_y < 0, 0], x_eval[eval_y < 0, 1], 'x', c='red')

    c = plt.Circle(tuple(a), R, fill=False)
    ax.add_patch(c)

    plt.axis('equal')

    plt.show()


