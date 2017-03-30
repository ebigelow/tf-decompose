
import numpy as np
import tensorflow as tf
from tqdm import trange, tqdm

import logging
logging.basicConfig(filename='loss.log', level=logging.DEBUG)
_log = logging.getLogger('CP')

def tenumerate(ls):
    return enumerate(tqdm(ls))

def shuffled(ls):
    return sorted(list(ls), key=lambda _: np.random.rand())


class DecomposedTensor:
    """
    Represent CP / Tucker decomposition of a tensor in TensorFlow.

    """

    def init_random(self, a=0.0, b=1.0):
        pass

    def init_reconstruct(self):
        """
        Initialize variable for components of reconstructed tensor `X`.

        """
        pass

    def init_norm(self):
        pass

    def get_train_ops(self, X_var, optimizer):
        """
        Get separate optimizers for each component matrix.

        """
        pass

    def train_als(self, X_data, optimizer, epochs=10):
        """
        Use alt. least-squares to find the CP/Tucker decomposition of tensor `X`.

        """
        X_var = tf.Variable(X_data)
        loss_op, train_ops = self.get_train_ops(X_var, optimizer)

        init_op = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init_op)

            for e in trange(epochs):
                for alt, train_op in tenumerate(shuffled(train_ops)):
                    _, loss = sess.run([train_op, loss_op], feed_dict={X_var: X_data})
                    _log.debug('[%3d:%3d] loss: %.5f' % (e, alt, loss))

            print 'final loss: %.5f' % loss
            return sess.run(self.X)


    def get_fit(self, X):
        normX = tf.reduce_sum(X ** 2)
        normY = tf.reduce_sum(self.X ** 2)
        inner = tf.reduce_sum(X * self.X)

        normresidual = normX  +  normY - 2*inner
        return 1 - (normresidual / normX)


    def train_als_early(self, X_data, optimizer, epochs=1000, stop_freq=50, stop_thresh=1e-10):
        """
        ALS with early stopping.

        """
        X_var = tf.Variable(X_data)
        loss_op, train_ops = self.get_train_ops(X_var, optimizer)
        fit_prev = 0.0

        init_op = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init_op)

            for e in trange(epochs):
                for alt, train_op in enumerate(shuffled(train_ops)):
                    _, loss = sess.run([train_op, loss_op], feed_dict={X_var: X_data})
                    _log.debug('[%3d:%3d] loss: %.5f' % (e, alt, loss))

                if e % stop_freq:
                    fit_op = self.get_fit(X_var)
                    fit = sess.run(fit_op, feed_dict={X_var: X_data})
                    fit_change = abs(fit - fit_prev)

                    if fit_change < stop_thresh and e > 0:
                        print 'Stopping early, fit_change: %.15f' % (fit_change)
                        break
                    fit_prev = fit

            print 'final loss: %.5f\nfinal fit %.5f' % (loss, fit)
            return sess.run(self.X)
