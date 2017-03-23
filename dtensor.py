
import numpy as np
import tensorflow as tf
from tqdm import trange

import logging
logging.basicConfig(filename='loss.log', level=logging.DEBUG)
_log = logging.getLogger('CP')


def shuffled(ls):
    return sorted(list(ls), key=lambda _: np.random.rand())



class DecomposedTensor:
    """
    Used for CP & Tucker decomposition of a tensor.

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
        Use alt. least-squares to find the Tucker decomposition of tensor `X`.

        """
        X_var = tf.Variable(X_data)
        loss_op, train_ops = self.get_train_ops(X_var, optimizer)

        init_op = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init_op)

            for e in trange(epochs):
                for alt, train_op in enumerate(shuffled(train_ops)):
                    _, loss = sess.run([train_op, loss_op], feed_dict={X_var: X_data})
                    _log.debug('[%3d:%3d] loss: %.5f' % (e, alt, loss))

            print 'final loss: %.5f' % loss
            return sess.run(self.X)
