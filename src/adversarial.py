# Adversarial Oversampling Method
import tensorflow as tf
import numpy as np
from attacks import fgm
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# set the random seed, so we can get stable results.
# tf.set_random_seed(0)


class AdversarialOversampling(object):
    def __init__(self, input_dims, hidden_neurons, eps=0.1, pfp=0.5):
        self.input_dims = input_dims
        self.hidden_neurons = hidden_neurons
        self.eps = eps
        self.pfp = pfp
        self.new = []
        self.data_t = None
        tf.reset_default_graph()

        with tf.variable_scope('model'):
            self.x = tf.placeholder(tf.float32, (None, input_dims), name='x')
            self.y = tf.placeholder(tf.float32, (None, 1), name='y')
            self.y_pred = self.simple_mlp(self.x)

            # calculate the accuracy
            with tf.variable_scope('acc'):
                self.y_ = tf.where(tf.greater(self.y_pred, 0.5), tf.ones_like(self.y_pred), tf.zeros_like(self.y_pred))
                count = tf.equal(self.y, self.y_)
                self.acc = tf.reduce_mean(tf.cast(count, tf.float32), name='acc')

            with tf.variable_scope('loss'):
                self.entropy_loss = -tf.reduce_mean(
                    self.y * tf.log(tf.clip_by_value(self.y_pred, 1e-10, 1.0)) +
                    (1 - self.y) * tf.log(tf.clip_by_value(1 - self.y_pred, 1e-10, 1.0)))

                self.reg_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
                self.loss = self.entropy_loss + self.reg_loss

            with tf.variable_scope('train_op'):
                optimizer = tf.train.AdamOptimizer(0.001)
                self.train_op = optimizer.minimize(self.loss)

        with tf.variable_scope('model', reuse=True):
            self.fgm_eps = tf.placeholder(tf.float32, (), name='fgm_eps')
            self.x_fgm = fgm(self.simple_mlp, self.x, epochs=1, eps=self.fgm_eps)

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

    def simple_mlp(self, x, logits=False):

        with tf.variable_scope('mlp'):
            z = tf.layers.dense(x,
                                units=10,
                                activation=tf.nn.sigmoid,
                                kernel_initializer=tf.glorot_normal_initializer(seed=666),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

            logits_ = tf.layers.dense(z,
                                      units=1,
                                      name='logits',
                                      kernel_initializer=tf.glorot_normal_initializer(seed=666),
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

        y_pred = tf.nn.sigmoid(logits_, name='y_pred')

        if logits:
            return y_pred, logits_

        return y_pred

    def evaluate(self, X_data, y_data):
        """
        Evaluate TF model by running env.acc.
        """
        acc = self.sess.run(self.acc, feed_dict={self.x: X_data, self.y: y_data})
        return acc

    def fit_sample(self, data, label, acc_threshold=0.8, shuffle=True):
        """
        Train a TF model by running env.train_op.
        """
        n_sample = data.shape[0]
        data = np.array(data)
        acc = 0.
        while acc <= acc_threshold:
            if shuffle:
                ind = np.arange(n_sample)
                np.random.shuffle(ind)
                X_data = data[ind]
                y_data = np.reshape(label[ind], (-1, 1))
            self.sess.run(self.train_op, feed_dict={self.x: X_data, self.y: y_data})
            acc = self.evaluate(X_data, y_data)

        data_t, data_f, label_t, label_f = [], [], [], []

        for i in range(n_sample):
            if label[i] == 1:
                data_t.append(data[i])
                label_t.append(label[i])
            if label[i] == 0:
                data_f.append(data[i])
                label_f.append(label[i])

        T = len(data_f) / (1 - self.pfp) - len(data_f) - len(data_t)

        self.data_t = np.array(data_t)
        temp = []

        while len(temp) <= T:
            y_pred = self.sess.run(self.y_, feed_dict={self.x: self.data_t})
            y_pred = np.squeeze(y_pred)
            indxs = y_pred != np.array(label_t)
            X_adv = self.sess.run(self.x_fgm, feed_dict={self.x: self.data_t[indxs], self.fgm_eps: self.eps})
            temp.extend(X_adv)

            if shuffle:
                ind = np.arange(n_sample)
                np.random.shuffle(ind)
                X_data = data[ind]
                y_data = np.reshape(label[ind], (-1, 1))
                self.sess.run(self.train_op, feed_dict={self.x: X_data, self.y: y_data})

        self.new = np.array(temp)
        self.new = np.append(data_t, self.new, axis=0)

        train_new = np.append(data_f, self.new, axis=0)
        label_new = np.append(np.zeros(len(data_f)), np.ones(len(self.new)), axis=0)
        return train_new, label_new