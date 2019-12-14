

import tensorflow as tf
import pandas as pd
import numpy as np
from src.calc_metrics import Metrics
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from attacks import fgsm

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


# set the random seed, so we can get stable results.
tf.set_random_seed(0)


def load_data(folder_path):
    files = os.listdir(folder_path)
    data_list, label_list = [], []

    for file in files:
        file_path = folder_path + file
        df = pd.read_csv(file_path)
        df_metrics = df.drop(['bug'], axis=1)
        df.loc[df['bug'] >= 1, 'bug'] = 1
        df_label = df['bug']
        data = np.array(df_metrics.values.tolist())
        label = np.array(df_label.values.tolist())
        data_list.append(data)
        label_list.append(label)

    return data_list, label_list


def calc_performance(label_train, label_pred, pred_proba):
    calc_metrics = Metrics(actual=label_train, prediction=label_pred)
    fpr, tpr, _ = roc_curve(label_train, pred_proba, pos_label=1)
    auc_value = auc(fpr, tpr)
    stats = np.array([j.stats() for j in calc_metrics()])
    stats = stats.flatten()
    recall = stats[0]
    pre = stats[3]
    acc = stats[4]
    f_score = stats[5]
    false_alarm = stats[1]
    return auc_value, recall, pre, acc, f_score, false_alarm


def simple_mlp(x, logits=False):

    with tf.variable_scope('mlp'):
        # z = tf.layers.dense(x, units=10, activation=tf.nn.sigmoid, kernel_initializer=tf.glorot_normal_initializer())
        z = tf.layers.dense(x,
                            units=10,
                            activation=tf.nn.sigmoid,
                            kernel_initializer=tf.glorot_normal_initializer(),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    logits_ = tf.layers.dense(z,
                              units=1,
                              name='logits',
                              kernel_initializer=tf.glorot_normal_initializer(),
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    y_pred = tf.nn.sigmoid(logits_, name='y_pred')

    if logits:
        return y_pred, logits_

    return y_pred


# define a cnn model class
class Dummy:
    pass

env = Dummy()

with tf.variable_scope('model'):
    # forward propagation
    env.x = tf.placeholder(tf.float32, (None, 20), name='x')
    env.y = tf.placeholder(tf.float32, (None, 1), name='y')
    env.y_pred, env.logits = simple_mlp(env.x, logits=True)

    # calculate the accuracy
    with tf.variable_scope('acc'):
        env.y_ = tf.where(tf.greater(env.y_pred, 0.5), tf.ones_like(env.y_pred), tf.zeros_like(env.y_pred))
        count = tf.equal(env.y, env.y_)
        env.acc = tf.reduce_mean(tf.cast(count, tf.float32), name='acc')

    # calculate loss function
    with tf.variable_scope('loss'):
        env.entropy_loss = -tf.reduce_mean(
            env.y * tf.log(tf.clip_by_value(env.y_pred, 1e-10, 1.0)) +
            (1 - env.y) * tf.log(tf.clip_by_value(1 - env.y_pred, 1e-10, 1.0)))

        env.reg_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        env.loss = env.entropy_loss + env.reg_loss

    # train operation
    with tf.variable_scope('train_op'):
        optimizer = tf.train.AdamOptimizer(learning_rate=0.05)
        env.train_op = optimizer.minimize(env.loss)

    env.saver = tf.train.Saver(max_to_keep=1000)

with tf.variable_scope('model', reuse=True):
    env.fgsm_eps = tf.placeholder(tf.float32, (), name='fgsm_eps')
    env.fgsm_epochs = tf.placeholder(tf.int32, (), name='fgsm_epochs')

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())


def train(sess, env, X_train, y_train, training_epochs=1, shuffle=True, batch_size=128):
    """
    Train a TF model by running env.train_op.
    """
    n_sample = X_train.shape[0]
    n_batch = (n_sample + batch_size - 1) // batch_size

    for epoch in range(training_epochs):
        # print('\nEpoch {0} / {1}'.format(epoch+1, training_epochs))

        if shuffle:
            ind = np.arange(n_sample)
            np.random.shuffle(ind)
            X_train = X_train[ind]
            y_train = y_train[ind]

        for batch in range(n_batch):
            start = batch * batch_size
            end = min(n_sample, start + batch_size)
            sess.run(env.train_op, feed_dict={env.x: X_train[start: end],
                                              env.y: y_train[start: end]})


def make_fgsm(sess, env, X_data, y_data, epochs=1, eps=0.01, batch_size=128):
    """
    Generate FGSM by running env.x_fgsm.
    """
    n_sample = X_data.shape[0]
    n_batch = (n_sample + batch_size - 1) // batch_size
    X_adv = np.empty_like(X_data)
    Grads = np.empty_like(X_data)
    get_eta = np.empty_like(X_data)

    with tf.variable_scope('model', reuse=True):
        env.x_fgsm = fgsm(simple_mlp, env.x, epochs=env.fgsm_epochs, eps=env.fgsm_eps)

    for batch in range(n_batch):
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        adv, gradients_loss, eta = sess.run(env.x_fgsm,
                                            feed_dict={env.x: X_data[start:end],
                                                       env.fgsm_eps: eps,
                                                       env.fgsm_epochs: epochs})
        X_adv[start:end] = adv
        Grads[start:end] = gradients_loss
        get_eta[start: end] = eta
    return X_adv, y_data, Grads, get_eta


def select_one_adversarial_example(sess, env, X_test, y_test, adv_eps):
    adv_epochs = 1
    adv_batch_size = 256

    pred_proba = sess.run(env.y_pred, feed_dict={env.x: X_test})
    pred_proba = np.squeeze(pred_proba)
    y_pred = np.array(pred_proba >= 0.5).astype(int).flatten()
    y_test = np.squeeze(y_test)

    X_adv, y_adv, Grads, get_eta = make_fgsm(sess, env, X_test, y_test, eps=adv_eps, epochs=adv_epochs, batch_size=adv_batch_size)

    pred_proba_after = sess.run(env.y_pred, feed_dict={env.x: X_adv})
    pred_proba_after = np.squeeze(pred_proba_after)
    y_pred_after = (pred_proba_after >= 0.5).astype(int).flatten()

    for i in range(len(y_test)):
        if (y_pred[i] == y_test[i] == 0 and y_pred_after[i] == 1 and pred_proba[i] < 0.4 and pred_proba_after[i] > 0.7) \
                or (y_pred[i] == y_test[i] == 1 and y_pred_after[i] == 0 and pred_proba[i] > 0.6 and pred_proba_after[i] < 0.3):

            print('Clean example:')
            print(X_test[i])
            print('True Label:')
            print(y_test[i])
            print('predicted Label before attack:')
            print(y_pred[i])
            print('predicted Label proba:')
            print(pred_proba[i])
            print('Adversarial example:')
            print(X_adv[i])
            print('Adversarial perturbation:')
            print(X_adv[i] - X_test[i])
            print('loss gradients:')
            print(Grads[i])
            print('predicted Label after attack:')
            print(y_pred_after[i])
            print('predicted Label proba after attack:')
            print(pred_proba_after[i])
            break

if __name__ == '__main__':

    # read the defect dataset
    folder_path = './imbalanced_datasets/'
    saved_file_path = './saved_results/'
    data_list, label_list = load_data(folder_path)

    batch_size = 256
    training_epochs = 500

    adv_eps = 0.05
    files = os.listdir(folder_path)

    for i in range(len(files)):
        print("*" * 20)
        print(" Dataset  " + files[i].rstrip('csv').strip('.'))
        print("*" * 20)

        input_dims = data_list[i].shape[1]
        X_, y_ = data_list[i], label_list[i]
        y_ = np.reshape(label_list[i], (-1, 1))

        X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size=0.3, stratify=y_, random_state=666)

        scaler = preprocessing.MinMaxScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        train(sess, env, X_train, y_train, training_epochs=training_epochs, shuffle=True, batch_size=batch_size)

        select_one_adversarial_example(sess, env, X_test, y_test, adv_eps)











