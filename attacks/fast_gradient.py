import tensorflow as tf

__all__ = [
    'fgm',                      # fast gradient method
    'fgsm'                      # fast gradient sign method
]


def fgm(model, x, eps=0.01, epochs=1):
    """
    Fast gradient method.

    See https://arxiv.org/abs/1412.6572 and https://arxiv.org/abs/1607.02533
    for details.  This implements the revised version since the original FGM
    has label leaking problem (https://arxiv.org/abs/1611.01236).

    :param model: A wrapper that returns the output as well as logits.
    :param x: The input placeholder.
    :param eps: The scale factor for noise.
    :param epochs: The maximum epoch to run.
    :param sign: Use gradient sign if True, otherwise use gradient value.
    :param clip_min: The minimum value in output.
    :param clip_max: The maximum value in output.

    :return: A tensor, contains adversarial samples for each input.
    """
    xadv = tf.identity(x)
    ybar = model(xadv)
    yshape = ybar.get_shape().as_list()
    ydim = yshape[1]

    indices = tf.argmax(ybar, axis=1)
    target = tf.cond(
        tf.equal(ydim, 1),
        lambda: tf.nn.relu(tf.sign(ybar - 0.5)),
        lambda: tf.one_hot(indices, ydim, on_value=1.0, off_value=0.0))

    if 1 == ydim:
        loss_fn = tf.nn.sigmoid_cross_entropy_with_logits
    else:
        loss_fn = tf.nn.softmax_cross_entropy_with_logits

    eps = tf.abs(eps)

    def _cond(xadv, i):
        return tf.less(i, epochs)

    def _body(xadv, i):
        ybar, logits = model(xadv, logits=True)
        loss = loss_fn(labels=target, logits=logits)
        dy_dx, = tf.gradients(loss, xadv)
        xadv = tf.stop_gradient(xadv + eps*tf.nn.l2_normalize(dy_dx, axis=1, epsilon=1e-10))
        return xadv, i+1

    xadv, _ = tf.while_loop(_cond, _body, (xadv, 0), back_prop=False, name='fast_gradient')
    return xadv


def fgsm(model, x, eps=0.01, epochs=1):
    """
    Fast gradient method.

    See https://arxiv.org/abs/1412.6572 and https://arxiv.org/abs/1607.02533
    for details.  This implements the revised version since the original FGM
    has label leaking problem (https://arxiv.org/abs/1611.01236).

    :param model: A wrapper that returns the output as well as logits.
    :param x: The input placeholder.
    :param eps: The scale factor for noise.
    :param epochs: The maximum epoch to run.

    :return: A tensor, contains adversarial samples for each input.
    """
    xadv = tf.identity(x)
    ybar = model(xadv)
    yshape = ybar.get_shape().as_list()
    ydim = yshape[1]

    indices = tf.argmax(ybar, axis=1)
    target = tf.cond(
        tf.equal(ydim, 1),
        lambda: tf.nn.relu(tf.sign(ybar - 0.5)),
        lambda: tf.one_hot(indices, ydim, on_value=1.0, off_value=0.0))

    if 1 == ydim:
        loss_fn = tf.nn.sigmoid_cross_entropy_with_logits
    else:
        loss_fn = tf.nn.softmax_cross_entropy_with_logits

    eps = tf.abs(eps)

    def _cond(xadv, i):
        return tf.less(i, epochs)

    def _body(xadv, i):
        ybar, logits = model(xadv, logits=True)
        loss = loss_fn(labels=target, logits=logits)
        dy_dx, = tf.gradients(loss, xadv)
        xadv = tf.stop_gradient(xadv + eps*tf.sign(dy_dx))
        return xadv, i+1

    xadv, _ = tf.while_loop(_cond, _body, (xadv, 0), back_prop=False, name='fast_gradient_sign')

    # ybar_1, logits_1 = model(x, logits=True)
    # loss_1 = loss_fn(labels=target, logits=logits_1)
    # gradient_loss, = tf.gradients(loss_1, x)
    # eta = eps * eps*tf.sign(gradient_loss)
    # return xadv, gradient_loss, eta

    return xadv
