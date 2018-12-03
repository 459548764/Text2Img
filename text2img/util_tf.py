import tensorflow as tf


def bn(input_layer,
       is_training,
       batch_norm_decay=0.9,
       scale=True,
       updates_collections=None):
        if is_training is None:
            raise ValueError('Specify train phase by `is_training`')
        return tf.contrib.layers.batch_norm(input_layer,
                                            decay=batch_norm_decay,
                                            is_training=is_training,
                                            updates_collections=updates_collections,
                                            scale=scale)


def glu(x):
    """ gated linear unit
    - x: batch, feature_1, ,,, feature_n, feature_n should be even value!!
    - return: batch, feature_1, ,,, feature_n/2
    """
    dim = x.get_shape()[-1]
    _dim = tf.cast(dim/2, tf.int32)
    x_0, x_1 = tf.slice(x, [_dim], -1)
    return x_0 * tf.sigmoid(x_1)


def full_connected(x,
                   weight_shape,
                   scope=None,
                   bias=True,
                   reuse=None):
    """ fully connected layer
    - weight_shape: input size, output size
    - priority: batch norm (remove bias) > dropout and bias term
    """
    with tf.variable_scope(scope or "fully_connected", reuse=reuse):
        w = tf.get_variable("weight", shape=weight_shape, dtype=tf.float32)
        x = tf.matmul(x, w)
        if bias:
            b = tf.get_variable("bias", initializer=[0.0] * weight_shape[-1])
            return tf.add(x, b)
        else:
            return x


def convolution(x,
                weight_shape,
                stride,
                padding="SAME",
                scope=None,
                bias=True,
                reuse=None,
                stddev=0.02):
    """2d convolution
     Parameter
    -------------------
    weight_shape: width, height, input channel, output channel
    stride (list): [stride for axis 1, stride for axis 2]
    """
    with tf.variable_scope(scope or "2d_convolution", reuse=reuse):
        w = tf.get_variable('weight',
                            shape=weight_shape,
                            dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        x = tf.nn.conv2d(x, w, strides=[1, stride[0], stride[1], 1], padding=padding)
        if bias:
            b = tf.get_variable("bias", initializer=[0.0] * weight_shape[-1])
            return tf.add(x, b)
        else:
            return x


def convolution_trans(x,
                      weight_shape,
                      output_shape,
                      stride,
                      padding="SAME",
                      scope=None,
                      bias=True,
                      reuse=None):
    """2d fractinally-strided convolution (transposed-convolution)
     Parameter
    --------------------
    weight_shape: width, height, output channel, input channel
    stride (list): [stride for axis 1, stride for axis 2]
    output_shape (list): [batch, width, height, output channel]
    """
    with tf.variable_scope(scope or "convolution_trans", reuse=reuse):
        w = tf.get_variable('weight', shape=weight_shape, dtype=tf.float32)
        x = tf.nn.conv2d_transpose(x,
                                   w,
                                   output_shape=output_shape,
                                   strides=[1, stride[0], stride[1], 1],
                                   padding=padding,
                                   data_format="NHWC")
        if bias:
            b = tf.get_variable("bias", initializer=[0.0] * weight_shape[2])
            return tf.add(x, b)
        else:
            return x


def dynamic_batch_size(inputs):
    """ Dynamic batch size, which is able to use in a model without deterministic batch size.
    See https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/python/ops/rnn.py
    """
    while nest.is_sequence(inputs):
        inputs = inputs[0]
    return array_ops.shape(inputs)[0]