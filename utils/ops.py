'''
Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights on this
computer program.

You can only use this computer program if you have closed a license agreement with MPG or you get the right to use
the computer program from someone who is authorized to grant you that right.

Any use of the computer program without a valid license is prohibited and liable to prosecution.

Copyright 2019 Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG). acting on behalf of its
Max Planck Institute for Intelligent Systems and the Max Planck Institute for Biological Cybernetics.
All rights reserved.

More information about VOCA is available at http://voca.is.tue.mpg.de.
For comments or questions, please email us at voca@tue.mpg.de
'''

import tensorflow as tf

def fc_layer(inputs,
             num_units_in,
             num_units_out,
             init_weights=None,
             weightini=0.1,
             bias=0.,
             trainable=True,
             scope=None,
             regularisation_constant=0.0):
    with tf.variable_scope(scope):
        weights_shape = [num_units_in, num_units_out]

        if init_weights is not None:
            weights_initializer = tf.constant(init_weights, dtype=tf.float32)
            weights_shape = None
        elif weightini == 0.:
            weights_initializer = tf.constant_initializer(weightini)
        else:
            weights_initializer = tf.truncated_normal_initializer(stddev=weightini)

        weights = tf.get_variable('weights',
                                  shape=weights_shape,
                                  initializer=weights_initializer,
                                  trainable=trainable,
                                  regularizer=tf.contrib.layers.l2_regularizer(scale=regularisation_constant))

        bias_shape = [num_units_out, ]
        bias_initializer = tf.constant_initializer(bias)
        biases = tf.get_variable('biases',
                                 shape=bias_shape,
                                 initializer=bias_initializer,
                                 trainable=trainable)
        outputs = tf.nn.xw_plus_b(inputs, weights, biases)
    return outputs

def custom_fc_layer(inputs,
             num_units_in,
             num_units_out,
             init_weights=None,
             weightini=0.1,
             bias=0.,
             trainable_weights=True,
             trainable_bias=True,
             regularisation_constant=0.0,
             output_weights=False,
              scope=None):
    with tf.variable_scope(scope):
        weights_shape = [num_units_in, num_units_out]

        if init_weights is not None:
            weights_initializer = tf.constant(init_weights, dtype=tf.float32)
            weights_shape = None
        elif weightini == 0.:
            weights_initializer = tf.constant_initializer(weightini)
        else:
            weights_initializer = tf.truncated_normal_initializer(stddev=weightini)

        weights = tf.get_variable('weights',
                                  shape=weights_shape,
                                  initializer=weights_initializer,
                                  trainable=trainable_weights,
                                  regularizer=tf.contrib.layers.l2_regularizer(scale=regularisation_constant))

        bias_shape = [num_units_out, ]
        bias_initializer = tf.constant_initializer(bias)
        biases = tf.get_variable('biases',
                                 shape=bias_shape,
                                 initializer=bias_initializer,
                                 trainable=trainable_bias)
        outputs = tf.nn.xw_plus_b(inputs, weights, biases)
    if not output_weights:
        return outputs
    else:
        return outputs, weights


def conv2d(inputs, n_filters,
           k_h=5, k_w=5,
           stride_h=2, stride_w=2,
           stddev=0.02,
           activation=None,
           bias=True,
           padding='SAME',
           scope=None,
           regularisation_constant=0.0):
    """2D Convolution with options for kernel size, stride, and init deviation.
    Parameters
    ----------
    inputs : Tensor
        Input tensor to convolve.
    n_filters : int
        Number of filters to apply.
    k_h : int, optional
        Kernel height.
    k_w : int, optional
        Kernel width.
    stride_h : int, optional
        Stride in rows.
    stride_w : int, optional
        Stride in cols.
    stddev : float, optional
        Initialization's standard deviation.
    activation : arguments, optional
        Function which applies a nonlinearity
    padding : str, optional
        'SAME' or 'VALID'
    scope : str, optional
        Variable scope to use.
    Returns
    -------
    x : Tensor
        Convolved input.
    """
    with tf.variable_scope(scope):
        w = tf.get_variable('weights',
                            [k_h, k_w, inputs.get_shape()[-1], n_filters],
                            initializer=tf.truncated_normal_initializer(stddev=stddev),
                            regularizer=tf.contrib.layers.l2_regularizer(scale=regularisation_constant))
        conv = tf.nn.conv2d(inputs,
                            w,
                            strides=[1, stride_h, stride_w, 1],
                            padding=padding)
        if bias:
            b = tf.get_variable('biases',
                                [n_filters],
                                initializer=tf.truncated_normal_initializer(stddev=stddev))
            conv = tf.nn.bias_add(conv, b)
        if activation:
            conv = activation(conv)
        return conv


class BatchNorm(object):
    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, reuse=False, is_training=True):
        return tf.contrib.layers.batch_norm(x,
                                            decay=self.momentum,
                                            updates_collections=None,
                                            epsilon=self.epsilon,
                                            center=True,
                                            scale=True,
                                            is_training=is_training,
                                            reuse=reuse,
                                            scope=self.name)