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
from utils.ops import fc_layer, conv2d, BatchNorm

class SpeechEncoder:
    def __init__(self, config, scope='SpeechEncoder'):
        self.scope = scope
        self._speech_encoding_dim = config['expression_dim']
        self._condition_speech_features = config['condition_speech_features']
        self._speech_encoder_size_factor = config['speech_encoder_size_factor']

    def __call__(self, speech_features, condition, is_training, reuse=False):
        with tf.variable_scope(self.scope, reuse=reuse):
            if reuse == True:
                tf.get_variable_scope().reuse_variables()

            batch_norm = BatchNorm(epsilon=1e-5, momentum=0.9)
            speech_features = batch_norm(speech_features, reuse=reuse, is_training=is_training)

            speech_feature_shape = speech_features.get_shape().as_list()
            speech_features_reshaped = tf.reshape(tensor=speech_features, shape=[-1, speech_feature_shape[1], 1, speech_feature_shape[2]])

            condition_shape = condition.get_shape().as_list()
            condition_reshaped = tf.reshape(tensor=condition, shape=[-1, condition_shape[1]])

            if self._condition_speech_features:
                #Condition input speech feature windows
                speech_feature_condition = tf.transpose(tf.reshape(tensor=condition_reshaped, shape=[-1, condition_shape[1], 1, 1]), perm=[0,2,3,1])
                speech_feature_condition = tf.tile(speech_feature_condition, [1, speech_feature_shape[1], 1, 1])
                speech_features_reshaped = tf.concat((speech_features_reshaped, speech_feature_condition), axis=-1, name='conditioning_speech_features')

            factor = self._speech_encoder_size_factor

            with tf.name_scope('conv1_time'):
                conv1_time = tf.nn.relu(conv2d(inputs=speech_features_reshaped,
                                                n_filters=int(32*factor),
                                                k_h=3, k_w=1,
                                                stride_h=2, stride_w=1,
                                                activation=tf.identity,
                                                scope='conv1'))
            with tf.name_scope('conv2_time'):
                conv2_time = tf.nn.relu(conv2d(inputs=conv1_time,
                                                n_filters=int(32*factor),
                                                k_h=3, k_w=1,
                                                stride_h=2, stride_w=1,
                                                activation=tf.identity,
                                                scope='conv2'))
            with tf.name_scope('conv3_time'):
                conv3_time = tf.nn.relu(conv2d(inputs=conv2_time,
                                                n_filters=int(64*factor),
                                                k_h=3, k_w=1,
                                                stride_h=2, stride_w=1,
                                                activation=tf.identity,
                                                scope='conv3'))
            with tf.name_scope('conv4_time'):
                conv4_time = tf.nn.relu(conv2d(inputs=conv3_time,
                                                n_filters=int(64*factor),
                                                k_h=3, k_w=1,
                                                stride_h=2, stride_w=1,
                                                activation=tf.identity,
                                                scope='conv4'))

            previous_shape = conv4_time.get_shape().as_list()
            time_conv_flattened = tf.reshape(conv4_time, [-1, previous_shape[1] * previous_shape[2] * previous_shape[3]])

            #Condition audio encoding on speaker style
            with tf.name_scope('concat_audio_embedding'):
                concatenated = tf.concat((time_conv_flattened, condition_reshaped), axis=1, name='conditioning_audio_embedding')
            # concatenated = time_conv_flattened

            units_in = concatenated.get_shape().as_list()[1]

            with tf.name_scope('fc1'):
                fc1 = tf.nn.tanh(fc_layer(concatenated, num_units_in=units_in, num_units_out=128, scope='fc1'))
            with tf.name_scope('fc2'):
                fc2 = fc_layer(fc1, num_units_in=128, num_units_out=self._speech_encoding_dim, scope='fc2')
            return fc2
