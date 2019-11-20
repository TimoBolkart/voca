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

from __future__ import absolute_import
from __future__ import division

import tensorflow as tf

def reconstruction_loss(predicted, real, want_absolute_loss=True, want_in_mm=False, weights=None):
    if weights is not None:
        assert predicted.shape[1] == real.shape[1] == weights.shape[0]
        tf_weights = tf.constant(weights, dtype=tf.float32)
        predicted = tf.einsum('abcd,bd->abcd', predicted, tf_weights)
        real = tf.einsum('abcd,bd->abcd', real, tf_weights)

    if want_in_mm:
        predicted, real = predicted * 1000, real * 1000
    if want_absolute_loss:
        return tf.reduce_mean(tf.reduce_sum(tf.abs(tf.subtract(predicted, real)), axis=2))
    else:
        return tf.reduce_mean(tf.reduce_sum(tf.squared_difference(predicted, real), axis=2))

def wing_reconstruction_loss(predicted, real):
    pass

def edge_reconstruction_loss(predicted, real, num_vertices, mesh_f, want_absolute_loss=True):
    from tfbody.mesh.utils import get_vertices_per_edge
    vpe = get_vertices_per_edge(num_vertices, mesh_f)
    edges_for = lambda x: tf.gather(x, vpe[:, 0], axis=1) - tf.gather(x, vpe[:, 1], axis=1)
    if want_absolute_loss:
        return tf.reduce_mean(tf.reduce_sum(tf.abs(tf.subtract(edges_for(predicted), edges_for(real))), axis=2))
    else:
        return tf.reduce_mean(tf.reduce_sum(tf.squared_difference(edges_for(predicted), edges_for(real)), axis=2))

def orthogonality_loss(W, want_absolute_loss=True, no_normalize=False):
    '''Computes the loss to a column-wise orthogonality. If no_normalize is true, the loss reflects
    the distance to orthogonalize the matrix columns <a,b> = 0 but the columnts are not orthonormal,
    i.e. <a,a> != 1'''
    wtw = tf.matmul(tf.transpose(W), W)

    if no_normalize:
        diag = tf.diag(tf.diag_part(wtw))
    else:
        num_rows = wtw.get_shape().as_list()[0]
        diag = tf.eye(num_rows)

    if want_absolute_loss:
        return tf.reduce_sum(tf.abs(tf.subtract(wtw, diag)))
    else:
        return tf.reduce_sum(tf.squared_difference((wtw, diag)))