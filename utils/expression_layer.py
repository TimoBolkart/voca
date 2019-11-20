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

import numpy as np
import tensorflow as tf
from utils.ops import fc_layer

class ExpressionLayer:
    def __init__(self, config, scope='ExpressionLayer'):
        self.expression_basis_fname = config['expression_basis_fname']
        self.init_expression = config['init_expression']

        self.num_vertices = config['num_vertices']
        self.expression_dim = config['expression_dim']
        self.scope = scope

    def __call__(self, parameters, if_reuse=False):
        with tf.variable_scope(self.scope, reuse=if_reuse):

            init_exp_basis = np.zeros((3*self.num_vertices, self.expression_dim))

            if self.init_expression:
                init_exp_basis[:, :min(self.expression_dim, 100)] = np.load(self.expression_basis_fname)[:, :min(self.expression_dim, 100)]

            with tf.name_scope('expression_offset'):
                exp_offset = fc_layer(parameters,
                                    num_units_in=self.expression_dim,
                                    num_units_out=3*self.num_vertices,
                                    init_weights=init_exp_basis.T,
                                    scope='expression_offset')
            return tf.reshape(exp_offset, [-1, self.num_vertices, 3, 1])