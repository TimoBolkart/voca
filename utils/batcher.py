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

from __future__ import division

import random
import copy

import numpy as np


class Batcher:
    def __init__(self, data_handler):

        self.data_handler = data_handler

        data_splits = data_handler.get_data_splits()
        self.training_indices = copy.deepcopy(data_splits[0])
        self.val_indices = copy.deepcopy(data_splits[1])
        self.test_indices = copy.deepcopy(data_splits[2])

        self.current_state = 0

    def get_training_size(self):
        return len(self.training_indices)

    def get_num_training_subjects(self):
        return self.data_handler.get_num_training_subjects()

    def convert_training_idx2subj(self, idx):
        return self.data_handler.convert_training_idx2subj(idx)

    def convert_training_subj2idx(self, subj):
        return self.data_handler.convert_training_subj2idx(subj)

    def get_training_batch(self, batch_size):
        """
        Get batch for training
        :param batch_size:
        :return:
        """
        if self.current_state == 0:
            random.shuffle(self.training_indices)

        if (self.current_state + batch_size) > (len(self.training_indices) + 1):
            self.current_state = 0
            return self.get_training_batch(batch_size)
        else:
            self.current_state += batch_size
            batch_indices = self.training_indices[self.current_state:(self.current_state + batch_size)]
            if len(batch_indices) != batch_size:
                self.current_state = 0
                return self.get_training_batch(batch_size)
            return self.data_handler.slice_data(batch_indices)

    def get_validation_batch(self, batch_size):
        """
        Validation batch for randomize, quantitative evaluation
        :param batch_size:
        :return:
        """
        if batch_size > len(self.val_indices):
            return self.data_handler.slice_data(self.val_indices)
        else:
            return self.data_handler.slice_data(list(np.random.choice(self.val_indices, size=batch_size)))

    def get_test_batch(self, batch_size):
        if batch_size > len(self.test_indices):
            return self.data_handler.slice_data(self.test_indices)
        else:
            return self.data_handler.slice_data(list(np.random.choice(self.test_indices, size=batch_size)))

    def get_num_batches(self, batch_size):
        return int(len(self.training_indices) / batch_size)

    def get_training_sequences_in_order(self, num_of_sequences):
        return self.data_handler.get_training_sequences(num_of_sequences)

    def get_validation_sequences_in_order(self, num_of_sequences):
        return self.data_handler.get_validation_sequences(num_of_sequences)
