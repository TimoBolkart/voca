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

import os
import pickle
import random
import numpy as np
from utils.audio_handler import AudioHandler

def load_from_config(config, key):
    if key in config:
        return config[key]
    else:
        raise ValueError('Key does not exist in config %s' % key)

def invert_data2array(data2array):
    array2data = {}
    for sub in data2array.keys():
        for seq in data2array[sub].keys():
            for frame, array_idx in data2array[sub][seq].items():
                array2data[array_idx] = (sub, seq, frame)
    return array2data

def compute_window_array_idx(data2array, window_size):
    def window_frame(frame_idx, window_size):
        l0 = max(frame_idx + 1 - window_size, 0)
        l1 = frame_idx + 1
        window_frames = np.zeros(window_size, dtype=int)
        window_frames[window_size - l1 + l0:] = np.arange(l0, l1)
        return window_frames

    array2window_ids = {}
    for sub in data2array.keys():
        for seq in data2array[sub].keys():
            for frame, array_idx in data2array[sub][seq].items():
                window_frames = window_frame(frame, window_size)
                array2window_ids[array_idx] = [data2array[sub][seq][id] for id in window_frames]
    return array2window_ids

class DataHandler:
    def __init__(self, config):
        subject_for_training = config['subject_for_training'].split(" ")
        sequence_for_training = config['sequence_for_training'].split(" ")
        subject_for_validation = config['subject_for_validation'].split(" ")
        sequence_for_validation = config['sequence_for_validation'].split(" ")
        subject_for_testing = config['subject_for_testing'].split(" ")
        sequence_for_testing = config['sequence_for_testing'].split(" ")
        self.num_consecutive_frames = config['num_consecutive_frames']

        self.audio_handler = AudioHandler(config)
        print("Loading data")
        self._load_data(config)
        print("Initialize data splits")
        self._init_data_splits(subject_for_training, sequence_for_training, subject_for_validation,
                               sequence_for_validation, subject_for_testing, sequence_for_testing)
        print("Initialize training, validation, and test indices")
        self._init_indices()

    def get_data_splits(self):
        return self.training_indices, self.validation_indices, self.testing_indices

    def slice_data(self, indices):
        return self._slice_data(indices)

    def get_training_sequences(self, num_sequences):
        return self._get_random_sequences(self.training_subjects, self.training_sequences, num_sequences)

    def get_validation_sequences(self, num_sequences):
        return self._get_random_sequences(self.validation_subjects, self.validation_sequences, num_sequences)

    def get_testing_sequences(self, num_sequences):
        return self._get_random_sequences(self.testing_subjects, self.testing_sequences, num_sequences)

    def get_num_training_subjects(self):
        return len(self.training_subjects)

    def convert_training_idx2subj(self, idx):
        if idx in self.training_idx2subj:
            return self.training_idx2subj[idx]
        else:
            return -1

    def convert_training_subj2idx(self, subj):
        if subj in self.training_subj2idx:
            return self.training_subj2idx[subj]
        else:
            return -1

    def _init_indices(self):
        def get_indices(subjects, sequences):
            indices = []
            for subj in subjects:
                if (subj not in self.raw_audio) or (subj not in self.data2array_verts):
                    if subj != '':
                        import pdb; pdb.set_trace()
                        print('subject missing %s' % subj)
                    continue

                for seq in sequences:
                    if (seq not in self.raw_audio[subj]) or (seq not in self.data2array_verts[subj]):
                        print('sequence data missing %s - %s' % (subj, seq))
                        continue

                    num_data_frames = max(self.data2array_verts[subj][seq].keys())+1
                    if self.processed_audio is not None:
                        num_audio_frames = len(self.processed_audio[subj][seq]['audio'])
                    else:
                        num_audio_frames = num_data_frames

                    try:
                        for i in range(min(num_data_frames, num_audio_frames)):
                            indexed_frame = self.data2array_verts[subj][seq][i]
                            indices.append(indexed_frame)
                    except KeyError:
                        print('Key error with subject: %s and sequence: %s" % (subj, seq)')
            return indices

        self.training_indices = get_indices(self.training_subjects, self.training_sequences)
        self.validation_indices = get_indices(self.validation_subjects, self.validation_sequences)
        self.testing_indices = get_indices(self.testing_subjects, self.testing_sequences)

        self.training_idx2subj = {idx: self.training_subjects[idx] for idx in np.arange(len(self.training_subjects))}
        self.training_subj2idx = {self.training_idx2subj[idx]: idx for idx in self.training_idx2subj.keys()}

    def _slice_data(self, indices):
        if self.num_consecutive_frames == 1:
            return self._slice_data_helper(indices)
        else:
            window_indices = []
            for id in indices:
                window_indices += self.array2window_ids[id]
            return self._slice_data_helper(window_indices)

    def _slice_data_helper(self, indices):
        face_vertices = self.face_vert_mmap[indices]
        face_templates = []
        processed_audio = []
        subject_idx = []
        for idx in indices:
            sub, sen, frame = self.array2data_verts[idx]
            face_templates.append(self.templates_data[sub])
            if self.processed_audio is not None:
                processed_audio.append(self.processed_audio[sub][sen]['audio'][frame])
            subject_idx.append(self.convert_training_subj2idx(sub))

        face_templates = np.stack(face_templates)
        subject_idx = np.hstack(subject_idx)
        assert face_vertices.shape[0] == face_templates.shape[0]

        if self.processed_audio is not None:
            processed_audio = np.stack(processed_audio)
            assert face_vertices.shape[0] == processed_audio.shape[0]
        return processed_audio, face_vertices, face_templates, subject_idx

    def _load_data(self, config):
        face_verts_mmaps_path = load_from_config(config, 'verts_mmaps_path')
        face_templates_path = load_from_config(config, 'templates_path')
        raw_audio_path = load_from_config(config, 'raw_audio_path')
        processed_audio_path = load_from_config(config, 'processed_audio_path')
        data2array_verts_path = load_from_config(config, 'data2array_verts_path')

        print("Loading face vertices")
        self.face_vert_mmap = np.load(face_verts_mmaps_path, mmap_mode='r')

        print("Loading templates")
        self.templates_data = pickle.load(open(face_templates_path, 'rb'), encoding='latin1')

        print("Loading raw audio")
        self.raw_audio = pickle.load(open(raw_audio_path, 'rb'), encoding='latin1')

        print("Process audio")
        if os.path.exists(processed_audio_path):
            self.processed_audio = pickle.load(open(processed_audio_path, 'rb'), encoding='latin1')
        else:
            self.processed_audio =  self._process_audio(self.raw_audio)
            if processed_audio_path != '':
                pickle.dump(self.processed_audio, open(processed_audio_path, 'wb'))

        print("Loading index maps")
        self.data2array_verts = pickle.load(open(data2array_verts_path, 'rb'))
        self.array2data_verts = invert_data2array(self.data2array_verts)
        self.array2window_ids = compute_window_array_idx(self.data2array_verts, self.num_consecutive_frames)

    def _init_data_splits(self, subject_for_training, sequence_for_training, subject_for_validation,
                          sequence_for_validation, subject_for_testing, sequence_for_testing):
        def select_valid_subjects(subjects_list):
            return [subj for subj in subjects_list]

        def select_valid_sequences(sequences_list):
            return [seq for seq in sequences_list]

        self.training_subjects = select_valid_subjects(subject_for_training)
        self.training_sequences = select_valid_sequences(sequence_for_training)

        self.validation_subjects = select_valid_subjects(subject_for_validation)
        self.validation_sequences = select_valid_sequences(sequence_for_validation)

        self.testing_subjects = select_valid_subjects(subject_for_testing)
        self.testing_sequences = select_valid_sequences(sequence_for_testing)

        all_instances = []
        for i in self.training_subjects:
            for j in self.training_sequences:
                all_instances.append((i, j))
        for i in self.validation_subjects:
            for j in self.validation_sequences:
                all_instances.append((i, j))
        for i in self.testing_subjects:
            for j in self.testing_sequences:
                all_instances.append((i, j))

        # All instances should contain all unique elements, otherwise the arguments were passed wrongly, so assertion
        if len(all_instances) != len(set(all_instances)):
            raise ValueError('User-specified data split not disjoint')

    def _get_random_sequences(self, subjects, sequences, num_sequences):
        if num_sequences == 0:
            return

        sub_seq_list = []
        for subj in subjects:
            if subj not in self.data2array_verts:
                continue
            for seq in sequences:
                if(seq not in self.raw_audio[subj]) or (seq not in self.data2array_verts[subj]):
                    continue
                sub_seq_list.append((subj, seq))
        st = random.getstate()
        random.seed(777)
        random.shuffle(sub_seq_list)
        random.setstate(st)

        if num_sequences > 0 and num_sequences < len(sub_seq_list):
            sub_seq_list = sub_seq_list[:num_sequences]
        return self._get_subject_sequences(sub_seq_list)

    def _get_subject_sequences(self, subject_sequence_list):
        face_vertices = []
        face_templates = []
        subject_idx = []

        raw_audio = []
        processed_audio = []
        for subj, seq in subject_sequence_list:
            frame_array_indices = []
            try:
                for frame, array_idx in self.data2array_verts[subj][seq].items():
                    frame_array_indices.append(array_idx)
            except KeyError:
                continue
            face_vertices.append(self.face_vert_mmap[frame_array_indices])
            face_templates.append(self.templates_data[subj])
            subject_idx.append(self.convert_training_subj2idx(subj))
            raw_audio.append(self.raw_audio[subj][seq])
            processed_seq_audio = []
            if self.processed_audio is not None:
                for frame, array_idx in self.data2array_verts[subj][seq].items():
                    processed_seq_audio.append(self.processed_audio[subj][seq]['audio'][frame])
            processed_audio.append(processed_seq_audio)
        return raw_audio, processed_audio, face_vertices, face_templates, subject_idx

    def _process_audio(self, raw_audio):
        return self.audio_handler.process(raw_audio)