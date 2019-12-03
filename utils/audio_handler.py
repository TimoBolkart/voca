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

import re
import copy
import resampy
import numpy as np
import tensorflow as tf
from python_speech_features import mfcc


def interpolate_features(features, input_rate, output_rate, output_len=None):
    num_features = features.shape[1]
    input_len = features.shape[0]
    seq_len = input_len / float(input_rate)
    if output_len is None:
        output_len = int(seq_len * output_rate)
    input_timestamps = np.arange(input_len) / float(input_rate)
    output_timestamps = np.arange(output_len) / float(output_rate)
    output_features = np.zeros((output_len, num_features))
    for feat in range(num_features):
        output_features[:, feat] = np.interp(output_timestamps,
                                             input_timestamps,
                                             features[:, feat])
    return output_features

class AudioHandler:
    def __init__(self, config):
        self.config = config
        self.audio_feature_type = config['audio_feature_type']
        self.num_audio_features = config['num_audio_features']
        self.audio_window_size = config['audio_window_size']
        self.audio_window_stride = config['audio_window_stride']

    def process(self, audio):
        if self.audio_feature_type.lower() == "none":
            return None
        elif self.audio_feature_type.lower() == 'deepspeech':
            return self.convert_to_deepspeech(audio)
        else:
            raise NotImplementedError("Audio features not supported")

    def convert_to_deepspeech(self, audio):
        def audioToInputVector(audio, fs, numcep, numcontext):
            # Get mfcc coefficients
            features = mfcc(audio, samplerate=fs, numcep=numcep)

            # We only keep every second feature (BiRNN stride = 2)
            features = features[::2]

            # One stride per time step in the input
            num_strides = len(features)

            # Add empty initial and final contexts
            empty_context = np.zeros((numcontext, numcep), dtype=features.dtype)
            features = np.concatenate((empty_context, features, empty_context))

            # Create a view into the array with overlapping strides of size
            # numcontext (past) + 1 (present) + numcontext (future)
            window_size = 2 * numcontext + 1
            train_inputs = np.lib.stride_tricks.as_strided(
                features,
                (num_strides, window_size, numcep),
                (features.strides[0], features.strides[0], features.strides[1]),
                writeable=False)

            # Flatten the second and third dimensions
            train_inputs = np.reshape(train_inputs, [num_strides, -1])

            train_inputs = np.copy(train_inputs)
            train_inputs = (train_inputs - np.mean(train_inputs)) / np.std(train_inputs)

            # Return results
            return train_inputs

        if type(audio) == dict:
            pass
        else:
            raise ValueError('Wrong type for audio')

        # Load graph and place_holders
        with tf.gfile.GFile(self.config['deepspeech_graph_fname'], "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        graph = tf.get_default_graph()
        tf.import_graph_def(graph_def, name="deepspeech")
        input_tensor = graph.get_tensor_by_name('deepspeech/input_node:0')
        seq_length = graph.get_tensor_by_name('deepspeech/input_lengths:0')
        layer_6 = graph.get_tensor_by_name('deepspeech/logits:0')

        n_input = 26
        n_context = 9

        processed_audio = copy.deepcopy(audio)
        with tf.Session(graph=graph) as sess:
            for subj in audio.keys():
                for seq in audio[subj].keys():
                    print('process audio: %s - %s' % (subj, seq))

                    audio_sample = audio[subj][seq]['audio']
                    sample_rate = audio[subj][seq]['sample_rate']
                    resampled_audio = resampy.resample(audio_sample.astype(float), sample_rate, 16000)
                    input_vector = audioToInputVector(resampled_audio.astype('int16'), 16000, n_input, n_context)

                    network_output = sess.run(layer_6, feed_dict={input_tensor: input_vector[np.newaxis, ...],
                                                                  seq_length: [input_vector.shape[0]]})

                    # Resample network output from 50 fps to 60 fps
                    audio_len_s = float(audio_sample.shape[0]) / sample_rate
                    num_frames = int(round(audio_len_s * 60))
                    network_output = interpolate_features(network_output[:, 0], 50, 60,
                                                          output_len=num_frames)

                    # Make windows
                    zero_pad = np.zeros((int(self.audio_window_size / 2), network_output.shape[1]))
                    network_output = np.concatenate((zero_pad, network_output, zero_pad), axis=0)
                    windows = []
                    for window_index in range(0, network_output.shape[0] - self.audio_window_size, self.audio_window_stride):
                        windows.append(network_output[window_index:window_index + self.audio_window_size])

                    processed_audio[subj][seq]['audio'] = np.array(windows)
        return processed_audio

