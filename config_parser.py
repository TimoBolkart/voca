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
import configparser

def set_default_paramters(config):
    config.add_section('Input Output')
    config.set('Input Output', 'checkpoint_dir', './training')
    config.set('Input Output', 'expression_basis_fname', './training_data/init_expression_basis.npy')
    config.set('Input Output', 'template_fname', './template/FLAME_sample.ply')
    config.set('Input Output', 'deepspeech_graph_fname', './ds_graph/output_graph.pb')

    config.set('Input Output', 'verts_mmaps_path', './training_data/data_verts.npy')
    config.set('Input Output', 'raw_audio_path', './training_data/raw_audio_fixed.pkl')
    config.set('Input Output', 'processed_audio_path', '')
    config.set('Input Output', 'templates_path', './training_data/templates.pkl')
    config.set('Input Output', 'data2array_verts_path', './training_data/subj_seq_to_idx.pkl')

    #Audio paramters
    config.add_section('Audio Parameters')
    config.set('Audio Parameters', 'audio_feature_type', 'deepspeech')       # deepspeech
    config.set('Audio Parameters', 'num_audio_features', '29')               # 29
    config.set('Audio Parameters', 'audio_window_size', '16')                # 16
    config.set('Audio Parameters', 'audio_window_stride', '1')               # 1
    config.set('Audio Parameters', 'condition_speech_features', 'True')      # True
    config.set('Audio Parameters', 'speech_encoder_size_factor', '1.0')      # 1

    # Model paramters
    config.add_section('Model Parameters')
    config.set('Model Parameters', 'num_vertices', '5023')                    # 5023
    config.set('Model Parameters', 'expression_dim', '50')                    # 50
    config.set('Model Parameters', 'init_expression', 'True')                 # True

    # Number of consecutive frames that are regressed in the same batch (must be >=2 if velocity is used)
    config.set('Model Parameters', 'num_consecutive_frames', '2')             # 2
    config.set('Model Parameters', 'absolute_reconstruction_loss', 'False')   # False
    config.set('Model Parameters', 'velocity_weight', '10.0')                 # 10.0
    config.set('Model Parameters', 'acceleration_weight', '0.0')              # 0.0
    config.set('Model Parameters', 'verts_regularizer_weight', '0.0')         # 0.0

    config.add_section('Data Setup')
    config.set('Data Setup', 'subject_for_training',
               "FaceTalk_170728_03272_TA FaceTalk_170904_00128_TA FaceTalk_170725_00137_TA FaceTalk_170915_00223_TA"
               " FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA ")
    config.set('Data Setup', 'sequence_for_training',
                "sentence01 sentence02 sentence03 sentence04 sentence05 sentence06 sentence07 sentence08 sentence09 sentence10 "
                "sentence11 sentence12 sentence13 sentence14 sentence15 sentence16 sentence17 sentence18 sentence19 sentence20 "
                "sentence21 sentence22 sentence23 sentence24 sentence25 sentence26 sentence27 sentence28 sentence29 sentence30 "
                "sentence31 sentence32 sentence33 sentence34 sentence35 sentence36 sentence37 sentence38 sentence39 sentence40")
    config.set('Data Setup', 'subject_for_validation', "FaceTalk_170811_03275_TA FaceTalk_170908_03277_TA")
    config.set('Data Setup', 'sequence_for_validation',
                "sentence21 sentence22 sentence23 sentence24 sentence25 sentence26 sentence27 sentence28 sentence29 sentence30 "
                "sentence31 sentence32 sentence33 sentence34 sentence35 sentence36 sentence37 sentence38 sentence39 sentence40")
    config.set('Data Setup', 'subject_for_testing', "FaceTalk_170809_00138_TA FaceTalk_170731_00024_TA ")
    config.set('Data Setup', 'sequence_for_testing',
                "sentence21 sentence22 sentence23 sentence24 sentence25 sentence26 sentence27 sentence28 sentence29 sentence30 "
                "sentence31 sentence32 sentence33 sentence34 sentence35 sentence36 sentence37 sentence38 sentence39 sentence40")

    config.add_section('Learning Parameters')
    config.set('Learning Parameters', 'batch_size', '64')                     # 64
    config.set('Learning Parameters', 'learning_rate', '1e-4')                # 1e-4
    config.set('Learning Parameters', 'decay_rate', '1.0')                    # 1.0
    config.set('Learning Parameters', 'epoch_num', '100')                     # 100
    config.set('Learning Parameters', 'adam_beta1_value', '0.9')              # 0.9

    config.add_section('Visualization Parameters')
    config.set('Visualization Parameters', 'num_render_sequences', '3')


def create_default_config(fname):
    config = configparser.ConfigParser()
    set_default_paramters(config)

    with open(fname, 'w') as configfile:
        config.write(configfile)
        configfile.close()

def read_config(fname):
    if not os.path.exists(fname):
        print('Config not found %s' % fname)
        return

    config = configparser.RawConfigParser()
    config.read(fname)

    config_parms = {}
    config_parms['checkpoint_dir'] = config.get('Input Output', 'checkpoint_dir')
    config_parms['expression_basis_fname'] = config.get('Input Output', 'expression_basis_fname')
    config_parms['template_fname'] = config.get('Input Output', 'template_fname')
    config_parms['deepspeech_graph_fname'] = config.get('Input Output', 'deepspeech_graph_fname')

    config_parms['verts_mmaps_path'] = config.get('Input Output', 'verts_mmaps_path')
    config_parms['raw_audio_path'] = config.get('Input Output', 'raw_audio_path')
    config_parms['processed_audio_path'] = config.get('Input Output', 'processed_audio_path')
    config_parms['templates_path'] = config.get('Input Output', 'templates_path')
    config_parms['data2array_verts_path'] = config.get('Input Output', 'data2array_verts_path')

    config_parms['audio_feature_type'] = config.get('Audio Parameters', 'audio_feature_type')
    config_parms['num_audio_features'] = config.getint('Audio Parameters', 'num_audio_features')
    config_parms['audio_window_size'] = config.getint('Audio Parameters', 'audio_window_size')
    config_parms['audio_window_stride'] = config.getint('Audio Parameters', 'audio_window_stride')
    config_parms['condition_speech_features'] = config.getboolean('Audio Parameters', 'condition_speech_features')
    config_parms['speech_encoder_size_factor'] = config.getfloat('Audio Parameters', 'speech_encoder_size_factor')


    config_parms['num_vertices'] = config.getint('Model Parameters', 'num_vertices')
    config_parms['expression_dim'] = config.getint('Model Parameters', 'expression_dim')
    config_parms['init_expression'] = config.getboolean('Model Parameters', 'init_expression')

    config_parms['num_consecutive_frames'] = config.getint('Model Parameters', 'num_consecutive_frames')
    config_parms['absolute_reconstruction_loss'] = config.getboolean('Model Parameters', 'absolute_reconstruction_loss')
    config_parms['velocity_weight'] = config.getfloat('Model Parameters', 'velocity_weight')
    config_parms['acceleration_weight'] = config.getfloat('Model Parameters', 'acceleration_weight')
    config_parms['verts_regularizer_weight'] = config.getfloat('Model Parameters', 'verts_regularizer_weight')

    config_parms['subject_for_training'] = config.get('Data Setup', 'subject_for_training')
    config_parms['sequence_for_training'] = config.get('Data Setup', 'sequence_for_training')
    config_parms['subject_for_validation'] = config.get('Data Setup', 'subject_for_validation')
    config_parms['sequence_for_validation'] = config.get('Data Setup', 'sequence_for_validation')
    config_parms['subject_for_testing'] = config.get('Data Setup', 'subject_for_testing')
    config_parms['sequence_for_testing'] = config.get('Data Setup', 'sequence_for_testing')

    config_parms['batch_size'] = config.getint('Learning Parameters', 'batch_size')
    config_parms['learning_rate'] = config.getfloat('Learning Parameters', 'learning_rate')
    config_parms['decay_rate'] = config.getfloat('Learning Parameters', 'decay_rate')
    config_parms['epoch_num'] = config.getint('Learning Parameters', 'epoch_num')
    config_parms['adam_beta1_value'] = config.getfloat('Learning Parameters', 'adam_beta1_value')

    config_parms['num_render_sequences'] = config.getint('Visualization Parameters', 'num_render_sequences')
    return config_parms

if __name__ == '__main__':
    pkg_path, _ = os.path.split(os.path.realpath(__file__))
    config_fname = os.path.join(pkg_path, 'training_config.cfg')

    print('Writing default config file - %s' % config_fname)
    create_default_config(config_fname)
