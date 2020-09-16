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
import glob
import argparse
from utils.inference import inference

def str2bool(val):
    if isinstance(val, bool):
        return val
    elif isinstance(val, str):
        if val.lower() in ['true', 't', 'yes', 'y']:
            return True
        elif val.lower() in ['false', 'f', 'no', 'n']:
            return False
    return False

parser = argparse.ArgumentParser(description='Voice operated character animation')
parser.add_argument('--tf_model_fname', default='./model/gstep_52280.model', help='Path to trained VOCA model')
parser.add_argument('--ds_fname', default='./ds_graph/output_graph.pb', help='Path to trained DeepSpeech model')
parser.add_argument('--audio_fname', default='./audio/test_sentence.wav', help='Path of input speech sequence')
parser.add_argument('--template_fname', default='./template/FLAME_sample.ply', help='Path of "zero pose" template mesh in" FLAME topology to be animated')
parser.add_argument('--condition_idx', type=int, default=3, help='Subject condition id in [1,8]')
parser.add_argument('--uv_template_fname', default='', help='Path of a FLAME template with UV coordinates')
parser.add_argument('--texture_img_fname', default='', help='Path of the texture image')
parser.add_argument('--out_path', default='./voca/animation_output', help='Output path')
parser.add_argument('--visualize', default='True', help='Visualize animation')

args = parser.parse_args()
tf_model_fname = args.tf_model_fname
ds_fname = args.ds_fname
audio_fname = args.audio_fname
template_fname = args.template_fname
condition_idx = args.condition_idx
out_path = args.out_path

uv_template_fname = args.uv_template_fname
texture_img_fname = args.texture_img_fname

if not os.path.exists(out_path):
    os.makedirs(out_path)

inference(tf_model_fname, ds_fname, audio_fname, template_fname, condition_idx, out_path, str2bool(args.visualize), uv_template_fname=uv_template_fname, texture_img_fname=texture_img_fname)

