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
import numpy as np
from subprocess import call
from psbody.mesh import Mesh
from utils.inference import render_sequence_meshes

parser = argparse.ArgumentParser(description='Sequence visualization')
parser.add_argument('--sequence_path', default='./animation_output', help='Path to motion sequence')
parser.add_argument('--audio_fname', default='', help='Path of speech sequence')
parser.add_argument('--out_path', default='./animation_visualization', help='Output path')


args = parser.parse_args()
sequence_path = args.sequence_path
audio_fname = args.audio_fname
out_path = args.out_path

sequence_fnames = sorted(glob.glob(os.path.join(sequence_path, '*.obj')))
if len(sequence_fnames) == 0:
    print('No meshes found')

sequence_vertices = []
f = None
for frame_idx, mesh_fname in enumerate(sequence_fnames):
    frame = Mesh(filename=mesh_fname)
    sequence_vertices.append(frame.v)
    if f is None:
        f = frame.f
template = Mesh(sequence_vertices[0], f)
sequence_vertices = np.stack(sequence_vertices)
render_sequence_meshes(audio_fname, sequence_vertices, template, out_path)