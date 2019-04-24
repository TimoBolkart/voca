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
from subprocess import call
from psbody.mesh import Mesh
from psbody.mesh.meshviewer import MeshViewer


parser = argparse.ArgumentParser(description='Sequence visualization')
parser.add_argument('--sequence_path', default='./animation_output', help='Path to motion sequence')
parser.add_argument('--audio_fname', default='', help='Path of speech sequence')
parser.add_argument('--out_path', default='./animation_visualization', help='Output path')


args = parser.parse_args()
sequence_path = args.sequence_path
audio_fname = args.audio_fname
out_path = args.out_path

img_path = os.path.join(out_path, 'img')
if not os.path.exists(img_path):
    os.makedirs(img_path)

mv = MeshViewer()

sequence_fnames = sorted(glob.glob(os.path.join(sequence_path, '*.obj')))
if len(sequence_fnames) == 0:
    print('No meshes found')

# Render images
for frame_idx, mesh_fname in enumerate(sequence_fnames):
    frame_mesh = Mesh(filename=mesh_fname)
    mv.set_dynamic_meshes([frame_mesh], blocking=True)

    img_fname = os.path.join(img_path, '%05d.png' % frame_idx)
    mv.save_snapshot(img_fname)

# Encode images to video
cmd_audio = []
if os.path.exists(audio_fname):
    cmd_audio += ['-i', audio_fname]

out_video_fname = os.path.join(out_path, 'video.mp4')
cmd = ['ffmpeg', '-framerate', '60', '-pattern_type', 'glob', '-i', os.path.join(img_path, '*.png')] + cmd_audio + [out_video_fname]
call(cmd)