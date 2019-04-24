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
from psbody.mesh import Mesh
from utils.inference import output_sequence_meshes
from smpl_webuser.serialization import load_model

parser = argparse.ArgumentParser(description='Edit VOCA motion sequences')

parser.add_argument('--source_path', default='', help='input sequence path')
parser.add_argument('--out_path', default='', help='output path')
parser.add_argument('--flame_model_path', default='./flame/generic_model.pkl', help='path to the FLAME model')
parser.add_argument('--mode', default='shape', help='edit shape or head pose')
parser.add_argument('--index', type=int, default=0, help='parameter to be varied')
parser.add_argument('--max_variation', type=float, default=1.0, help='maximum variation')

args = parser.parse_args()
source_path = args.source_path
out_path = args.out_path
flame_model_fname = args.flame_model_path

def alter_sequence_shape(source_path, out_path, flame_model_fname, pc_idx=0, pc_range=(0,3)):
    '''
    Load existing animation sequence in "zero pose" and change the identity dependent shape over time.
    :param source_path:         path of the animation sequence (files must be provided in OBJ file format)
    :param out_path:            output path of the altered sequence
    :param flame_model_fname:   path of the FLAME model
    :param pc_idx               Identity shape parameter to be varied in [0,300) as FLAME provides 300 shape paramters
    :param pc_range             Tuple (start/end, max/min) defining the range of the shape variation.
                                i.e. (0,3) varies the shape from 0 to 3 stdev and back to 0
    '''

    if pc_idx < 0 or pc_idx >= 300:
        print('shape parameter index out of range [0,300)')
        return

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # Load sequence files
    sequence_fnames = sorted(glob.glob(os.path.join(source_path, '*.obj')))
    num_frames = len(sequence_fnames)
    if num_frames == 0:
        print('No sequence meshes found')
        return

    # Load FLAME head model
    model = load_model(flame_model_fname)
    model_parms = np.zeros((num_frames, 300))

    # Generate interpolated shape parameters for each frame
    x1, y1 = [0, num_frames/2], pc_range
    x2, y2 = [num_frames/2, num_frames], pc_range[::-1]

    xsteps1 = np.arange(0, num_frames/2)
    xsteps2 = np.arange(num_frames/2, num_frames)

    model_parms[:, pc_idx] = np.hstack((np.interp(xsteps1, x1, y1), np.interp(xsteps2, x2, y2)))

    predicted_vertices = np.zeros((num_frames, model.v_template.shape[0], model.v_template.shape[1]))

    for frame_idx in range(num_frames):
        model.v_template[:] = Mesh(filename=sequence_fnames[frame_idx]).v
        model.betas[:300] = model_parms[frame_idx]
        predicted_vertices[frame_idx] = model.r

    output_sequence_meshes(predicted_vertices, Mesh(model.v_template, model.f), out_path)

def alter_sequence_head_pose(source_path, out_path, flame_model_fname, pose_idx=3, rot_angle=np.pi/6):
    '''
    Load existing animation sequence in "zero pose" and change the head pose (i.e. rotation around the neck) over time.
    :param source_path:         path of the animation sequence (files must be provided in OBJ file format)
    :param out_path:            output path of the altered sequence
    :param flame_model_fname:   path of the FLAME model
    :param pose_idx:            head pose parameter to be varied in [3,6)
    :param rot_angle:           maximum rotation angle in [0,2pi)
    '''

    if pose_idx < 3 or pose_idx >= 6:
        print('pose parameter index out of range [3,6)')
        return

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # Load sequence files
    sequence_fnames = sorted(glob.glob(os.path.join(source_path, '*.obj')))
    num_frames = len(sequence_fnames)
    if num_frames == 0:
        print('No sequence meshes found')
        return

    # Load FLAME head model
    model = load_model(flame_model_fname)
    model_parms = np.zeros((num_frames, model.pose.shape[0]))

    # Generate interpolated pose parameters for each frame
    x1, y1 = [0, num_frames/4], [0, rot_angle]
    x2, y2 = [num_frames/4, num_frames/2], [rot_angle, 0]
    x3, y3 = [num_frames/2, 3*num_frames/4], [0, -rot_angle]
    x4, y4 = [3*num_frames/4, num_frames], [-rot_angle, 0]

    xsteps1 = np.arange(0, num_frames/4)
    xsteps2 = np.arange(num_frames/4, num_frames/2)
    xsteps3 = np.arange(num_frames/2, 3*num_frames/4)
    xsteps4 = np.arange(3*num_frames/4, num_frames)

    model_parms[:, pose_idx] = np.hstack((np.interp(xsteps1, x1, y1),
                                   np.interp(xsteps2, x2, y2),
                                   np.interp(xsteps3, x3, y3),
                                   np.interp(xsteps4, x4, y4)))

    predicted_vertices = np.zeros((num_frames, model.v_template.shape[0], model.v_template.shape[1]))

    for frame_idx in range(num_frames):
        model.v_template[:] = Mesh(filename=sequence_fnames[frame_idx]).v
        model.pose[:] = model_parms[frame_idx]
        predicted_vertices[frame_idx] = model.r

    output_sequence_meshes(predicted_vertices, Mesh(model.v_template, model.f), out_path)

if(args.mode == 'shape'):
    pc_idx = args.index
    pc_range = (0, args.max_variation)
    alter_sequence_shape(source_path, out_path, flame_model_fname, pc_idx=pc_idx, pc_range=pc_range)
elif(args.mode == 'pose'):
    pose_idx = args.index
    rot_angle = args.max_variation
    alter_sequence_head_pose(source_path, out_path, flame_model_fname, pose_idx=pose_idx, rot_angle=rot_angle)
else:
    print('Unknown mode')