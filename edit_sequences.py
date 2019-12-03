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
parser.add_argument('--mode', default='shape', help='edit shape, head pose, or add eye blinks')
parser.add_argument('--index', type=int, default=0, help='parameter to be varied')
parser.add_argument('--max_variation', type=float, default=1.0, help='maximum variation')
parser.add_argument('--num_blinks', type=int, default=1, help='number of eye blinks')
parser.add_argument('--blink_duration', type=int, default=15, help='blink_duration')

args = parser.parse_args()
source_path = args.source_path
out_path = args.out_path
flame_model_fname = args.flame_model_path

def add_eye_blink(source_path, out_path, flame_model_fname, num_blinks, blink_duration):
    '''
    Load existing animation sequence in "zero pose" and add eye blinks over time
    :param source_path:         path of the animation sequence (files must be provided in OBJ file format)
    :param out_path:            output path of the altered sequence
    :param flame_model_fname:   path of the FLAME model
    :param num_blinks:          number of blinks within the sequence
    :param blink_duration:      duration of a blink in number of frames
    '''

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

    blink_exp_betas = np.array(
        [0.04676158497927314, 0.03758675711005459, -0.8504121184951298, 0.10082324210507627, -0.574142329926028,
         0.6440016589938355, 0.36403779939335984, 0.21642312586261656, 0.6754551784690193, 1.80958618462892,
         0.7790133813372259, -0.24181691256476057, 0.826280685961679, -0.013525679499256753, 1.849393698014113,
         -0.263035686247264, 0.42284248271332153, 0.10550891351425384, 0.6720993875023772, 0.41703592560736436,
         3.308019065485072, 1.3358509602858895, 1.2997143108969278, -1.2463587328652894, -1.4818961382824924,
         -0.6233880069345369, 0.26812528424728455, 0.5154889093160832, 0.6116267181402183, 0.9068826814583771,
         -0.38869613253448576, 1.3311776710005476, -0.5802565274559162, -0.7920775624092143, -1.3278601781150017,
         -1.2066425872386706, 0.34250140710360893, -0.7230686724732668, -0.6859285483325263, -1.524877347586566,
         -1.2639479212965923, -0.019294228307535275, 0.2906175769381998, -1.4082782880837976, 0.9095436721066045,
         1.6007365724960054, 2.0302381182163574, 0.5367600947801505, -0.12233184771794232, -0.506024823810769,
         2.4312326730634783, 0.5622323258974669, 0.19022395712837198, -0.7729758559103581, -1.5624233513002923,
         0.8275863297957926, 1.1661887586553132, 1.2299311381779416, -1.4146929897142397, -0.42980549225554004,
         -1.4282801579740614, 0.26172301287347266, -0.5109318114918897, -0.6399495909195524, -0.733476856285442,
         1.219652074726591, 0.08194907995352405, 0.4420398361785991, -1.184769973221183, 1.5126082924326332,
         0.4442281271081217, -0.005079477284341147, 1.764084274265486, 0.2815940264026848, 0.2898827213634057,
         -0.3686662696397026, 1.9125365942683656, 2.1801452989500274, -2.3915065327980467, 0.5794919897154226,
         -1.777680085517591, 2.9015718628823604, -2.0516886588315777, 0.4146899057365943, -0.29917763685660903,
         -0.5839240983516372, 2.1592457102697007, -0.8747902386178202, -0.5152943072876817, 0.12620001057735733,
         1.3144109838803493, -0.5027032013330108, 1.2160353388774487, 0.7543834001473375, -3.512095548974531,
         -0.9304382646186183, -0.30102930208709433, 0.9332135959962723, -0.52926196689098, 0.23509772959302958])

    step = blink_duration//3
    blink_weights = np.hstack((np.interp(np.arange(step), [0,step], [0,1]), np.ones(step), np.interp(np.arange(step), [0,step], [1,0])))

    frequency = num_frames // (num_blinks+1)
    weights = np.zeros(num_frames)
    for i in range(num_blinks):
        x1 = (i+1)*frequency-blink_duration//2
        x2 = x1+3*step
        if x1 >= 0 and x2 < weights.shape[0]:
            weights[x1:x2] = blink_weights

    predicted_vertices = np.zeros((num_frames, model.v_template.shape[0], model.v_template.shape[1]))

    for frame_idx in range(num_frames):
        model.v_template[:] = Mesh(filename=sequence_fnames[frame_idx]).v
        model.betas[300:] = weights[frame_idx]*blink_exp_betas
        predicted_vertices[frame_idx] = model.r

    output_sequence_meshes(predicted_vertices, Mesh(model.v_template, model.f), out_path)


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
    x1, y1 = [0, num_frames//4], [0, rot_angle]
    x2, y2 = [num_frames//4, num_frames//2], [rot_angle, 0]
    x3, y3 = [num_frames//2, 3*num_frames//4], [0, -rot_angle]
    x4, y4 = [3*num_frames//4, num_frames], [-rot_angle, 0]

    xsteps1 = np.arange(0, num_frames//4)
    xsteps2 = np.arange(num_frames//4, num_frames/2)
    xsteps3 = np.arange(num_frames//2, 3*num_frames//4)
    xsteps4 = np.arange(3*num_frames//4, num_frames)

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
elif(args.mode == 'blink'):
    num_blinks = args.num_blinks
    blink_duration = args.blink_duration
    add_eye_blink(source_path, out_path, flame_model_fname, num_blinks, blink_duration)
else:
    print('Unknown mode')