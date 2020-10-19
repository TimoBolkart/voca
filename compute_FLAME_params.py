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
import chumpy as ch
from scipy.sparse.linalg import cg
from psbody.mesh import Mesh
from utils.inference import output_sequence_meshes
from smpl_webuser.serialization import load_model

parser = argparse.ArgumentParser(description='Edit VOCA motion sequences')

parser.add_argument('--source_path', default='', help='input sequence path')
parser.add_argument('--params_fname', default='', help='path of the computed parameter file')
parser.add_argument('--out_path', default='', help='FLAME meshes output path')
parser.add_argument('--flame_model_path', default='./flame/generic_model.pkl', help='path to the FLAME model')
parser.add_argument('--template_fname', default='./template/FLAME_sample.ply', help='Path of "zero pose" template mesh in" FLAME topology used for the animation')

args = parser.parse_args()
source_path = args.source_path
params_fname = args.params_fname
out_path = args.out_path
flame_model_fname = args.flame_model_path
template_fname = args.template_fname

def compute_FLAME_params(source_path, params_out_fname, flame_model_fname, template_fname):
    '''
    Load a template and an existing animation sequence in "zero pose" and compute the FLAME shape, jaw pose, and expression paramters. 
    Outputs one set of shape paramters for the entire sequence, and pose and expression parameters for each frame.
    :param source_path:         path of the animation sequence (files must be provided in OBJ file format)
    :param params_out_fname     output path of the FLAME paramters file
    :param flame_model_fname:   path of the FLAME model
    :param template_fname       "zero pose" template used to generate the sequence
    '''

    if not os.path.exists(os.path.dirname(params_out_fname)):
        os.makedirs(os.path.dirname(params_out_fname))
    
    # Load sequence files
    sequence_fnames = sorted(glob.glob(os.path.join(source_path, '*.obj')))
    num_frames = len(sequence_fnames)
    if num_frames == 0:
        print('No sequence meshes found')
        return

    model = load_model(flame_model_fname)

    print('Optimize for template identity parameters')
    template_mesh = Mesh(filename=template_fname)
    ch.minimize(template_mesh.v - model, x0=[model.betas[:300]], options={'sparse_solver': lambda A, x: cg(A, x, maxiter=2000)[0]})

    betas = model.betas.r[:300].copy()
    model.betas[:] = 0.

    model.v_template[:] = template_mesh.v
    
    model_pose = np.zeros((num_frames, model.pose.shape[0]))
    model_exp = np.zeros((num_frames, 100))

    for frame_idx in range(num_frames):
        print('Process frame %d/%d' % (frame_idx+1, num_frames))
        model.betas[:] = 0.
        model.pose[:] = 0.
        frame_vertices = Mesh(filename=sequence_fnames[frame_idx]).v
        # Optimize for jaw pose and facial expression
        ch.minimize(frame_vertices - model, x0=[model.pose[6:9], model.betas[300:]], options={'sparse_solver': lambda A, x: cg(A, x, maxiter=2000)[0]})
        model_pose[frame_idx] = model.pose.r.copy()
        model_exp[frame_idx] = model.betas.r[300:].copy()

    np.save(params_out_fname, {'shape': betas, 'pose': model_pose, 'expression': model_exp})


def output_FLAME_meshes(flame_model_fname, params_fname, out_path):
    '''
    Reconstruct meshes given a sequence of FLAME paramters
    :param flame_model_fname:   path of the FLAME model
    :param params_fname         path of the FLAME paramters file
    :param out_path:            output path of the FLAME meshes
    '''    
    
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    model = load_model(flame_model_fname)
    params = np.load(params_fname, allow_pickle=True).item()

    shape = params['shape']
    pose = params['pose']
    exp = params['expression']

    model.betas[:300] = shape

    num_frames = pose.shape[0]
    for frame_idx in range(num_frames):
        model.pose[:] = pose[frame_idx, :]
        model.betas[300:] = exp[frame_idx, :]
        out_fname = os.path.join(out_path, '%05d_FLAME.obj' % frame_idx)
        Mesh(model.r, model.f).write_obj(out_fname)

if os.path.exists(params_fname) and out_path != '':
    output_FLAME_meshes(flame_model_fname, params_fname, out_path)
else:
    compute_FLAME_params(source_path, params_fname, flame_model_fname, template_fname)