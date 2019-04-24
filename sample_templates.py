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
import argparse
import numpy as np
from psbody.mesh import Mesh
from smpl_webuser.serialization import load_model


parser = argparse.ArgumentParser(description='Sample templates from FLAME')
parser.add_argument('--flame_model_path', default='./flame/generic_model.pkl', help='path to the FLAME model')
parser.add_argument('--num_samples', type=int, default=1, help='Number of samples')
parser.add_argument('--out_path', default='./template', help='Output path')

args = parser.parse_args()
flame_model_fname = args.flame_model_path
num_samples = args.num_samples
out_path = args.out_path

if not os.path.exists(out_path):
    os.makedirs(out_path)

out_fname = lambda num : os.path.join(out_path, 'FLAME_sample_%03d.ply' % num)
flame_model = load_model(flame_model_fname)

for i in range(num_samples):
    # Assign random shape parameters.
    # Beware changing expression parameters (i.e. betas[300:400]) or pose will result in meshes that cannot be used
    # as template for VOCA as VOCA requires "zero pose" templates in neutral expression
    flame_model.betas[:100] = np.random.randn(100) * 1.0

    # Output mesh in "zero pose"
    Mesh(flame_model.r, flame_model.f).write_ply(out_fname(i+1))
