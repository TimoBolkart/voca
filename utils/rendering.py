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

from __future__ import division

import cv2
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
from opendr.camera import ProjectPoints
from opendr.lighting import LambertianPointLight
from opendr.renderer import ColoredRenderer
from psbody.mesh import Mesh

def get_unit_factor(unit):
    if unit == 'mm':
        return 1000.0
    elif unit == 'cm':
        return 100.0
    elif unit == 'm':
        return 1.0
    else:
        raise ValueError('Unit not supported')

def render_mesh_helper(mesh, t_center, rot=np.zeros(3), v_colors=None, errors=None, error_unit='m', min_dist_in_mm=0.0, max_dist_in_mm=3.0, z_offset=0):
    render_mesh = Mesh(mesh.v, mesh.f)
    render_mesh.set_vertex_colors(0.7 * np.ones_like(render_mesh.v))
    if v_colors is not None:
        if v_colors.shape[0] == render_mesh.v.shape[0]:
            render_mesh.set_vertex_colors(v_colors)

    camera_params = {'c': np.array([400, 400]),
                     'k': np.array([-0.19816071, 0.92822711, 0, 0, 0]),
                     'f': np.array([4754.97941935 / 2, 4754.97941935 / 2])}

    render_mesh.v[:] = cv2.Rodrigues(rot)[0].dot((mesh.v-t_center).T).T+t_center

    rt = np.array([np.pi, 0, 0])
    t = np.array([0, 0, 1.0-z_offset])

    rn = ColoredRenderer()
    rn.camera = ProjectPoints(rt=rt, t=t + t_center, f=camera_params['f'], k=camera_params['k'],
                              c=camera_params['c'])
    rn.frustum = {'near': 0.01, 'far': 3.0, 'height': 800, 'width': 800}
    rn.set(v=render_mesh.v, f=render_mesh.f, vc=render_mesh.vc, bgcolor=[0.0, 0.0, 0.0])

    angle = np.pi / 4
    pos = rn.camera.t
    albedo = rn.vc
    pos_rot = cv2.Rodrigues(np.array([angle, 0, 0]))[0].dot(pos)

    if errors is not None:
        factor = 0.1
        unit_factor = get_unit_factor('mm')/get_unit_factor(error_unit)
        errors = unit_factor*errors

        norm = mpl.colors.Normalize(vmin=min_dist_in_mm, vmax=max_dist_in_mm)
        cmap = cm.get_cmap(name='jet')
        colormapper = cm.ScalarMappable(norm=norm, cmap=cmap)
        rgba_per_v = colormapper.to_rgba(errors)
        rgb_per_v = rgba_per_v[:, 0:3]
        rn.vc = rgb_per_v
    else:
        factor = 0.5
        rn.vc = np.zeros_like(rn.vc)

    light_color = np.array([1., 1., 1.])
    rn.vc += factor * LambertianPointLight(f=rn.f, v=rn.v, num_verts=len(rn.v), light_pos=pos_rot,
                                           vc=albedo,
                                           light_color=light_color)

    pos_rot = cv2.Rodrigues(np.array([-angle, 0, 0]))[0].dot(pos)
    rn.vc += factor * LambertianPointLight(f=rn.f, v=rn.v, num_verts=len(rn.v), light_pos=pos_rot,
                                           vc=albedo,
                                           light_color=light_color)

    pos_rot = cv2.Rodrigues(np.array([0, angle, 0]))[0].dot(pos)
    rn.vc += factor * LambertianPointLight(f=rn.f, v=rn.v, num_verts=len(rn.v), light_pos=pos_rot,
                                           vc=albedo,
                                           light_color=light_color)

    pos_rot = cv2.Rodrigues(np.array([0, -angle, 0]))[0].dot(pos)
    rn.vc += factor * LambertianPointLight(f=rn.f, v=rn.v, num_verts=len(rn.v), light_pos=pos_rot,
                                           vc=albedo,
                                           light_color=light_color)
    return (255.0 * rn.r[..., ::-1]).astype('uint8')

