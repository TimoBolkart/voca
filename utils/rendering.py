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
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl' # Uncommnet this line while running remotely
import cv2
import pyrender
import trimesh
import tempfile
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
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

def render_mesh_helper(mesh, t_center, rot=np.zeros(3), tex_img=None, v_colors=None, errors=None, error_unit='m', min_dist_in_mm=0.0, max_dist_in_mm=3.0, z_offset=0):
    camera_params = {'c': np.array([400, 400]),
                     'k': np.array([-0.19816071, 0.92822711, 0, 0, 0]),
                     'f': np.array([4754.97941935 / 2, 4754.97941935 / 2])}

    frustum = {'near': 0.01, 'far': 3.0, 'height': 800, 'width': 800}

    mesh_copy = Mesh(mesh.v, mesh.f)
    mesh_copy.v[:] = cv2.Rodrigues(rot)[0].dot((mesh_copy.v-t_center).T).T+t_center

    texture_rendering = tex_img is not None and hasattr(mesh, 'vt') and hasattr(mesh, 'ft')
    if texture_rendering:
        intensity = 0.5
        tex = pyrender.Texture(source=tex_img, source_channels='RGB') 
        material = pyrender.material.MetallicRoughnessMaterial(baseColorTexture=tex)

        # Workaround as pyrender requires number of vertices and uv coordinates to be the same
        temp_filename = '%s.obj' % next(tempfile._get_candidate_names())
        mesh.write_obj(temp_filename)
        tri_mesh = trimesh.load(temp_filename, process=False)
        try:
            os.remove(temp_filename)
        except:
            print('Failed deleting temporary file - %s' % temp_filename)
        render_mesh = pyrender.Mesh.from_trimesh(tri_mesh, material=material)
    elif errors is not None:
        intensity = 0.5
        unit_factor = get_unit_factor('mm')/get_unit_factor(error_unit)
        errors = unit_factor*errors

        norm = mpl.colors.Normalize(vmin=min_dist_in_mm, vmax=max_dist_in_mm)
        cmap = cm.get_cmap(name='jet')
        colormapper = cm.ScalarMappable(norm=norm, cmap=cmap)
        rgba_per_v = colormapper.to_rgba(errors)
        rgb_per_v = rgba_per_v[:, 0:3]
    elif v_colors is not None:
        intensity = 0.5
        rgb_per_v = v_colors
    else:
        intensity = 1.5
        rgb_per_v = None

    if not texture_rendering:
        tri_mesh = trimesh.Trimesh(vertices=mesh_copy.v, faces=mesh_copy.f, vertex_colors=rgb_per_v)
        render_mesh = pyrender.Mesh.from_trimesh(tri_mesh, smooth=True)

    scene = pyrender.Scene(ambient_light=[.2, .2, .2], bg_color=[255, 255, 255])
    camera = pyrender.IntrinsicsCamera(fx=camera_params['f'][0],
                                      fy=camera_params['f'][1],
                                      cx=camera_params['c'][0],
                                      cy=camera_params['c'][1],
                                      znear=frustum['near'],
                                      zfar=frustum['far'])

    scene.add(render_mesh, pose=np.eye(4))

    camera_pose = np.eye(4)
    camera_pose[:3,3] = np.array([0, 0, 1.0-z_offset])
    scene.add(camera, pose=[[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 1],
                            [0, 0, 0, 1]])

    angle = np.pi / 6.0
    pos = camera_pose[:3,3]
    light_color = np.array([1., 1., 1.])
    light = pyrender.PointLight(color=light_color, intensity=intensity)

    light_pose = np.eye(4)
    light_pose[:3,3] = pos
    scene.add(light, pose=light_pose.copy())

    light_pose[:3,3] = cv2.Rodrigues(np.array([angle, 0, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3,3] =  cv2.Rodrigues(np.array([-angle, 0, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3,3] = cv2.Rodrigues(np.array([0, -angle, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    light_pose[:3,3] = cv2.Rodrigues(np.array([0, angle, 0]))[0].dot(pos)
    scene.add(light, pose=light_pose.copy())

    flags = pyrender.RenderFlags.SKIP_CULL_FACES
    try:
        r = pyrender.OffscreenRenderer(viewport_width=frustum['width'], viewport_height=frustum['height'])
        color, _ = r.render(scene, flags=flags)
    except:
        print('pyrender: Failed rendering frame')
        color = np.zeros((frustum['height'], frustum['width'], 3), dtype='uint8')

    return color[..., ::-1]