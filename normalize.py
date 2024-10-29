import trimesh

import math

import glob

import numpy as np

# model = trimesh.load('/home/zhanb0b/projects/surface_diffusion/shapes/wukong/wukong_SubTool1.stl', process=False)
model = trimesh.util.concatenate([trimesh.load(m, process=False) for m in glob.glob('shapes/wukong/*.stl')])
# print(model.vertices.max(axis=0), model.vertices.min(axis=0))

# print(model.bounding_box.centroid)

# print((model.vertices.max(axis=0) + model.vertices.min(axis=0)) / 2)

def normalize_meshes(mesh):
    mesh.vertices -= (mesh.vertices.max(axis=0) + mesh.vertices.min(axis=0)) / 2

    scale = (1 / np.abs(mesh.vertices).max()) * 0.99

    mesh.vertices *= scale

    points, _ = trimesh.sample.sample_surface(mesh, 10000000)

    mesh.vertices -= points.mean()
    mesh.vertices /= points.std()

    return mesh

model = normalize_meshes(model)

angle = math.pi / 2
direction = [1, 0, 0]
center = [0, 0, 0]

rot_matrix = trimesh.transformations.rotation_matrix(angle, direction, center)

model.apply_transform(rot_matrix)

model.export('shapes/wukong.obj')