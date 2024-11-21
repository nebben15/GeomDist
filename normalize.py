import argparse 

import trimesh

import math

import glob

import numpy as np

parser = argparse.ArgumentParser('Inference', add_help=False)
parser.add_argument('--path', required=True, type=str)
parser.add_argument('--output', required=True, type=str)

args = parser.parse_args()

model = trimesh.load(args.path, process=False)

def normalize_meshes(mesh):
    mesh.vertices -= (mesh.vertices.max(axis=0) + mesh.vertices.min(axis=0)) / 2

    scale = (1 / np.abs(mesh.vertices).max()) * 0.99

    mesh.vertices *= scale

    points, _ = trimesh.sample.sample_surface(mesh, 10000000)

    mesh.vertices -= points.mean()
    mesh.vertices /= points.std()

    return mesh

model = normalize_meshes(model)

# angle = math.pi / 2
# direction = [1, 0, 0]
# center = [0, 0, 0]

# rot_matrix = trimesh.transformations.rotation_matrix(angle, direction, center)

# model.apply_transform(rot_matrix)

model.export(args.output)