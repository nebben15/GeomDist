import trimesh

import numpy as np
import os

import torch
from torch.utils import data

class Points(data.Dataset):
    def __init__(self, ply_path):
        # points = trimesh.load(ply_path).vertices
        # self.points = np.array(points)
        if os.path.exists('test.npy'):
            points = np.load('test.npy')
        else:
            points, _ = trimesh.sample.sample_surface(trimesh.load('test.obj'), 50000000*5)
            np.save('test.npy', points)
        self.points = torch.from_numpy(points) - 0.12

    def __len__(self):
        return self.points.shape[0]# * 16

    def __getitem__(self, idx):
        # idx = idx % self.points.shape[0]
        return self.points[idx]