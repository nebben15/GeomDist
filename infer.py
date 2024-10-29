import argparse 

import torch

import trimesh

from models import EDMPrecond

torch.manual_seed(0)

import numpy as np
np.random.seed(0)

import random
random.seed(0)

parser = argparse.ArgumentParser('Inference', add_help=False)
parser.add_argument('--pth', default='output/lamp_cube/checkpoint-0.pth', type=str)
parser.add_argument('--target', default='Gaussian', type=str)
parser.add_argument('--N', default=1000000, type=int)

args = parser.parse_args()

model = EDMPrecond().cuda()
model.load_state_dict(torch.load(args.pth, map_location='cpu')['model'], strict=True)
# noise = torch.randn(1000000, 3).cuda()
if args.target == 'Gaussian':
    noise = torch.randn(args.N, 3).cuda()
elif args.target == 'Uniform':
    noise = (torch.rand(args.N, 3).cuda() - 0.5) / np.sqrt(1/12)
else:
    raise NotImplementedError

sample = model.sample(batch_seeds=noise, num_steps=64)
# print(sample.shape)

# sample.export('ouput_a.obj')
trimesh.PointCloud(sample.detach().cpu().numpy()).export('sample.ply')

# noise = torch.randn(1000000, 3).cuda()
# for sigma in range(1, 33):

#     id = noise.pow(2).sum(dim=1).sqrt() < sigma / 10
#     sample = model.sample(batch_seeds=noise[id], num_steps=64)
#     # (a.pow(2).sum(dim=1)<15).sum()
#     # print(sample.shape)

#     # sample.export('ouput_a.obj')
#     trimesh.PointCloud(noise[id].detach().cpu().numpy()).export('ci-noise-{:02d}.ply'.format(sigma))
#     trimesh.PointCloud(sample.detach().cpu().numpy()).export('ci-surface-{:02d}.ply'.format(sigma))