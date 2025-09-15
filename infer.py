import argparse 
from pathlib import Path
import os

import torch

import trimesh

from models import EDMPrecond

torch.manual_seed(0)

import numpy as np
np.random.seed(0)

import random
random.seed(0)

# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True

parser = argparse.ArgumentParser('Inference', add_help=False)
parser.add_argument('--pth', required=True, type=str)
parser.add_argument('--texture', action='store_true')
parser.add_argument('--target', default='Gaussian', type=str)
parser.add_argument('--N', default=1000000, type=int)
parser.add_argument('--num-steps', default=64, type=int)
parser.add_argument('--noise_mesh', default=None, type=str)
parser.add_argument('--output', required=True, type=str)
parser.add_argument('--intermediate', action='store_true')
parser.add_argument('--depth', default=6, type=int)
parser.set_defaults(texture=False)
parser.set_defaults(intermediate=False)

args = parser.parse_args()

Path(args.output).mkdir(parents=True, exist_ok=True)

if args.texture:
    model = EDMPrecond(channels=6, depth=args.depth).cuda()
else:
    model = EDMPrecond(depth=args.depth).cuda()

model.load_state_dict(torch.load(args.pth, map_location='cpu', weights_only=False)['model'], strict=True)

if args.target == 'Gaussian':
    noise = torch.randn(args.N, 3).cuda()
elif args.target == 'Uniform':
    noise = (torch.rand(args.N, 3).cuda() - 0.5) / np.sqrt(1/12)
elif args.target == 'Sphere':
    n = torch.randn(args.N, 3).cuda()
    n = torch.nn.functional.normalize(n, dim=1)
    noise = n / np.sqrt(1/3)
elif args.target == 'Mesh':
    assert args.noise_mesh is not None
    noise, _ = trimesh.sample.sample_surface(trimesh.load(args.noise_mesh), args.N)
    noise = torch.from_numpy(noise).float().cuda()
else:
    raise NotImplementedError

if args.texture:
    color = (torch.rand(args.N, 3).cuda() - 0.5) / np.sqrt(1/12)
    noise = torch.cat([noise, color], dim=1)

sample, intermediate_steps = model.sample(batch_seeds=noise, num_steps=args.num_steps)

if args.texture:
    sample = sample.detach().cpu().numpy()
    vertices, colors = sample[:, :3], sample[:, 3:]
    colors = (colors * np.sqrt(1/12) + 0.5) * 255.0
    colors = np.concatenate([colors, np.ones_like(colors[:, 0:1]) * 255.0], axis=1).astype(np.uint8) # alpha channel
    trimesh.PointCloud(vertices, colors).export(os.path.join(args.output, 'sample.ply'))

    if args.intermediate:
        for i, s in enumerate(intermediate_steps):
            vertices, colors = s[:, :3], s[:, 3:]
            colors = (colors * np.sqrt(1/12) + 0.5) * 255.0
            colors = np.concatenate([colors, np.ones_like(colors[:, 0:1]) * 255.0], axis=1).astype(np.uint8) # alpha channel

            trimesh.PointCloud(vertices, colors).export(os.path.join(args.output, 'sample-{:03d}.ply'.format(i)))

else:
    trimesh.PointCloud(sample.detach().cpu().numpy()).export(os.path.join(args.output, 'sample.ply'))

    if args.intermediate:
        for i, s in enumerate(intermediate_steps):
            trimesh.PointCloud(s).export(os.path.join(args.output, 'sample-{:03d}.ply'.format(i)))
