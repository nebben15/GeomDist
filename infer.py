import argparse 
from pathlib import Path
import os

import torch

import open3d as o3d
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
parser.add_argument('--feature-dim', default=0, type=int, help='Dimensionality of additional features')
parser.add_argument('--max_categorical_feature', default=1, type=int, help="Maximum value of categorical features. Used for scaling.")
parser.set_defaults(texture=False)
parser.set_defaults(intermediate=False)

args = parser.parse_args()

Path(args.output).mkdir(parents=True, exist_ok=True)

if args.texture:
    model = EDMPrecond(channels=6, depth=args.depth).cuda()
    mode = 'geometry+texture'
    total_dim = 6
elif args.feature_dim > 0:
    model = EDMPrecond(channels=3 + args.feature_dim, depth=args.depth).cuda()
    mode = 'geometry+feature'
    total_dim = 3 + args.feature_dim
else:
    model = EDMPrecond(depth=args.depth).cuda()
    mode = 'geometry'
    total_dim = 3

model.load_state_dict(torch.load(args.pth, map_location='cpu', weights_only=False)['model'], strict=True)

if args.target == 'Gaussian':
    noise = torch.randn(args.N, total_dim).cuda()
elif args.target == 'Uniform':
    noise = (torch.rand(args.N, total_dim).cuda() - 0.5) / np.sqrt(1/12)
elif args.target == 'Sphere':
    n = torch.randn(args.N, total_dim).cuda()
    n = torch.nn.functional.normalize(n, dim=1)
    noise = n / np.sqrt(1/3)
elif args.target == 'Mesh':
    assert args.noise_mesh is not None
    noise, _ = trimesh.sample.sample_surface(trimesh.load(args.noise_mesh), args.N)
    noise = torch.from_numpy(noise).float().cuda()
else:
    raise NotImplementedError

# if args.feature_dim > 0:
#     additional_features = torch.randn(args.N, args.feature_dim).cuda()
#     noise = torch.cat([noise, additional_features], dim=1)

# if args.texture:
#     color = (torch.rand(args.N, 3).cuda() - 0.5) / np.sqrt(1/12)
#     noise = torch.cat([noise, color], dim=1)

sample, intermediate_steps = model.sample(batch_seeds=noise, num_steps=args.num_steps)

if mode == 'geometry+texture':
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

elif mode == 'geometry+feature':
    sample = sample.detach().cpu().numpy()
    vertices = sample[:, :3].astype(np.float32)
    features = sample[:, 3:].astype(np.float32)
    features = (features * np.sqrt(1/12) + 0.5) * args.max_categorical_feature
    pcd_t = o3d.t.geometry.PointCloud()
    # Positions must be set this way
    pcd_t.point.positions = o3d.core.Tensor(vertices, dtype=o3d.core.Dtype.Float32)
    # Set multi-dimensional feature as a single attribute
    pcd_t.point['features'] = o3d.core.Tensor(features, dtype=o3d.core.Dtype.Float32)
    # # Set custom attributes
    # for i in range(features.shape[1]):
    #     attr_name = f'feat_dim_{i}'
    #     col = features[:, i].reshape(-1, 1)
    #     # Need to wrap as Open3D tensor
    #     tcol = o3d.core.Tensor(col, dtype=o3d.core.Dtype.Float32)
    #     pcd_t.point[attr_name] = tcol
    # Write to PLY, ensuring tensor attributes are included
    o3d.t.io.write_point_cloud(os.path.join(args.output, 'sample.ply'), pcd_t, write_ascii=False)

    if args.intermediate:
        for i, s in enumerate(intermediate_steps):
                sample = s.detach().cpu().numpy()
                vertices = sample[:, :3].astype(np.float32)
                features = sample[:, 3:].astype(np.float32)
                pcd_t = o3d.t.geometry.PointCloud()
                # Positions must be set this way
                pcd_t.point.positions = o3d.core.Tensor(vertices, dtype=o3d.core.Dtype.Float32)
                # Set custom attributes
                for i in range(features.shape[1]):
                    attr_name = f'feat_dim_{i}'
                    col = features[:, i].reshape(-1, 1)
                    # Need to wrap as Open3D tensor
                    tcol = o3d.core.Tensor(col, dtype=o3d.core.Dtype.Float32)
                    pcd_t.point[attr_name] = tcol
                # Write to PLY, ensuring tensor attributes are included
                o3d.t.io.write_point_cloud(os.path.join(args.output, f'sample-{i:03d}.ply'), pcd_t, write_ascii=False)
else:
    trimesh.PointCloud(sample.detach().cpu().numpy()).export(os.path.join(args.output, 'sample.ply'))

    if args.intermediate:
        for i, s in enumerate(intermediate_steps):
            trimesh.PointCloud(s).export(os.path.join(args.output, 'sample-{:03d}.ply'.format(i)))
