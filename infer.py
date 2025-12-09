import argparse 
from pathlib import Path
import os
import re

import torch

import open3d as o3d
import trimesh

from models import EDMPrecond
from util import features as fe

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
parser.add_argument('--feature-file', default=None, type=str, help='Path to feature TXT with header (FEATURES_COUNT/DIM/MAX_ABS) to set feature dim and de-normalize outputs')
parser.set_defaults(texture=False)
parser.set_defaults(intermediate=False)

args = parser.parse_args()

Path(args.output).mkdir(parents=True, exist_ok=True)

feature_dim = 0
feature_max_vals = None

# Derive epoch number from checkpoint filename (e.g., checkpoint-5.pth -> 5)
ckpt_name = os.path.basename(args.pth)
match = re.search(r'checkpoint-(\d+)', ckpt_name)
epoch_tag = match.group(1) if match else 'unknown'
base_name = f"sample_e{epoch_tag}_n{args.N}"

if args.texture:
    model = EDMPrecond(channels=6, depth=args.depth).cuda()
    mode = 'geometry+texture'
    total_dim = 6
elif args.feature_file is not None:
    # Load feature header to infer dimensionality and per-dim maxima
    _, feature_max_vals, feature_dim = fe.load_features_with_metadata(args.feature_file)
    model = EDMPrecond(channels=3 + feature_dim, depth=args.depth).cuda()
    mode = 'geometry+feature'
    total_dim = 3 + feature_dim
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

sample, intermediate_steps = model.sample(batch_seeds=noise, num_steps=args.num_steps)

if mode == 'geometry+texture':
    sample = sample.detach().cpu().numpy()
    vertices, colors = sample[:, :3], sample[:, 3:]
    colors = (colors * np.sqrt(1/12) + 0.5) * 255.0
    colors = np.concatenate([colors, np.ones_like(colors[:, 0:1]) * 255.0], axis=1).astype(np.uint8) # alpha channel
    trimesh.PointCloud(vertices, colors).export(os.path.join(args.output, f'{base_name}.ply'))

    if args.intermediate:
        for step_idx, s in enumerate(intermediate_steps):
            vertices, colors = s[:, :3], s[:, 3:]
            colors = (colors * np.sqrt(1/12) + 0.5) * 255.0
            colors = np.concatenate([colors, np.ones_like(colors[:, 0:1]) * 255.0], axis=1).astype(np.uint8) # alpha channel

            trimesh.PointCloud(vertices, colors).export(os.path.join(args.output, f'{base_name}_i{step_idx:03d}.ply'))

elif mode == 'geometry+feature':
    sample = sample.detach().cpu().numpy()
    vertices = sample[:, :3].astype(np.float32)
    features = sample[:, 3:].astype(np.float32)
    # De-normalize using per-dimension maxima from feature header
    if feature_max_vals is None:
        raise ValueError('--feature-file is required in geometry+feature mode to determine scaling.')
    # Default: treat dims as unsigned unless a signed_mask is provided later
    features = fe.renormalize_features(features, feature_max_vals)
    pcd_t = o3d.t.geometry.PointCloud()
    # Positions must be set this way
    pcd_t.point.positions = o3d.core.Tensor(vertices, dtype=o3d.core.Dtype.Float32)
    # Open3D PLY IO only supports (N,1) attributes (besides positions/normals/colors),
    # so split multi-dim features into separate scalar attributes.
    for j in range(features.shape[1]):
        attr_name = f'feat_dim_{j}'
        col = features[:, j].reshape(-1, 1)
        tcol = o3d.core.Tensor(col, dtype=o3d.core.Dtype.Float32)
        pcd_t.point[attr_name] = tcol
    # Write to PLY, ensuring tensor attributes are included
    o3d.t.io.write_point_cloud(os.path.join(args.output, f'{base_name}.ply'), pcd_t, write_ascii=False)

    if args.intermediate:
        for step_idx, s in enumerate(intermediate_steps):
            sample_i = s.detach().cpu().numpy()
            vertices_i = sample_i[:, :3].astype(np.float32)
            features_i = sample_i[:, 3:].astype(np.float32)
            # Apply same de-normalization
            features_i = fe.renormalize_features(features_i, feature_max_vals)
            pcd_t = o3d.t.geometry.PointCloud()
            # Positions must be set this way
            pcd_t.point.positions = o3d.core.Tensor(vertices_i, dtype=o3d.core.Dtype.Float32)
            # Set custom attributes
            for j in range(features_i.shape[1]):
                attr_name = f'feat_dim_{j}'
                col = features_i[:, j].reshape(-1, 1)
                # Need to wrap as Open3D tensor
                tcol = o3d.core.Tensor(col, dtype=o3d.core.Dtype.Float32)
                pcd_t.point[attr_name] = tcol
            # Write to PLY, ensuring tensor attributes are included
            o3d.t.io.write_point_cloud(os.path.join(args.output, f'{base_name}_i{step_idx:03d}.ply'), pcd_t, write_ascii=False)
else:
    trimesh.PointCloud(sample.detach().cpu().numpy()).export(os.path.join(args.output, f'{base_name}.ply'))

    if args.intermediate:
        for step_idx, s in enumerate(intermediate_steps):
            trimesh.PointCloud(s).export(os.path.join(args.output, f'{base_name}_i{step_idx:03d}.ply'))
