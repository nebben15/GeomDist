import argparse 

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
parser.add_argument('--pth', default='output/lamp_cube/checkpoint-0.pth', type=str)
parser.add_argument('--texture', action='store_true')
parser.add_argument('--N', default=1000000, type=int)
parser.add_argument('--num-steps', default=64, type=int)
parser.add_argument('--noise_mesh', default=None, type=str)
parser.add_argument('--data_path', default='shapes/Jellyfish_lamp_part_A__B_normalized.obj', type=str)
parser.set_defaults(texture=False)

args = parser.parse_args()

if args.texture:
    model = EDMPrecond(channels=6).cuda()
else:
    model = EDMPrecond().cuda()

mesh = trimesh.load(args.data_path)
samples, _ = trimesh.sample.sample_surface(mesh,  args.N)
samples = samples.astype(np.float32)
samples = torch.from_numpy(samples).float().cuda()

model.load_state_dict(torch.load(args.pth, map_location='cpu')['model'], strict=True)

sample, intermediate_steps = model.inverse(samples=samples, num_steps=args.num_steps)
print(sample.mean(), sample.std())

# sample, intermediate_steps = model.sample(batch_seeds=sample, num_steps=args.num_steps)
# print(sample.mean(), sample.std())
# print((samples - sample).pow(2).sum(dim=1).mean().sqrt())

# sample.export('ouput_a.obj')
if args.texture:
    sample = sample.detach().cpu().numpy()
    vertices, colors = sample[:, :3], sample[:, 3:]
    colors = (colors * np.sqrt(1/12) + 0.5) * 255.0
    colors = np.concatenate([colors, np.ones_like(colors[:, 0:1]) * 255.0], axis=1).astype(np.uint8) # alpha channel
    trimesh.PointCloud(vertices, colors).export('sample.ply')

    for i, s in enumerate(intermediate_steps):
        vertices, colors = s[:, :3], s[:, 3:]
        colors = (colors * np.sqrt(1/12) + 0.5) * 255.0
        colors = np.concatenate([colors, np.ones_like(colors[:, 0:1]) * 255.0], axis=1).astype(np.uint8) # alpha channel

        trimesh.PointCloud(vertices, colors).export('sample-{:03d}.ply'.format(i))

else:
    trimesh.PointCloud(sample.detach().cpu().numpy()).export('sample.ply')

    for i, s in enumerate(intermediate_steps):
        trimesh.PointCloud(s).export('sample-{:03d}.ply'.format(i))

# noise = torch.randn(1000000, 3).cuda()
# for sigma in range(1, 33):

#     id = noise.pow(2).sum(dim=1).sqrt() < sigma / 10
#     sample = model.sample(batch_seeds=noise[id], num_steps=64)
#     # (a.pow(2).sum(dim=1)<15).sum()
#     # print(sample.shape)

#     # sample.export('ouput_a.obj')
#     trimesh.PointCloud(noise[id].detach().cpu().numpy()).export('ci-noise-{:02d}.ply'.format(sigma))
#     trimesh.PointCloud(sample.detach().cpu().numpy()).export('ci-surface-{:02d}.ply'.format(sigma))