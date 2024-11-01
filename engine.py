# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable

import torch
import torch.nn.functional as F

import numpy as np

import util.misc as misc
import util.lr_sched as lr_sched

from torch.autograd import Variable
from math import exp

from einops import rearrange, repeat

import trimesh

from PIL import Image

# from models import EDMLoss

def train_one_epoch(model: torch.nn.Module,
                    data_loader, optimizer: torch.optim.Optimizer,
                    criterion,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20
    
    accum_iter = args.accum_iter

    optimizer.zero_grad()

    # criterion = EDMLoss()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    
    print(data_loader)

    noise = None

    if isinstance(data_loader, dict):
        obj_file = data_loader['obj_file']
        batch_size = data_loader['batch_size']

        if obj_file is not None:
            if obj_file.endswith('.obj'):
                mesh = trimesh.load(obj_file)
                if data_loader['texture_path'] is not None:
                    img = Image.open(data_loader['texture_path'])
                    material = trimesh.visual.texture.SimpleMaterial(image=img)
                    assert mesh.visual.uv is not None
                    texture = trimesh.visual.TextureVisuals(mesh.visual.uv, image=img, material=material)
                    mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, visual=texture, process=False)

                    samples, _, colors = trimesh.sample.sample_surface(mesh,  2048*64*4*64, sample_color=True)
                    colors = colors[:, :3] # remove alpha
                    colors = (colors.astype(np.float32) / 255.0 - 0.5)  / np.sqrt(1/12) # [-1, 1]
                    samples = np.concatenate([samples, colors], axis=1)
                else:
                    samples, _ = trimesh.sample.sample_surface(mesh,  2048*64*4*64)
            else:
                samples = trimesh.load(obj_file).vertices

        else:
            if data_loader['primitive'] == 'sphere':
                n = torch.randn(2048*64*4*64, 3)
                n = torch.nn.functional.normalize(n, dim=1)
                samples = n / np.sqrt(1/3)
                samples = samples.numpy()
            elif data_loader['primitive'] == 'plane':
                samples = torch.rand(2048*64*4*64, 3) - 0.5
                samples[:, 2] = 0
                samples = (samples - 0) / np.sqrt(2/9*2*0.5**3)
                samples = samples.numpy()
            elif data_loader['primitive'] == 'gaussian':
                samples = np.random.randn(2048*64*4*64, 3).astype(np.float32)
            else:
                raise NotImplementedError

        if data_loader['noise_mesh'] is not None:
            noise, _ = trimesh.sample.sample_surface(trimesh.load(data_loader['noise_mesh']),  2048*64*4*64)
        else:
            noise = None

        # samples, _ = trimesh.sample.sample_surface(trimesh.load('shapes/Jellyfish_lamp_part_A__B_normalized.obj'),  2048*64*4*64)
        samples = samples.astype(np.float32)# - 0.12
        # data_loader = range(512)
        data_loader = range(data_loader['epoch_size'])

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # xyz = xyz.to(device, non_blocking=True)
        # xyz, _ = trimesh.sample.sample_surface(trimesh.load('test.obj'), 2048*64*2)

        if isinstance(batch, int):
            ind = np.random.default_rng().choice(samples.shape[0], batch_size, replace=True)
            xyz = samples[ind]
            xyz = torch.from_numpy(xyz).float().to(device, non_blocking=True)
        else:
            xyz = batch.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=False):
            if noise is not None:
                ind = np.random.default_rng().choice(noise.shape[0], batch_size, replace=True)
                init_noise = noise[ind]
                init_noise = torch.from_numpy(init_noise).float().to(device, non_blocking=True)
            else:
                init_noise = None
            loss = criterion(model, xyz, init_noise=init_noise)
            
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
