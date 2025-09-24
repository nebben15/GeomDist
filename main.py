import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

torch.set_num_threads(10)
import util.lr_decay as lrd
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models as models
from models import EDMLoss

from engine import train_one_epoch

from points import Points


def get_args_parser():
    parser = argparse.ArgumentParser('Train', add_help=False)
    parser.add_argument('--batch_size', default=2048*64*2, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    

    # Model parameters
    parser.add_argument('--model', default='EDMPrecond', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--depth', default=6, type=int, metavar='MODEL')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=5e-7, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=5e-7, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=1, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--target', default='Gaussian', type=str, )
    parser.add_argument('--data_path', default='shapes/Jellyfish_lamp_part_A__B_normalized.obj', type=str,
                        help='dataset path')

    parser.add_argument('--texture_path', default=None, type=str,
                        help='dataset path')

    parser.add_argument('--feature_path', default=None, type=str, help='Path to the feature file (optional)')

    parser.add_argument('--feature_interpolation', default='barycentric', type=str, help='How the extra vertex features are interpolated for sampled points on the surface (nearest-neighbor, barycentric)')

    parser.add_argument('--noise_mesh', default=None, type=str,
                        help='dataset path')
     
    parser.add_argument('--output_dir', default='./output/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output/',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=32, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser

def get_feature_dim(path):
    """
    Get the feature dimension from a .txt file.

    Args:
        path (str): Path to the .txt file.

    Returns:
        int: The feature dimension (number of columns).

    Raises:
        ValueError: If the file is not well-formed (rows have inconsistent lengths).
    """
    with open(path, 'r') as file:
        lines = file.readlines()

    # Ensure the file is not empty
    if not lines:
        raise ValueError(f"The file {path} is empty.")

    # Get the feature dimension from the first row
    first_row = lines[1].strip().split()
    feature_dim = len(first_row)

    # Check that all rows have the same number of columns
    for i, line in enumerate(lines):
        if i != 0 and len(line.strip().split()) != feature_dim:
            raise ValueError(f"Inconsistent feature dimensions in file {path} at line {i + 1}.")

    return feature_dim

def main(args):

    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    cudnn.deterministic=True

    # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
    # in PyTorch 1.12 and later.
    torch.backends.cuda.matmul.allow_tf32 = True

    # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
    torch.backends.cudnn.allow_tf32 = True

    if True:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()


    neural_rendering_resolution = 128
    if args.data_path.endswith('.obj') or args.data_path.endswith('.ply'):
        if args.feature_path is not None:
            mode = 'geometry+feature'
        elif args.texture_path is not None:
            mode = 'geometry+texture'
        else:
            mode = 'geometry'

        data_loader_train = {
            'obj_file': args.data_path,
            'batch_size': args.batch_size,
            'epoch_size': 512,
            'texture_path': args.texture_path if mode == 'geometry+texture' else None,
            'feature_path': args.feature_path if mode == 'geometry+feature' else None,
            'mode': mode,
        }
        if args.noise_mesh is not None:
            data_loader_train['noise_mesh'] = args.noise_mesh
        else:
            data_loader_train['noise_mesh'] = None
    elif 'sphere' in args.data_path or 'plane' in args.data_path or 'volume' in args.data_path:
        data_loader_train = {
            'obj_file': None,
            'primitive': args.data_path,
            'batch_size': args.batch_size,
            'epoch_size': 512,
            'texture_path': args.texture_path,
        }
        if args.noise_mesh is not None:
            data_loader_train['noise_mesh'] = args.noise_mesh
        else:
            data_loader_train['noise_mesh'] = None
    else:
        raise NotImplementedError
    print(data_loader_train)

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None


    criterion = EDMLoss(dist=args.target)
    channels = 3
    if mode == "geometry+feature":
        channels += get_feature_dim(path=args.feature_path)
    elif mode == "geometry+texture":
        channels = 6
    model = models.__dict__[args.model](channels=channels, depth=args.depth)
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 128

    print("base lr: %.2e" % (args.lr * 128 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=args.lr)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_iou = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        # if args.distributed and args.data_path.endswith('.ply'):
        #     data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, criterion, device, epoch, loss_scaler,
            args.clip_grad,
            log_writer=log_writer,
            args=args,
            feature_interpolation=args.feature_interpolation
        )
        if args.output_dir and (epoch % 5 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        if epoch % 1 == 0 or epoch + 1 == args.epochs:

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            # **{f'test_{k}': v for k, v in test_stats.items()},
                            'epoch': epoch,
                            'n_parameters': n_parameters}
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            'epoch': epoch,
                            'n_parameters': n_parameters}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")


            
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))



if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
