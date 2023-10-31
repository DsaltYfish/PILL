import os
import argparse
import datetime
import json
import time
import copy
import random
import numpy as np
from pathlib import Path
import ruamel.yaml as yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.cuda.amp import autocast, GradScaler
import timm.optim.optim_factory as optim_factory

from model.modeling_llama import Llama
from engine import train_one_epoch, val_one_epoch
import utils
import utils.misc as misc

from utils import lr_sched
from data import CCPretrainDataSet, data_collate
# from train import get_args_parser


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=3, type=int)
    parser.add_argument('--accum_iter', default=8, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--llama_model_path', default='/data/zfy/llama/', type=str,
                        help='path of llama model')

    # parser.add_argument('--llama_model_path', default='/data/zfy/llama1/weight/pyllama_data/', type=str,
    #                     help='path of llama model')                        

    parser.add_argument('--llm_model', default='llama-2-7b-chat', type=str, metavar='MODEL',
                        help='Name of llm model to train')

    # parser.add_argument('--llm_model', default='7B', type=str, metavar='MODEL',
    #                     help='Name of llm model to train')                        

    parser.add_argument('--use_vicuna', action='store_true', help='use vicuna weights')

    parser.add_argument('--task', default='', type=str, metavar='MODEL',
                        help='What task wants to train')
    
    parser.add_argument('--max_length', type=int, default=128, metavar='LENGTH',
                        help='the maximum sequence length')

    parser.add_argument('--max_seq_len', type=int, default=512, metavar='LENGTH',
                        help='the maximum sequence length')
    
    parser.add_argument('--evaluate', type=bool, default=False, metavar='LENGTH',
                        help='the maximum sequence length')


    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.02,
                        help='weight decay (default: 0.02)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--grad_checkpoint', type=bool, default=False, 
                        help='saving memory costs via gradient_checkpointing')
    parser.add_argument('--warmup_epochs', type=int, default=0.03, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/instruction_dataset/', type=str,
                        help='dataset path')
                        
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--resume', default=False,
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
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

    #datasets
    parser.add_argument('--data_root', type=str, default='/data/zfy/dataset/llava')

    parser.add_argument('--do_pretrain', type=bool, default=True, 
                        help='pre-train on large scale vl instruction')

    return parser



def main(args):
    misc.init_distributed_mode(args)
    
    device = torch.device(args.device)

    # torch.autograd.set_detect_anomaly(True)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset #### 

    dataset_train = CCPretrainDataSet(args, args.llama_model_path, args.max_length)


    train_loader = torch.utils.data.DataLoader(
        dataset_train, shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        collate_fn=data_collate,
        drop_last=True,
    )


    if args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    #### Model #### 
    print("Creating model")
    model = Llama(args)

    model = model.to(device)

    loss_scaler = misc.NativeScalerWithGradNormCount()

    # print(model)
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module   

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(params=param_groups, lr=args.lr, betas=(0.9, 0.95))
    # optimizer = optim.Adafactor(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # print(optimizer)
    
    if args.resume is not None:
        misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print("Start training")
    start_time = time.time()

    # best = 100
    # best_epoch = 0

    for epoch in range(args.start_epoch, args.epochs):        
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            
        # misc.cosine_lr_schedule(optimizer, epoch, args.epochs, args.lr, args.min_lr)
        
        train_stats = train_one_epoch(
            model, train_loader,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
    
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch,
                    #  'best_epoch': best_epoch,
        } 
        if args.output_dir:
            # if val_stats['closs'] < best:
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

                # best = val_stats['closs']
                # best_epoch = epoch

                    
        with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
            f.write(json.dumps(log_stats) + "\n")

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
        # dist.barrier()
        torch.cuda.empty_cache()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 

if __name__ == '__main__':

    args = get_args_parser()
    # config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # main(args)
        
    # yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args)
