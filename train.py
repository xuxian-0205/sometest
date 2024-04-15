# coding=utf-8
from __future__ import print_function, division
import sys
sys.path.append('core')
import time
import argparse
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import argparse, configparser
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

from torch.utils.data import DataLoader
from core.ttc_gogo import TTC
from core.utils.loss import compute_supervision_coarse, compute_coarse_loss, backwarp
import dc_flow_eval as evaluate
import core.dataset_exp_orin as datasets

from torch.utils.tensorboard import SummaryWriter

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass


# exclude extremly large displacements
MAX_FLOW = 400
SUM_FREQ = 20
VAL_FREQ = 5000
def mid2ttc(mid):
    ttc = 0.1/(1-mid)
    return ttc
def all_loss(loss,boxnum,bl,loss_dc,loss_flow):

    metrics = {
        '1loss': loss.item() ,
        '2loss': loss_dc.item(),
        '3loss': loss_flow.item(),
        '4all': (loss_dc.item()+loss.item()+loss_flow.item()),

        '5num': boxnum.sum().item(),
    }
    return loss_dc+loss+loss_flow,metrics



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler
    

class Logger:
    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None
        self.last_time = time.time()
    def _print_training_status(self):
        now_time = time.time()
        metrics_data = [self.running_loss[k]/SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        time_str = ("time = %.2f, "%(now_time-self.last_time))
        data = open("/home/ans/records.txt", 'a')
        # print the training status
        print(training_str + metrics_str +time_str)
        data.write(training_str + metrics_str+"\n")
        data.close()
        self.last_time = now_time
        if self.writer is None:
            self.writer = SummaryWriter()

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % SUM_FREQ == SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter()

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()


def train(args):

    model = nn.DataParallel(TTC(args), device_ids=args.gpus)
    print("Parameter Count: %d" % count_parameters(model))





    pretrained_dict = torch.load(args.restore_ckpt)
    old_list = {}

    for k, v in pretrained_dict.items():
        if k.find('fnet')>0 or k.find('cnet')>0 or k.find('weightt')>0:
            old_list.update({k:v}) 
    model.load_state_dict(old_list,strict=False)


    model.cuda()
    model.train()
    '''
    if args.stage != 'chairs':
        model.module.freeze_bn()
    '''
    train_loader = datasets.fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)

    total_steps = 0
    scaler = GradScaler(enabled=args.mixed_precision)
    logger = Logger(model, scheduler)

    VAL_FREQ = 2000
    add_noise = True

    should_keep_training = True
    while should_keep_training:

        for i_batch, data_blob in enumerate(train_loader):
            optimizer.zero_grad()
            image1, image2, flow, dc_change, valid ,bl,boxnum= [x.cuda() for x in data_blob]

            if args.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
                image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)

            #if sum(boxnum[0:5])==0 or sum(boxnum[5:10])==0:
                #continue

            loss_ob,boxnum,loss_dc,loss_flow = model(image1, image2,flow,dc_change,bl,boxnum,iters=12)
            #ttc_gogo.py TTC
            loss1, metrics = all_loss(loss_ob.mean(),boxnum,bl,loss_dc.mean(),loss_flow.mean())
            loss = loss1
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            logger.push(metrics)
            model.eval()
            if total_steps % VAL_FREQ == VAL_FREQ - 1:
                evaluate.validate_kitti_test(model.module)
                PATH = 'checkpoints/%d_%s.pth' % (total_steps+1, args.name)
                torch.save(model.state_dict(), PATH)
            model.train()
            total_steps += 1

            if total_steps > args.num_steps:
                should_keep_training = False
                break

    logger.close()
    PATH = 'checkpoints/%s.pth' % args.name
    torch.save(model.state_dict(), PATH)

    return PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training") 
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--validation', type=str, nargs='+')

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--gpus', type=int, nargs='+')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--add_noise', action='store_true')
    parser.add_argument('--num_heads', default=1, type=int,
                        help='number of heads in attention and aggregation')
    parser.add_argument('--position_only', default=False, action='store_true',
                        help='only use position-wise attention')
    parser.add_argument('--position_and_content', default=False, action='store_true',
                        help='use position and content-wise attention')
    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    train(args)