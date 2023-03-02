import os
import numpy as np
import gzip
import json
import datetime
import pdb
from datetime import datetime as dt
import matplotlib.pyplot as plt
import pickle
import sys
import logging
from logging import *
import shutil
import math
from pathlib import Path
from typing import List, Dict, Tuple
from collections import OrderedDict
import random
from tqdm import tqdm
import operator
import heapq
    
import torch
from torch.optim.lr_scheduler import LambdaLR
import torch.distributed as dist


_FMT = "[%(asctime)s] %(levelname)s: %(message)s"
_DATEFMT = "%m/%d/%Y %H:%M:%S"

logging.basicConfig(
    level=logging.INFO, format=_FMT, datefmt=_DATEFMT, stream=sys.stdout
)


def copy_source(file, output_dir):
    import shutil
    shutil.copytree(file, os.path.join(output_dir, os.path.basename(file)))


def init_distributed():

    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    dist_url = "env://" # default
    # dist_url = "tcp://127.0.0.1:52111"

    # only works with torch.distributed.launch // torch.run
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])

    dist.init_process_group(
            backend="nccl",
            init_method=dist_url,
            timeout=datetime.timedelta(seconds=7200),
            world_size=world_size,
            rank=rank)

    # this will make all .cuda() calls work properly
    torch.cuda.set_device(local_rank)
    # synchronizes all the threads to reach this point before moving on
    dist.barrier()
    setup_for_distributed(rank == 0)

    
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
    
    
def global_meters_all_avg(args, *meters):
    """meters: scalar values of loss/accuracy calculated in each rank
    https://discuss.pytorch.org/t/right-ways-to-serialize-and-load-ddp-model-checkpoints/122719/2
    """
    tensors = [torch.tensor(meter, device=args.device, dtype=torch.float32) for meter in meters]
    
    for tensor in tensors:
        # each item of `tensors` is all-reduced starting from index 0 (in-place)
        dist.all_reduce(tensor)
        
    res = [(tensor / args.world_size).item() for tensor in tensors]
    if len(res) == 1:
        return res[0]
    else:
        return res


def need_logging(args):
    if hasattr(args, 'use_ddp'): 
        if not args.use_ddp or ((not args.ddp_log_each_rank and is_main_process()) or args.ddp_log_each_rank):  
            # need to log when
            # not using ddp
            # or using ddp under:
            # either (1) need to log on each rank 
            # or (2) no need to log on each rank but the current tank is the main process
            return True
        else:
            return False
    else:
        return True
    
    
def minmax_normalize(nums, min_val=None, max_val=None):
    if len(nums) > 0:
        if not min_val:
            min_val = min(nums)
        if not max_val:
            max_val = max(nums)
        for i in range(len(nums)):
            nums[i] = (nums[i] - min_val) / (max_val - min_val)
    return nums


def get_topK_largest_values_in_list(input_list, topk):
    if len(input_list) > 0:
        return heapq.nlargest(topk, input_list)
    else:
        return []


def get_indices_of_topK_largest_values_in_list(input_list, topk):
    if len(input_list) > 0:
        return list(zip(*heapq.nlargest(topk, enumerate(input_list), key=operator.itemgetter(1))))[0]
    else:
        return []


def fileHandler(path, format, datefmt, mode="w"):
    handler = logging.FileHandler(path, mode=mode)
    formatter = logging.Formatter(format, datefmt=datefmt)
    handler.setFormatter(formatter)
    return handler

  
def getLogger(
    name=None,
    path=None,
    level=logging.INFO,
    format=_FMT,
    datefmt=_DATEFMT,
):

    logger = logging.getLogger(name)
    logger.setLevel(level)
    if path is not None:
        from pathlib import Path
        path = str(Path(path).resolve())
        if not any(map(lambda hdr: hasattr(hdr, 'baseName') and hdr.baseFilename == path, logger.handlers)):
            handler = fileHandler(path, format, datefmt)
            logger.addHandler(handler)

    return logger


def numpy_topk_indices(array, k):
    idx = np.argpartition(array, -k)[-k:]  # Indices not sorted
    return idx[np.argsort(array[idx])][::-1]  # Indices sorted by value from largest to smallest
    

def save_checkpoint(state, is_best, dir='checkpoints/', name='checkpoint', filename=None):
    os.makedirs(dir, exist_ok=True)
    if not filename:
        filename = os.path.join(dir, name + f"_e{state['epoch']}" + '.pth')
    torch.save(state, filename)
    if is_best:
        best_filename = dir + name + '_model_best.pth'
        shutil.copyfile(filename, best_filename)
        
        
def save_checkpoint_best_only(state, dir='checkpoints/', name='checkpoint'):
    os.makedirs(dir, exist_ok=True)
    best_filename = os.path.join(dir, name + '_model_best.pth')
    torch.save(state, best_filename)


def set_seed(seed):
    """for reproducibility
    :param seed:
    :return:
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

        
class AverageMeter():
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
    
@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    if maxk == 1:
        return res[0]
    else:
        return res
    
    
@torch.no_grad()
def multilabel_cls_exact_match(pred, target):
    # pdb.set_trace()
    # https://mmuratarat.github.io/2020-01-25/multilabel_classification_metrics
    return torch.sum(torch.sum(pred==target, dim=1)==target.shape[1])/target.shape[0]



def trim(stat, prefix='module'):
    r"""Remove prefix of state_dict keys.
    """

    stat_new = OrderedDict()
    for k, v in stat.items():
        if k.startswith(prefix):
            stat_new[k[len(prefix)+1:]] = v

    return stat_new if stat_new else stat


def adjust_lr(optimizer, new_lr):
    print('change learning rate:',new_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def plot_line(x, y, plot_name):
    fig = plt.figure()
    ax = plt.axes()
    ax.plot(x, y)
    plt.show()
    plt.ylabel('LR')
    plt.xlabel('Epoch')
    plt.tight_layout()
    plt.savefig(plot_name, bbox_inches='tight', transparent=True, pad_inches=0)
    plt.close()
    return 



def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases following the
    values of the cosine function between 0 and `pi * cycles` after a warmup
    period during which it increases linearly between 0 and 1.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)
