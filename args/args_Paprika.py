import os
import yaml
import argparse
from datetime import datetime
import random
import shutil

import torch

from utils.common_utils import copy_source, init_distributed, is_main_process


def get_args_parser():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--cfg', type=str, 
                        default="config/config.yml",
                        help="config file path")
    
    parser.add_argument("--load_pretrained", type=int, 
                        required=False,
                        help="whether to load pretrained weights for training")
    
    parser.add_argument("--checkpoint", type=str, 
                        required=False,
                        help="a path to model checkpoint file to load pretrained weights")
    
    parser.add_argument('--use_wandb', type=int, 
                        default=0,
                        help="1 means use wandb to log experiments, 0 otherwise")
    
    parser.add_argument('--use_ddp', type=int, 
                        default=0,
                        help="1 means use pytorch GPU parallel distributed training, 0 otherwise")
    
    parser.add_argument('--local_rank', type=int, default=0)
    
    args = parser.parse_args()
    
    
    #################################
    # Read yaml file to update args #
    #################################
    with open(args.cfg, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    for k, v in cfg.items():
        parser.add_argument('--{}'.format(k), default=v, type=type(v))
    args = parser.parse_args()
    
    
    #################################
    # Setup log and checkpoint path #
    #################################
    args.curr_time = datetime.now().strftime("%Y-%m-%dT%H-%M-%SZ")
    
    args.log_dir = os.path.abspath(
            os.path.join(args.log_dir, args.curr_time))
    args.checkpoint_dir = os.path.abspath(
            os.path.join(args.checkpoint_dir, args.curr_time))
    
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    try:
        copy_source(os.getcwd(), args.log_dir)
        
        sourcecode_folder_name = os.path.splitext(os.path.basename(os.getcwd()))[0]
        wandb_dir = os.path.join(args.log_dir, sourcecode_folder_name, 'wandb')
        if os.path.exists(wandb_dir):
            shutil.rmtree(wandb_dir)
            
    except:
        pass
            
        
    ###############################################
    # Dynamically modify some args here if needed #
    ###############################################
    if args.num_workers == -1:
        args.num_workers = torch.get_num_threads() - 1
    
    
    args.working_abspath = os.path.abspath('./')
    
    if args.seed < 0:
        args.seed = random.randint(0, 1000000)
        
        
    ###############
    # Setup DDP #
    ###############
    if args.use_ddp:
        init_distributed()
        args.device = torch.device('cuda')
        args.world_size = torch.cuda.device_count()
        args.local_rank = int(os.environ['LOCAL_RANK'])
    
    else:
        if not torch.cuda.is_available():
            args.device = torch.device('cpu')
        else:
            args.device = torch.device(args.device)

            
    ###############
    # Setup wandb #
    ###############
    if args.use_wandb:
        import wandb
        if (not args.use_ddp) or (args.use_ddp and is_main_process()):
            
            wandb.init(project=args.project, entity=args.entity, 
                       name=args.exp_name, notes=args.notes)
            wandb.config.update(args)

    
    return args

