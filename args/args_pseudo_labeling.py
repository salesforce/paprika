import os
import yaml
import argparse
from datetime import datetime
import shutil
import random

from utils.common_utils import copy_source


def get_args_parser():
    
    parser = argparse.ArgumentParser()
     
    parser.add_argument('--cfg', type=str, 
                        default="config/config.yml",
                        help="config file path")
    
    parser.add_argument('--machine_idx', type=int, 
                        default=1,
                        help="machine index of the current run (if manually use multiple machines)")
    parser.add_argument('--machine_each', type=int, 
                        default=500000,
                        help="number of samples for each machine to process")
    
    args = parser.parse_args()
    
    
    #################################
    # Read yaml file to update args #
    #################################
    with open(args.cfg, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    for k, v in cfg.items():
        parser.add_argument('--{}'.format(k), default=v, type=type(v))
    args = parser.parse_args()
    
    ################################
    # Modify random seed if needed #
    ################################
    if args.seed < 0:
        args.seed = random.randint(0, 1000000)
        
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
    
    
    ###############
    # Copy source #
    ###############
    try:
        copy_source(os.getcwd(), args.log_dir)
        
        sourcecode_folder_name = os.path.splitext(os.path.basename(os.getcwd()))[0]
        wandb_dir = os.path.join(args.log_dir, sourcecode_folder_name, 'wandb')
        if os.path.exists(wandb_dir):
            shutil.rmtree(wandb_dir)
            
    except:
        pass
            
    
    return args

