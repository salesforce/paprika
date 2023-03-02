import os
# os.environ['OMP_NUM_THREADS'] = '1'
import sys
sys.path.insert(0, os.path.abspath('./'))
import time
import platform
import warnings
warnings.filterwarnings("ignore")

from args.args_DS import get_args_parser
from datasets import return_dataset
from models import create_model
from utils.common_utils import (
    set_seed, 
    getLogger, need_logging, 
    save_checkpoint, trim,
    AverageMeter, accuracy, multilabel_cls_exact_match,
    get_rank, is_main_process)

import torch
from torch import nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
torch.autograd.set_detect_anomaly(True)
import torch.distributed as dist


def main_train_adapter(args):
    start_time = time.time()
    
    set_seed(args.seed)
    
    if args.use_ddp:
        rank = get_rank()
    else:
        rank = 0
    args.rank = rank
        
    if args.use_ddp and args.ddp_log_each_rank:  # if hope to log each rank, change log file path
        args.log_dir = os.path.join(args.log_dir, str(rank))
        os.makedirs(args.log_dir, exist_ok=True)
    
    logfile_path = os.path.abspath(
        os.path.join(args.log_dir, 'train_adapter.log'))
        
    if need_logging(args):
        logger = getLogger(name=__name__, path=logfile_path)

        logger.info("Rank: {} Working config: {}\n".format(rank, args))
        logger.info("Rank: {} Host: {}".format(rank, platform.node()))
        logger.info("Rank: {} Logfile path: {}".format(rank, logfile_path))
        logger.info("Rank: {} Checkpoint saving directory: {}".format(rank, args.checkpoint_dir))
        logger.info("\n" + "Rank: {} ".format(rank) + '-'*20)
        if torch.cuda.device_count() > 0:
            logger.info("Rank: {} Using {} GPU(s)".format(rank, torch.cuda.device_count()))
    else:
        logger = None
        
    if args.use_wandb and is_main_process():
        import wandb
        wandb.config.update(args, allow_val_change=True)
    
    # Define datasets
    adapter_train_dataset = return_dataset(args, logger, 'HowTo100M')
    
    if need_logging(args):
        logger.info('Rank: {} Total number of samples is {} for < adapter > training data'.format(
            rank, adapter_train_dataset.__len__()))

    if args.use_ddp and not args.partition_dataset:  # distributed training
        adapter_train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset=adapter_train_dataset, shuffle=True)  
        
        adapter_train_loader = DataLoader(adapter_train_dataset,
                                          batch_size=args.adapter_batch_size,
                                          sampler=adapter_train_sampler,
                                          num_workers=args.num_workers, 
                                          pin_memory=True)
    else:
        if args.use_ddp:
            from monai.data import ThreadDataLoader
            # https://github.com/Project-MONAI/tutorials/blob/main/acceleration/distributed_training/brats_training_ddp.py#L70
            adapter_train_loader = ThreadDataLoader(adapter_train_dataset,
                                              batch_size=args.adapter_batch_size,
                                              shuffle=True,
                                              num_workers=0)
        else:
            adapter_train_loader = DataLoader(adapter_train_dataset,
                                              batch_size=args.adapter_batch_size,
                                              shuffle=True,
                                              num_workers=args.num_workers, 
                                              # drop_last=True,
                                              pin_memory=True)
            
    # Define adapter model
    adapter_model = create_model(args, logger, args.adapter_name)
    
    if args.use_ddp:  # distributed training
        local_rank = int(os.environ['LOCAL_RANK'])
        if need_logging(args):
            logger.info('Rank: {} The local_rank is {}'.format(rank, local_rank))
        adapter_model.to(args.device)
        adapter_model = nn.parallel.DistributedDataParallel(adapter_model, device_ids=[local_rank])
    else:
        adapter_model = nn.DataParallel(adapter_model).to(args.device)
    adapter_model_n_parameters = sum(p.numel() for p in adapter_model.parameters() if p.requires_grad)
    if need_logging(args):
        logger.info('Rank: {} Number of params is {} for < adapter > training model'.format(rank, adapter_model_n_parameters))
    
    if args.load_pretrained: # Load checkpoint
        adapter_checkpoint = torch.load(args.checkpoint, map_location=args.device)
        adapter_params = adapter_checkpoint['state_dict']
        adapter_params = trim(adapter_params)
        adapter_model.module.load_state_dict(
            adapter_params, strict=False) if hasattr(
            adapter_model, 'module') else adapter_model.load_state_dict(
            adapter_params, strict=False)
        if need_logging(args):
            logger.info("Rank: {} Loaded adapter checkpoint from {}".format(rank, args.checkpoint))
            
    # Define adapter criterion
    adapter_criterion = torch.nn.BCEWithLogitsLoss(reduction='mean').to(args.device)         

    # Define adapter optimizer
    if args.adapter_optimizer == 'adam':
        adapter_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, adapter_model.parameters()), 
            lr=args.adapter_learning_rate, weight_decay=args.adapter_weight_decay)
    else:
        logger.info('adapter_optimizer != adam, not implemented!')
        os._exit(0)
        
    # Define adapter scheduler
    if not args.adapter_lr_warm_up:
        adapter_scheduler = None
        from utils.common_utils import adjust_lr
        
        adapter_lr_plan = {}
        # adapter_lr_plan = {20: 5.e-5, 30: 1.e-5}
    else:
        from utils.common_utils import get_cosine_schedule_with_warmup
        adapter_scheduler = get_cosine_schedule_with_warmup(
            adapter_optimizer, args.adapter_warmup_steps, len(adapter_train_loader) * args.adapter_num_epochs)
    
    if args.cudnn_benchmark:
        cudnn.benchmark = True
        
    
    if need_logging(args):
        logger.info('Rank: {} Starting training loop for the < adapter > ...'.format(rank))
    training_adapter_start_time = time.time()
          
    ##################################################################################################################
    for adapter_epoch in range(1, args.adapter_num_epochs + 1):
        
        if args.use_ddp and not args.partition_dataset:  # distributed training
            adapter_train_loader.sampler.set_epoch(adapter_epoch)
        
        if adapter_scheduler is None:
            if adapter_epoch in adapter_lr_plan:
                adjust_lr(adapter_optimizer, adapter_lr_plan[adapter_epoch])
        
        torch.cuda.empty_cache()
        
        #################################
        # --- train adapter for one epoch
        #################################
        if need_logging(args):
            logger.info('Rank: {} '.format(rank) + '='*90)
        train_adapter_for_one_epoch_start_time = time.time()
        adapter_acc, adapter_loss = train_adapter_for_one_epoch(
            args, logger, 
            adapter_train_loader, adapter_model, 
            adapter_criterion, adapter_optimizer, adapter_scheduler, 
            adapter_epoch)
        if need_logging(args):
            logger.info("Rank: {} Finished training < adapter > adapter_epoch-{}, took {} seconds".format(
                rank, adapter_epoch, round(time.time() - train_adapter_for_one_epoch_start_time, 2)))
            logger.info('Rank: {} '.format(rank) + '='*90)
        
        # save adapter checkpoint
        if adapter_epoch >= args.adapter_start_save_epoch and adapter_epoch % args.adapter_save_freq == 0:
            if args.use_ddp:  # distributed training
                if is_main_process():
                    save_adapter_checkpoint(args, logger, adapter_epoch, adapter_model, adapter_optimizer)
                dist.barrier()
            else:
                save_adapter_checkpoint(args, logger, adapter_epoch, adapter_model, adapter_optimizer)

        
        if args.use_wandb and is_main_process():
            wandb_logdict = {
                "adapter_epoch": adapter_epoch,
                "adapter_train_loss": adapter_loss,
                "adapter_train_acc": adapter_acc
            }
            
              
            wandb.log(wandb_logdict, step=adapter_epoch)
            
            
    if need_logging(args):
        logger.info('\n\n\n' + 'Rank: {} '.format(rank) + '#'*90)       
        logger.info("Rank: {} Finished training < adapter > for all epochs, took {} seconds".format(
            rank, round(time.time() - training_adapter_start_time, 2)))                    
                 
    if args.use_ddp:
        dist.destroy_process_group()
        
    return


def save_adapter_checkpoint(args, logger, adapter_epoch, adapter_model, adapter_optimizer):
    save_checkpoint(
        {'cfg': args, 
         'epoch': adapter_epoch,
         'state_dict': adapter_model.module.state_dict() if hasattr(
             adapter_model, 'module') else adapter_model.state_dict(),
         'optimizer': adapter_optimizer.state_dict()
        },  
        False,
        dir=args.checkpoint_dir, 
        name='Adapter-' + args.curr_time,
        filename = os.path.join(args.checkpoint_dir, 'Adapter-' + args.curr_time + f"_e{adapter_epoch}" + '.pth')
    )
    
    if need_logging(args):
        logger.info('Rank: {} Checkpoint saved in {}'.format(
            args.rank,
            os.path.abspath(
                os.path.join(
                    args.checkpoint_dir, 
                    'Adapter-' + args.curr_time + f"_e{adapter_epoch}" + '.pth')
            )
        ))
    return


def train_adapter_for_one_epoch(
    args, logger, 
    train_loader, model, criterion, optimizer, scheduler, epoch):
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss = AverageMeter()  
    acc = AverageMeter()
    
    batch_start_time = time.time()

    model.train()
    criterion.train() 
    
    if hasattr(args, 'adapter_save_by_steps') and args.adapter_save_by_steps:
        save_count = 0
    
    for i, batch_data in enumerate(train_loader):
        segment_video_feat, pseudo_targets = batch_data
        targets_thisbatch = pseudo_targets.to(args.device)
            
        data_time.update(time.time() - batch_start_time)
        
        optimizer.zero_grad()
        if (args.adapter_name == 'mlp_with_skip' and 
            args.skip_connection_refined_feat_ratio == 'learnable' and 
            epoch > 10):
            preds_thisbatch = model(segment_video_feat, update_ratio=True)
        else:
            preds_thisbatch = model(segment_video_feat)
        
        # measure accuracy and record loss 
        loss_thisbatch = criterion(preds_thisbatch, targets_thisbatch) 

        y_true = targets_thisbatch.long()
        y_pred = (preds_thisbatch>0.5).long()
        acc_thisbatch = multilabel_cls_exact_match(y_pred, y_true)

        loss.update(loss_thisbatch.item(), len(targets_thisbatch))
        acc.update(acc_thisbatch.item(), len(targets_thisbatch))
            
            
        loss_thisbatch.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
            
        # save adapter checkpoint
        if hasattr(args, 'adapter_save_by_steps') and args.adapter_save_by_steps:
            if epoch >= args.adapter_start_save_epoch and ((i+1) % args.adapter_save_step_freq == 0):
                save_count += 1
                if args.use_ddp:  # distributed training
                    if is_main_process():
                        save_adapter_checkpoint(
                            args, logger, 
                            '{}-{}'.format(epoch, save_count), 
                            model, optimizer)
                    dist.barrier()
                else:
                    save_adapter_checkpoint(
                        args, logger, 
                        '{}-{}'.format(epoch, save_count),
                        model, optimizer)
        
        # finish
        batch_time.update(time.time() - batch_start_time)
        batch_start_time = time.time()
        
        # log loss of this batch
        if (i+1) % args.adapter_batch_train_log_freq == 0:
            if need_logging(args):
                logger.info('Rank: {} '.format(args.rank) + 'Train < Adapter > [e{:02d}][{}/{}] '
                            'Batch Processing Time {batch_time.avg:.3f} '
                            'Batch Data Loading Time {data_time.avg:.3f} '
                            'Loss {loss.avg:.4f} '
                            'Acc {acc.avg:.4f}'.format(
                                epoch, i+1, len(train_loader), 
                                batch_time=batch_time, data_time=data_time, 
                                loss=loss, acc=acc))
            
    if need_logging(args):
        logger.info('Rank: {} '.format(args.rank) + 'Done training data loop!')
        log_line = 'Rank: {} Train < Adapter > [e{:02d}] ---- '.format(args.rank, epoch)
        log_line += 'End Loss {:.4f}  End Acc {:.4f} '.format(loss.avg, acc.avg)

        logger.info(log_line)

    return acc.avg, loss.avg
    

if __name__ == '__main__':
    
    args = get_args_parser()
    
    main_train_adapter(args)
    