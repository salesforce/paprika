import os
# os.environ['OMP_NUM_THREADS'] = '1'
import sys
sys.path.insert(0, os.path.abspath('./'))
import time
import platform
import warnings
warnings.filterwarnings("ignore")

from args.args_PAFoMo import get_args_parser
from datasets import return_dataset
from models import create_model
from utils.common_utils import (
    set_seed, 
    getLogger, need_logging, 
    save_checkpoint, trim,
    AverageMeter, global_meters_all_avg, accuracy, multilabel_cls_exact_match,
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
            from monai.data import ThreadDataLoader  # faster
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
    adapter_criterion = []
    if 'VNM' in args.adapter_objective:
        VNM_criterion = torch.nn.BCEWithLogitsLoss(reduction='mean').to(args.device)
        adapter_criterion.append(VNM_criterion)
            
    if 'VTM' in args.adapter_objective:
        if args.adapter_VTM_enable_wikihow_tasks and args.adapter_VTM_enable_howto100m_tasks:
            VTM_criterion = [
                torch.nn.BCEWithLogitsLoss(reduction='mean').to(args.device) for _ in range(2)]
            adapter_criterion.append(VTM_criterion)
        else:
            assert args.adapter_VTM_enable_wikihow_tasks or args.adapter_VTM_enable_howto100m_tasks
            VTM_criterion = torch.nn.BCEWithLogitsLoss(reduction='mean').to(args.device)
            adapter_criterion.append(VTM_criterion)
            
    if 'TCL' in args.adapter_objective:
        if args.adapter_TCL_enable_wikihow_tasknodes and args.adapter_TCL_enable_howto100m_tasknodes:
            TCL_criterion = [
                torch.nn.BCEWithLogitsLoss(reduction='mean').to(args.device) for _ in range(2)]
            adapter_criterion.append(TCL_criterion)
        else:
            assert args.adapter_TCL_enable_wikihow_tasknodes or args.adapter_TCL_enable_howto100m_tasknodes
            TCL_criterion = torch.nn.BCEWithLogitsLoss(reduction='mean').to(args.device)
            adapter_criterion.append(TCL_criterion)
        
    if 'NRL' in args.adapter_objective:
        NRL_criterion = [
            torch.nn.BCEWithLogitsLoss(reduction='mean').to(args.device) for _ in range(2*args.pretrain_khop)]
        adapter_criterion.append(NRL_criterion)
        
    assert len(adapter_criterion) > 0
    

    # Define adapter optimizer
    if args.adapter_optimizer == 'adam':
        adapter_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, adapter_model.parameters()), 
            lr=args.adapter_learning_rate, weight_decay=args.adapter_weight_decay)
    else:
        if need_logging(args):
            logger.info('Rank: {} adapter_optimizer != adam, not implemented!\nFunc: {}\nFile:{}'.format(
                rank, __name__, __file__))
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
                "adapter_epoch": adapter_epoch
            }
            
            if 'VNM' in args.adapter_objective:
                adapter_train_acc_VNM = adapter_acc['VNM']
                adapter_train_loss_VNM = adapter_loss['VNM']
                wandb_logdict.update({
                    "adapter_train_acc_VNM": adapter_train_acc_VNM,
                    "adapter_train_loss_VNM": adapter_train_loss_VNM
                })
                
                if args.use_ddp and not args.partition_dataset:
                    wandb_logdict.update({
                        "adapter_train_acc_VNM_global": adapter_acc['VNM_global'],
                        "adapter_train_loss_VNM_global": adapter_loss['VNM_global']
                    })
                
            if 'VTM' in args.adapter_objective:
                adapter_train_acc_VTM = adapter_acc['VTM']
                adapter_train_loss_VTM = adapter_loss['VTM']
                wandb_logdict.update({
                    "adapter_train_acc_VTM": adapter_train_acc_VTM,
                    "adapter_train_loss_VTM": adapter_train_loss_VTM
                })
                
                if args.adapter_VTM_enable_wikihow_tasks and args.adapter_VTM_enable_howto100m_tasks:
                    wandb_logdict.update({
                        "adapter_train_acc_VTM_wikihow": adapter_acc['VTM_wikihow'],
                        "adapter_train_loss_VTM_wikihow": adapter_loss['VTM_wikihow'],
                        "adapter_train_acc_VTM_howto100m": adapter_acc['VTM_howto100m'],
                        "adapter_train_loss_VTM_howto100m": adapter_loss['VTM_howto100m']
                    })
                    
                
                if args.use_ddp and not args.partition_dataset:
                    wandb_logdict.update({
                        "adapter_train_acc_VTM_global": adapter_acc['VTM_global'],
                        "adapter_train_loss_VTM_global": adapter_loss['VTM_global']
                    })
                    
                    if args.adapter_VTM_enable_wikihow_tasks and args.adapter_VTM_enable_howto100m_tasks:
                        wandb_logdict.update({
                            "adapter_train_acc_VTM_wikihow_global": adapter_acc['VTM_wikihow_global'],
                            "adapter_train_loss_VTM_wikihow_global": adapter_loss['VTM_wikihow_global'],
                            "adapter_train_acc_VTM_howto100m_global": adapter_acc['VTM_howto100m_global'],
                            "adapter_train_loss_VTM_howto100m_global": adapter_loss['VTM_howto100m_global']
                        })
                        
            if 'TCL' in args.adapter_objective:
                adapter_train_acc_TCL = adapter_acc['TCL']
                adapter_train_loss_TCL = adapter_loss['TCL']
                wandb_logdict.update({
                    "adapter_train_acc_TCL": adapter_train_acc_TCL,
                    "adapter_train_loss_TCL": adapter_train_loss_TCL
                })
                
                if (args.adapter_TCL_enable_wikihow_tasknodes and 
                    args.adapter_TCL_enable_howto100m_tasknodes):
                    
                    wandb_logdict.update({
                        "adapter_train_acc_TCL_wikihow": adapter_acc['TCL_wikihow'],
                        "adapter_train_loss_TCL_wikihow": adapter_loss['TCL_wikihow'],
                        "adapter_train_acc_TCL_howto100m": adapter_acc['TCL_howto100m'],
                        "adapter_train_loss_TCL_howto100m": adapter_loss['TCL_howto100m']
                    })
                    
                
                if args.use_ddp and not args.partition_dataset:
                    wandb_logdict.update({
                        "adapter_train_acc_TCL_global": adapter_acc['TCL_global'],
                        "adapter_train_loss_TCL_global": adapter_loss['TCL_global']
                    })
                    
                    if (args.adapter_TCL_enable_wikihow_tasknodes and 
                        args.adapter_TCL_enable_howto100m_tasknodes):
                        
                        wandb_logdict.update({
                            "adapter_train_acc_TCL_wikihow_global": adapter_acc['TCL_wikihow_global'],
                            "adapter_train_loss_TCL_wikihow_global": adapter_loss['TCL_wikihow_global'],
                            "adapter_train_acc_TCL_howto100m_global": adapter_acc['TCL_howto100m_global'],
                            "adapter_train_loss_TCL_howto100m_global": adapter_loss['TCL_howto100m_global']
                        })
                    
            if 'NRL' in args.adapter_objective:
                adapter_train_acc_NRL = adapter_acc['NRL']
                adapter_train_loss_NRL = adapter_loss['NRL']
                wandb_logdict.update({
                    "adapter_train_acc_NRL": adapter_train_acc_NRL,
                    "adapter_train_loss_NRL": adapter_train_loss_NRL
                })
                
                if args.use_ddp and not args.partition_dataset:
                    wandb_logdict.update({
                        "adapter_train_acc_NRL_global": adapter_acc['NRL_global'],
                        "adapter_train_loss_NRL_global": adapter_loss['NRL_global']
                    })
                    
                    
            wandb.log(wandb_logdict, step=adapter_epoch)
            
            
    if need_logging(args):
        logger.info('\n\n\n' + 'Rank: {} '.format(rank) + '#'*90)       
        logger.info("Rank: {} Finished training < adapter > for all epochs, took {} seconds".format(
            rank, round(time.time() - training_adapter_start_time, 2)))                    
    
    
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
    
    acc = dict()
    loss = dict()
    
    if 'VNM' in args.adapter_objective:
        loss_VNM = AverageMeter()  
        acc_VNM = AverageMeter()
        
    if 'VTM' in args.adapter_objective:
        loss_VTM = AverageMeter()  
        acc_VTM = AverageMeter()
        if args.adapter_VTM_enable_wikihow_tasks and args.adapter_VTM_enable_howto100m_tasks:
            loss_VTM_wikihow = AverageMeter()  
            acc_VTM_wikihow = AverageMeter()
            
            loss_VTM_howto100m = AverageMeter()  
            acc_VTM_howto100m = AverageMeter()
            
    if 'TCL' in args.adapter_objective:
        loss_TCL = AverageMeter()  
        acc_TCL = AverageMeter()
        if args.adapter_TCL_enable_wikihow_tasknodes and args.adapter_TCL_enable_howto100m_tasknodes:
            loss_TCL_wikihow = AverageMeter()  
            acc_TCL_wikihow = AverageMeter()
            
            loss_TCL_howto100m = AverageMeter()  
            acc_TCL_howto100m = AverageMeter()
            
    if 'NRL' in args.adapter_objective:
        loss_NRL = AverageMeter()  
        acc_NRL = AverageMeter()
    
    model.train()
    
    
    if args.adapter_objective == 'PKG_VNM_VTM_TCL_NRL':
        VNM_criterion, VTM_criterion, TCL_criterion, NRL_criterion = criterion
        
        VNM_criterion.train() 
        
        if args.adapter_VTM_enable_wikihow_tasks and args.adapter_VTM_enable_howto100m_tasks:
            for VTM_sub_idx in range(len(VTM_criterion)):
                VTM_criterion[VTM_sub_idx].train() 
        else:
            VTM_criterion.train() 
            
        if args.adapter_TCL_enable_wikihow_tasknodes and args.adapter_TCL_enable_howto100m_tasknodes:
            for TCL_sub_idx in range(len(TCL_criterion)):
                TCL_criterion[TCL_sub_idx].train() 
        else:
            TCL_criterion.train() 
            
        # NRL_criterion.train() 
        for NRL_sub_idx in range(2*args.pretrain_khop):
            NRL_criterion[NRL_sub_idx].train() 
    
    
    if hasattr(args, 'adapter_save_by_steps') and args.adapter_save_by_steps:
        save_count = 0
        
    batch_start_time = time.time()

    for i, batch_data in enumerate(train_loader):
            
        if args.adapter_objective == 'PKG_VNM_VTM_TCL_NRL':
            segment_video_feat, pseudo_targets_VNM, pseudo_targets_VTM, pseudo_targets_TCL, pseudo_targets_NRL = batch_data
            pseudo_targets_VNM = pseudo_targets_VNM.to(args.device)
            
            if args.adapter_VTM_enable_wikihow_tasks and args.adapter_VTM_enable_howto100m_tasks:
                pseudo_targets_VTM = (pseudo_targets_VTM[0].to(args.device), pseudo_targets_VTM[1].to(args.device))
            else:
                pseudo_targets_VTM = pseudo_targets_VTM.to(args.device)
                
            if args.adapter_TCL_enable_wikihow_tasknodes and args.adapter_TCL_enable_howto100m_tasknodes:
                pseudo_targets_TCL = (pseudo_targets_TCL[0].to(args.device), pseudo_targets_TCL[1].to(args.device))
            else:
                pseudo_targets_TCL = pseudo_targets_TCL.to(args.device)
                
            pseudo_targets_NRL = pseudo_targets_NRL.to(args.device)
        
            
        data_time.update(time.time() - batch_start_time)
        bs = len(batch_data)
        
        optimizer.zero_grad()
        
        if args.adapter_objective == 'PKG_VNM_VTM_TCL_NRL':
            VNM_answer, VTM_answer, TCL_answer, NRL_answer = model(segment_video_feat)
            
        
        # measure accuracy and record loss 
        if 'VNM' in args.adapter_objective:
            loss_thisbatch_VNM = VNM_criterion(VNM_answer, pseudo_targets_VNM) 
            
            y_true = pseudo_targets_VNM.long()
            y_pred = (VNM_answer>0.5).long()
            acc_thisbatch_VNM = multilabel_cls_exact_match(y_pred, y_true)
            
            loss_VNM.update(loss_thisbatch_VNM.item(), bs)
            acc_VNM.update(acc_thisbatch_VNM.item(), bs)
        
        if 'VTM' in args.adapter_objective:
            
            if args.adapter_VTM_enable_wikihow_tasks and args.adapter_VTM_enable_howto100m_tasks:

                # wikihow
                VTM_sub_idx = 0
                loss_thisbatch_VTM_wikihow = VTM_criterion[VTM_sub_idx](
                    VTM_answer[VTM_sub_idx], pseudo_targets_VTM[VTM_sub_idx]) 

                y_true = pseudo_targets_VTM[VTM_sub_idx].long()
                y_pred = (VTM_answer[VTM_sub_idx]>0.5).long()
                acc_thisbatch_VTM_wikihow = multilabel_cls_exact_match(y_pred, y_true)

                # howto100m
                VTM_sub_idx = 1
                loss_thisbatch_VTM_howto100m = VTM_criterion[VTM_sub_idx](
                    VTM_answer[VTM_sub_idx], pseudo_targets_VTM[VTM_sub_idx]) 

                y_true = pseudo_targets_VTM[VTM_sub_idx].long()
                y_pred = (VTM_answer[VTM_sub_idx]>0.5).long()
                acc_thisbatch_VTM_howto100m = multilabel_cls_exact_match(y_pred, y_true)

            else:
                loss_thisbatch_VTM = VTM_criterion(VTM_answer, pseudo_targets_VTM) 
                y_true = pseudo_targets_VTM.long()
                y_pred = (VTM_answer>0.5).long()
                acc_thisbatch_VTM = multilabel_cls_exact_match(y_pred, y_true)
                    
                      
            if args.adapter_VTM_enable_wikihow_tasks and args.adapter_VTM_enable_howto100m_tasks:
                loss_VTM_wikihow.update(loss_thisbatch_VTM_wikihow.item(), bs)
                acc_VTM_wikihow.update(acc_thisbatch_VTM_wikihow.item(), bs) 
                
                loss_VTM_howto100m.update(loss_thisbatch_VTM_howto100m.item(), bs)
                acc_VTM_howto100m.update(acc_thisbatch_VTM_howto100m.item(), bs) 
                
                loss_thisbatch_VTM = loss_thisbatch_VTM_wikihow + loss_thisbatch_VTM_howto100m
                acc_thisbatch_VTM = (acc_thisbatch_VTM_wikihow + acc_thisbatch_VTM_howto100m) / 2
                
            loss_VTM.update(loss_thisbatch_VTM.item(), bs)
            acc_VTM.update(acc_thisbatch_VTM.item(), bs)   
               
                
        if 'TCL' in args.adapter_objective:
            if (args.adapter_TCL_enable_wikihow_tasknodes and 
                args.adapter_TCL_enable_howto100m_tasknodes):

                # wikihow
                TCL_sub_idx = 0
                loss_thisbatch_TCL_wikihow = TCL_criterion[TCL_sub_idx](
                    TCL_answer[TCL_sub_idx], pseudo_targets_TCL[TCL_sub_idx]) 

                y_true = pseudo_targets_TCL[TCL_sub_idx].long()
                y_pred = (TCL_answer[TCL_sub_idx]>0.5).long()
                acc_thisbatch_TCL_wikihow = multilabel_cls_exact_match(y_pred, y_true)

                # howto100m
                TCL_sub_idx = 1
                loss_thisbatch_TCL_howto100m = TCL_criterion[TCL_sub_idx](
                    TCL_answer[TCL_sub_idx], pseudo_targets_TCL[TCL_sub_idx]) 

                y_true = pseudo_targets_TCL[TCL_sub_idx].long()
                y_pred = (TCL_answer[TCL_sub_idx]>0.5).long()
                acc_thisbatch_TCL_howto100m = multilabel_cls_exact_match(y_pred, y_true)

            else:
                loss_thisbatch_TCL = TCL_criterion(TCL_answer, pseudo_targets_TCL) 
                y_true = pseudo_targets_TCL.long()
                y_pred = (TCL_answer>0.5).long()
                acc_thisbatch_TCL = multilabel_cls_exact_match(y_pred, y_true)
                    
                      
            if args.adapter_TCL_enable_wikihow_tasknodes and args.adapter_TCL_enable_howto100m_tasknodes:
                loss_TCL_wikihow.update(loss_thisbatch_TCL_wikihow.item(), bs)
                acc_TCL_wikihow.update(acc_thisbatch_TCL_wikihow.item(), bs) 
                
                loss_TCL_howto100m.update(loss_thisbatch_TCL_howto100m.item(), bs)
                acc_TCL_howto100m.update(acc_thisbatch_TCL_howto100m.item(), bs) 
                
                loss_thisbatch_TCL = loss_thisbatch_TCL_wikihow + loss_thisbatch_TCL_howto100m
                acc_thisbatch_TCL = (acc_thisbatch_TCL_wikihow + acc_thisbatch_TCL_howto100m) / 2
                
            loss_TCL.update(loss_thisbatch_TCL.item(), bs)
            acc_TCL.update(acc_thisbatch_TCL.item(), bs)   
                
                
        if 'NRL' in args.adapter_objective:
            for NRL_sub_idx in range(len(NRL_answer)):
                if NRL_sub_idx == 0:
                    loss_thisbatch_NRL = NRL_criterion[NRL_sub_idx](
                        NRL_answer[NRL_sub_idx], pseudo_targets_NRL[:, NRL_sub_idx, :])

                    y_true = pseudo_targets_NRL[:, NRL_sub_idx, :].long()
                    y_pred = (NRL_answer[NRL_sub_idx]>0.5).long()
                    acc_thisbatch_NRL = multilabel_cls_exact_match(y_pred, y_true)
                        
                else:
                    loss_thisbatch_NRL += NRL_criterion[NRL_sub_idx](
                        NRL_answer[NRL_sub_idx], pseudo_targets_NRL[:, NRL_sub_idx, :])

                    y_true = pseudo_targets_NRL[:, NRL_sub_idx, :].long()
                    y_pred = (NRL_answer[NRL_sub_idx]>0.5).long()
                    acc_thisbatch_NRL += multilabel_cls_exact_match(y_pred, y_true)
                        
                        
            acc_thisbatch_NRL = acc_thisbatch_NRL/len(NRL_answer)
            
            loss_NRL.update(loss_thisbatch_NRL.item(), bs)
            acc_NRL.update(acc_thisbatch_NRL.item(), bs)
            
            
        if args.adapter_objective == 'PKG_VNM_VTM_TCL_NRL':
            loss_thisbatch = (args.VNM_loss_ratio * loss_thisbatch_VNM + 
                              args.VTM_loss_ratio * loss_thisbatch_VTM + 
                              args.TCL_loss_ratio * loss_thisbatch_TCL + 
                              args.NRL_loss_ratio * loss_thisbatch_NRL)
            
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
            if args.adapter_objective == 'PKG_VNM_VTM_TCL_NRL':
                if need_logging(args):
                    logger.info('Rank: {} '.format(args.rank) + 'Train < Adapter > [e{:02d}][{}/{}] '
                                'Batch Processing Time {batch_time.avg:.3f} '
                                'Batch Data Loading Time {data_time.avg:.3f} '
                                'Loss VNM {loss_VNM.avg:.4f} '
                                'Acc VNM {acc_VNM.avg:.4f} '
                                'Loss VTM {loss_VTM.avg:.4f} '
                                'Acc VTM {acc_VTM.avg:.4f} '
                                'Loss TCL {loss_TCL.avg:.4f} '
                                'Acc TCL {acc_TCL.avg:.4f} '
                                'Loss NRL {loss_NRL.avg:.4f} '
                                'Acc NRL {acc_NRL.avg:.4f}'.format(
                                    epoch, i+1, len(train_loader), 
                                    batch_time=batch_time, data_time=data_time,
                                    loss_VNM=loss_VNM, acc_VNM=acc_VNM,
                                    loss_VTM=loss_VTM, acc_VTM=acc_VTM,
                                    loss_TCL=loss_TCL, acc_TCL=acc_TCL,
                                    loss_NRL=loss_NRL, acc_NRL=acc_NRL))
        
        
    if need_logging(args):
        logger.info('Rank: {} '.format(args.rank) + 'Done data loop!')
        
    if 'VNM' in args.adapter_objective:
        acc['VNM'] = acc_VNM.avg
        loss['VNM'] = loss_VNM.avg
    
    if 'VTM' in args.adapter_objective:
        acc['VTM'] = acc_VTM.avg
        loss['VTM'] = loss_VTM.avg
        
        if args.adapter_VTM_enable_wikihow_tasks and args.adapter_VTM_enable_howto100m_tasks:
            acc['VTM_wikihow'] = acc_VTM_wikihow.avg
            loss['VTM_wikihow'] = loss_VTM_wikihow.avg
            
            acc['VTM_howto100m'] = acc_VTM_howto100m.avg
            loss['VTM_howto100m'] = loss_VTM_howto100m.avg
            
    if 'TCL' in args.adapter_objective:
        acc['TCL'] = acc_TCL.avg
        loss['TCL'] = loss_TCL.avg
        
        if args.adapter_TCL_enable_wikihow_tasknodes and args.adapter_TCL_enable_howto100m_tasknodes:
            acc['TCL_wikihow'] = acc_TCL_wikihow.avg
            loss['TCL_wikihow'] = loss_TCL_wikihow.avg
            
            acc['TCL_howto100m'] = acc_TCL_howto100m.avg
            loss['TCL_howto100m'] = loss_TCL_howto100m.avg
            
    if 'NRL' in args.adapter_objective:
        acc['NRL'] = acc_NRL.avg
        loss['NRL'] = loss_NRL.avg
    
    
    if args.use_ddp:
        if not args.partition_dataset:  
            if 'VNM' in args.adapter_objective:
                acc['VNM_global'] = global_meters_all_avg(args, acc['VNM'])
                loss['VNM_global'] = global_meters_all_avg(args, loss['VNM']) 

            if 'VTM' in args.adapter_objective:
                acc['VTM_global'] = global_meters_all_avg(args, acc['VTM'])
                loss['VTM_global'] = global_meters_all_avg(args, loss['VTM'])

                if args.adapter_VTM_enable_wikihow_tasks and args.adapter_VTM_enable_howto100m_tasks:
                    acc['VTM_wikihow_global'] = global_meters_all_avg(args, acc['VTM_wikihow'])
                    loss['VTM_wikihow_global'] = global_meters_all_avg(args, loss['VTM_wikihow'])

                    acc['VTM_howto100m_global'] = global_meters_all_avg(args, acc['VTM_howto100m'])
                    loss['VTM_howto100m_global'] = global_meters_all_avg(args, loss['VTM_howto100m'])
            
            if 'TCL' in args.adapter_objective:
                acc['TCL_global'] = global_meters_all_avg(args, acc['TCL'])
                loss['TCL_global'] = global_meters_all_avg(args, loss['TCL'])

                if args.adapter_TCL_enable_wikihow_tasknodes and args.adapter_TCL_enable_howto100m_tasknodes:
                    acc['TCL_wikihow_global'] = global_meters_all_avg(args, acc['TCL_wikihow'])
                    loss['TCL_wikihow_global'] = global_meters_all_avg(args, loss['TCL_wikihow'])

                    acc['TCL_howto100m_global'] = global_meters_all_avg(args, acc['TCL_howto100m'])
                    loss['TCL_howto100m_global'] = global_meters_all_avg(args, loss['TCL_howto100m'])
            
            if 'NRL' in args.adapter_objective:
                acc['NRL_global'] = global_meters_all_avg(args, acc['NRL'])
                loss['NRL_global'] = global_meters_all_avg(args, loss['NRL'])

     
            if need_logging(args):
                log_line = 'Rank: {} Train < Adapter > [e{:02d}] '.format(args.rank, epoch)
                if 'VNM' in args.adapter_objective:
                    log_line += 'Global Loss VNM {:.4f}  Global Acc VNM {:.4f} '.format(loss['VNM_global'], acc['VNM_global'])
               
                if 'VTM' in args.adapter_objective:
                    log_line += 'Global Loss VTM {:.4f}  Global Acc VTM {:.4f} '.format(loss['VTM_global'], acc['VTM_global'])
                
                if 'TCL' in args.adapter_objective:
                    log_line += 'Global Loss TCL {:.4f}  Global Acc TCL {:.4f} '.format(loss['TCL_global'], acc['TCL_global'])
               
                if 'NRL' in args.adapter_objective:
                    log_line += 'Global Loss NRL {:.4f}  Global Acc NRL {:.4f} '.format(loss['NRL_global'], acc['NRL_global'])
                
                logger.info(log_line)
        else:
            if need_logging(args):
                log_line = 'Rank: {} Train < Adapter > [e{:02d}] ---- '.format(args.rank, epoch)
                
                if 'VNM' in args.adapter_objective:
                    log_line += 'End VNM Loss {:.4f}  End VNM Acc {:.4f} '.format(loss['VNM'], acc['VNM'])
                
                if 'VTM' in args.adapter_objective:
                    log_line += 'End VTM Loss {:.4f}  End VTM Acc {:.4f} '.format(loss['VTM'], acc['VTM'])
                    if args.adapter_VTM_enable_wikihow_tasks and args.adapter_VTM_enable_howto100m_tasks:
                        log_line += 'End VTM wikihow Loss {:.4f}  End VTM wikihow Acc {:.4f} '.format(
                            loss['VTM_wikihow'], acc['VTM_wikihow'])
                        log_line += 'End VTM howto100m Loss {:.4f}  End VTM howto100m Acc {:.4f} '.format(
                            loss['VTM_howto100m'], acc['VTM_howto100m'])
                        
                if 'TCL' in args.adapter_objective:
                    log_line += 'End TCL Loss {:.4f}  End TCL Acc {:.4f} '.format(loss['TCL'], acc['TCL'])
                    if args.adapter_TCL_enable_wikihow_tasknodes and args.adapter_TCL_enable_howto100m_tasknodes:
                        log_line += 'End TCL wikihow Loss {:.4f}  End TCL wikihow Acc {:.4f} '.format(
                            loss['TCL_wikihow'], acc['TCL_wikihow'])
                        log_line += 'End TCL howto100m Loss {:.4f}  End TCL howto100m Acc {:.4f} '.format(
                            loss['TCL_howto100m'], acc['TCL_howto100m'])
                        
                if 'NRL' in args.adapter_objective:
                    log_line += 'End NRL Loss {:.4f}  End NRL Acc {:.4f} '.format(loss['NRL'], acc['NRL'])
                    
                logger.info(log_line)
            
    return acc, loss
    
    
    

    

if __name__ == '__main__':
    
    args = get_args_parser()
    
    main_train_adapter(args)
     