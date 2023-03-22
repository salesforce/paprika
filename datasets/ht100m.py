import os
import pdb
import sys
from pathlib import Path 
import pickle
import numpy as np
import random
import copy
import json
import time
import glob
from tqdm import tqdm
from collections import defaultdict
from scipy.sparse import csr_matrix

import torch
from torch.utils.data import Dataset

from utils.common_utils import numpy_topk_indices, minmax_normalize, need_logging


class HT100M(Dataset):
    def __init__(self, args, logger):
        
        self.args, self.logger = args, logger
        
        load_time_start = time.time()
        if (not hasattr(args, 'partition_dataset')) or (
            not args.partition_dataset) or (
            not args.use_ddp):
            
            with open(os.path.join(args.howto100m_dir, 
                                   'feats', 
                                   'feats_all-mean_agg.pickle'), 'rb') as f:
                self.feats_all = pickle.load(f)
        else:
            with open(os.path.join(
                args.howto100m_dir, 'feats', 
                'feats_all-mean_agg-rank_{}-of-{}.pickle'.format(
                    args.rank, args.world_size)), 'rb') as f:
                self.feats_all = pickle.load(f)
            
                
        if need_logging(args):
            logger.info('\n\nRank: {} '.format(args.rank) + 
                        'Loading video feats (len: {}) took {} seconds'.format(
                            len(self.feats_all), 
                            round(time.time()-load_time_start, 2)))
        
        if 'PKG' in self.args.adapter_objective:
            if 'VNM' in self.args.adapter_objective:
                if (not hasattr(args, 'partition_dataset')) or (
                    not args.partition_dataset) or (
                    not args.use_ddp):
                    
                    sample_pseudo_label_savepath = os.path.join(
                        args.howto100m_dir, 'pseudo_labels', 
                        'VNM-criteria_{}-threshold_{}-topK_{}-size_{}.pickle'.format(
                            args.label_find_matched_nodes_criteria, 
                            args.label_find_matched_nodes_for_segments_thresh,
                            args.label_find_matched_nodes_for_segments_topK,
                            args.num_nodes))
                else:
                    sample_pseudo_label_savepath = os.path.join(
                        args.howto100m_dir, 'pseudo_labels', 
                        'VNM-criteria_{}-threshold_{}-topK_{}-size_{}-rank_{}-of-{}.pickle'.format(
                            args.label_find_matched_nodes_criteria, 
                            args.label_find_matched_nodes_for_segments_thresh,
                            args.label_find_matched_nodes_for_segments_topK,
                            args.num_nodes,
                            args.rank, 
                            args.world_size))
                load_time_start = time.time()
                with open(sample_pseudo_label_savepath, 'rb') as f:
                    self.pseudo_labels_all_VNM = pickle.load(f)
                if need_logging(args):
                    logger.info('Rank: {} '.format(args.rank) +
                                'Loading VNM pseudo labels (len: {}) took {} seconds'.format(
                                    len(self.pseudo_labels_all_VNM), 
                                    round(time.time()-load_time_start, 2)))
            
            if 'VTM' in self.args.adapter_objective:
                if (not hasattr(args, 'partition_dataset')) or (
                    not args.partition_dataset) or (
                    not args.use_ddp):
                    
                    sample_pseudo_label_savepath = os.path.join(
                        args.howto100m_dir, 'pseudo_labels', 
                        'VTM-criteria_{}-threshold_{}-topK_{}.pickle'.format(
                            args.label_find_tasks_criteria, 
                            args.label_find_tasks_thresh,
                            args.label_find_tasks_topK))
                else:
                    sample_pseudo_label_savepath = os.path.join(
                        args.howto100m_dir, 'pseudo_labels', 
                        'VTM-criteria_{}-threshold_{}-topK_{}-rank_{}-of-{}.pickle'.format(
                            args.label_find_tasks_criteria, 
                            args.label_find_tasks_thresh,
                            args.label_find_tasks_topK,
                            args.rank, 
                            args.world_size))
                load_time_start = time.time()
                with open(sample_pseudo_label_savepath, 'rb') as f:
                    self.pseudo_labels_all_VTM = pickle.load(f)
                if need_logging(args):
                    logger.info('Rank: {} '.format(args.rank) +
                                'Loading VTM pseudo labels (len: {}) took {} seconds'.format(
                                    len(self.pseudo_labels_all_VTM), 
                                    round(time.time()-load_time_start, 2)))
            
            
            if 'TCL' in self.args.adapter_objective:
                if (not hasattr(args, 'partition_dataset')) or (
                    not args.partition_dataset) or (
                    not args.use_ddp):
                    
                    sample_pseudo_label_savepath = os.path.join(
                        args.howto100m_dir, 'pseudo_labels', 
                        'TCL-criteria_{}-threshold_{}-topK_{}.pickle'.format(
                            args.label_find_tasks_criteria, 
                            args.label_find_tasks_thresh,
                            args.label_find_tasks_topK))
                else:
                    sample_pseudo_label_savepath = os.path.join(
                        args.howto100m_dir, 'pseudo_labels',  
                        'TCL-criteria_{}-threshold_{}-topK_{}-rank_{}-of-{}.pickle'.format(
                            args.label_find_tasks_criteria, 
                            args.label_find_tasks_thresh,
                            args.label_find_tasks_topK,
                            args.rank, 
                            args.world_size))
                load_time_start = time.time()
                with open(sample_pseudo_label_savepath, 'rb') as f:
                    self.pseudo_labels_all_TCL = pickle.load(f)
                if need_logging(args):
                    logger.info('Rank: {} '.format(args.rank) +
                                'Loading TCL pseudo labels (len: {}) took {} seconds'.format(
                                    len(self.pseudo_labels_all_TCL), 
                                    round(time.time()-load_time_start, 2)))
            
            if 'NRL' in self.args.adapter_objective:
                for cap in args.adapter_NRL_num_neighbors_to_consider:
                    assert cap <= args.label_find_neighbors_topK
                
                if (not hasattr(args, 'partition_dataset')) or (
                    not args.partition_dataset) or (
                    not args.use_ddp):
                    
                    sample_pseudo_label_savepath = os.path.join(
                        args.howto100m_dir, 'pseudo_labels', 
                        'NRL-hop_{}-criteria_{}-threshold_{}-topK_{}-size_{}.pickle'.format(
                            args.label_khop,
                            args.label_find_neighbors_criteria, 
                            args.label_find_neighbors_thresh,
                            args.label_find_neighbors_topK,
                            args.num_nodes))
                else:
                    sample_pseudo_label_savepath = os.path.join(
                        args.howto100m_dir, 'pseudo_labels', 
                        'NRL-hop_{}-criteria_{}-threshold_{}-topK_{}-size_{}-rank_{}-of-{}.pickle'.format(
                            args.label_khop,
                            args.label_find_neighbors_criteria, 
                            args.label_find_neighbors_thresh,
                            args.label_find_neighbors_topK,
                            args.num_nodes,
                            args.rank, 
                            args.world_size))
                load_time_start = time.time()
                with open(sample_pseudo_label_savepath, 'rb') as f:
                    self.pseudo_labels_all_NRL = pickle.load(f)
                if need_logging(args):
                    logger.info('Rank: {} '.format(args.rank) +
                                'Loading NRL pseudo labels (len: {}) took {} seconds'.format(
                                    len(self.pseudo_labels_all_NRL), 
                                    round(time.time()-load_time_start, 2)))
            
            
        elif 'DS' in self.args.adapter_objective:
            if (not hasattr(args, 'partition_dataset')) or (
                not args.partition_dataset) or (
                not args.use_ddp):
                
                sample_pseudo_label_savepath = os.path.join(
                    args.howto100m_dir, 'DS_pseudo_labels', 
                    'DS-criteria_{}-threshold_{}-topK_{}-size_{}.pickle'.format(
                        args.label_find_matched_steps_criteria, 
                        args.label_find_matched_steps_for_segments_thresh,
                        args.label_find_matched_steps_for_segments_topK,
                        args.adapter_num_classes
                        ))
            else:
                sample_pseudo_label_savepath = os.path.join(
                    args.howto100m_dir, 'DS_pseudo_labels', 
                    'DS-criteria_{}-threshold_{}-topK_{}-size_{}-rank_{}-of-{}.pickle'.format(
                        args.label_find_matched_steps_criteria, 
                        args.label_find_matched_steps_for_segments_thresh,
                        args.label_find_matched_steps_for_segments_topK,
                        args.adapter_num_classes,
                        args.rank,
                        args.world_size
                        ))
            load_time_start = time.time()
            with open(sample_pseudo_label_savepath, 'rb') as f:
                self.pseudo_labels_all_DS = pickle.load(f)
            if need_logging(args):
                logger.info('Rank: {} '.format(args.rank) +
                            'Loading DS pseudo labels (len: {}) took {} seconds'.format(
                                len(self.pseudo_labels_all_DS), 
                                round(time.time()-load_time_start, 2)))
        else:
            if need_logging(args):
                self.logger.info(
                    'Rank: {} '.format(args.rank) + 
                    'The adapter_objective is not implemented!' +
                    '\nFunc: {}\nFile:{}'.format(__name__, __file__))
            os._exit(0)

            
    def __len__(self):
        return len(self.feats_all)
                   
        
    def parse_pseudo_label_VNM(self, index):
        pseudo_label_indices = self.pseudo_labels_all_VNM[index]['indices']
        
        pseudo_label_VNM = np.zeros(self.args.num_nodes)
        for i in range(self.args.adapter_VNM_num_matched_nodes_for_segments):
            pseudo_label_VNM[pseudo_label_indices[i]] = 1
            
        return pseudo_label_VNM
    
    
    def parse_pseudo_label_VTM(self, index):
        if self.args.adapter_VTM_enable_wikihow_tasks:
            wikihow_task_ids = self.pseudo_labels_all_VTM[index]['wikihow_tasks']
        if self.args.adapter_VTM_enable_howto100m_tasks:
            howto100m_task_ids = self.pseudo_labels_all_VTM[index][
                'howto100m_tasks']['indices'][:self.args.adapter_VTM_num_howto100m_tasks_to_consider]
            
        if self.args.adapter_VTM_enable_wikihow_tasks:
            pseudo_label_VTM_wikihow = np.zeros(self.args.wikihow_num_tasks)
            for i in range(len(wikihow_task_ids)):
                pseudo_label_VTM_wikihow[wikihow_task_ids[i]] = 1

        if self.args.adapter_VTM_enable_howto100m_tasks:
            pseudo_label_VTM_howto100m = np.zeros(self.args.howto100m_num_tasks)
            for i in range(len(howto100m_task_ids)):
                pseudo_label_VTM_howto100m[howto100m_task_ids[i]] = 1

        if (self.args.adapter_VTM_enable_wikihow_tasks and 
            self.args.adapter_VTM_enable_howto100m_tasks):
            pseudo_label_VTM = (pseudo_label_VTM_wikihow, pseudo_label_VTM_howto100m)
        elif self.args.adapter_VTM_enable_wikihow_tasks:
            pseudo_label_VTM = pseudo_label_VTM_wikihow
        elif self.args.adapter_VTM_enable_howto100m_tasks:
            pseudo_label_VTM = pseudo_label_VTM_howto100m
                
        return pseudo_label_VTM
    
    
    def parse_pseudo_label_TCL(self, index):
        if self.args.adapter_TCL_enable_wikihow_tasknodes:
            wikihow_tasknode_ids = self.pseudo_labels_all_TCL[index]['wikihow_tasknodes']
        if self.args.adapter_TCL_enable_howto100m_tasknodes:
            howto100m_tasknode_ids = self.pseudo_labels_all_TCL[index][
                'howto100m_tasknodes']['indices'][
                :self.args.adapter_TCL_num_howto100m_tasknodes_to_consider]
            
        if self.args.adapter_TCL_enable_wikihow_tasknodes:
            pseudo_label_TCL_wikihownodes = np.zeros(self.args.num_nodes)
            for i in range(len(wikihow_tasknode_ids)):
                pseudo_label_TCL_wikihownodes[wikihow_tasknode_ids[i]] = 1

        if self.args.adapter_TCL_enable_howto100m_tasknodes:
            pseudo_label_TCL_howto100mnodes = np.zeros(self.args.num_nodes)
            for i in range(len(howto100m_tasknode_ids)):
                pseudo_label_TCL_howto100mnodes[howto100m_tasknode_ids[i]] = 1

        if (self.args.adapter_TCL_enable_wikihow_tasknodes and 
            self.args.adapter_TCL_enable_howto100m_tasknodes):

            pseudo_label_TCL = (pseudo_label_TCL_wikihownodes, pseudo_label_TCL_howto100mnodes)
        elif self.args.adapter_TCL_enable_wikihow_tasknodes:
            pseudo_label_TCL = pseudo_label_TCL_wikihownodes
        elif self.args.adapter_TCL_enable_howto100m_tasknodes:
            pseudo_label_TCL = pseudo_label_TCL_howto100mnodes
                
        return pseudo_label_TCL
        
    
    def parse_pseudo_label_NRL(self, index):
        NRL_pseudo_labels_this_sample = self.pseudo_labels_all_NRL[index] 
        NRL_sub_name = ['{}-hop-{}'.format(
                h, direction) 
                        for direction in ['out', 'in']
                        for h in range(1, self.args.pretrain_khop+1)]
        # ['1-hop-out', '2-hop-out', '1-hop-in', '2-hop-in']
            
        pseudo_label_NRL = torch.zeros((len(NRL_sub_name), self.args.num_nodes))

        for NRL_sub_idx in range(len(NRL_sub_name)):
            pseudo_label_indices = NRL_pseudo_labels_this_sample[
                NRL_sub_name[NRL_sub_idx]]['indices']

            cap = self.args.adapter_NRL_num_neighbors_to_consider[
                int(NRL_sub_name[NRL_sub_idx].split('-')[0]) - 1]
            pseudo_label_indices = pseudo_label_indices[:cap]
                
            for node_idx in pseudo_label_indices:
                 pseudo_label_NRL[NRL_sub_idx, node_idx] = 1
        
        return pseudo_label_NRL
    
    
    def parse_pseudo_label_DS(self, index):
        pseudo_label_indices = self.pseudo_labels_all_DS[index]['indices']
            
        pseudo_label_DS = np.zeros(self.args.adapter_num_classes)
        for i in range(len(pseudo_label_indices)):
            pseudo_label_DS[pseudo_label_indices[i]] = 1
                
        return pseudo_label_DS
     
    
    def __getitem__(self, index):
        
        segment_video_feat = self.feats_all[index]
        # (512,)

        if 'PKG' in self.args.adapter_objective:
            # obtain answers for various adapter question types
            if 'VNM' in self.args.adapter_objective:
                pseudo_label_VNM = self.parse_pseudo_label_VNM(index)
                
            if 'VTM' in self.args.adapter_objective:
                pseudo_label_VTM = self.parse_pseudo_label_VTM(index)
                
            if 'TCL' in self.args.adapter_objective:
                pseudo_label_TCL = self.parse_pseudo_label_TCL(index)
                
            if 'NRL' in self.args.adapter_objective:
                pseudo_label_NRL = self.parse_pseudo_label_NRL(index)
                
            if self.args.adapter_objective == 'PKG_VNM_VTM_TCL_NRL':
                return (torch.FloatTensor(segment_video_feat),
                        pseudo_label_VNM,
                        pseudo_label_VTM, 
                        pseudo_label_TCL, 
                        pseudo_label_NRL)

        else:  # 'DS'
            pseudo_label_DS = self.parse_pseudo_label_DS(index)
            return torch.FloatTensor(segment_video_feat), pseudo_label_DS
        

        