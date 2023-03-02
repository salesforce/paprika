import os
import pickle
import pandas as pd
import numpy as np
import glob
from collections import defaultdict

import torch
from torch.utils.data import Dataset


def get_task_cls_train_test_splits(cross_task_video_dir, train_ratio=0.8):
    print('generating task cls train test splits ...')
    task_sids = os.listdir(cross_task_video_dir)
    
    train_split = defaultdict()
    test_split = defaultdict()
        
    for task_sid in task_sids:
        videos_this_task = glob.glob(os.path.join(cross_task_video_dir, task_sid, '*.*'))

        train_split[task_sid] = np.random.choice(
            videos_this_task, int(len(videos_this_task)*train_ratio), replace=False).tolist()
        test_split[task_sid] = [
            vid for vid in videos_this_task if vid not in train_split[task_sid]]
    return train_split, test_split


def get_step_cls_train_test_splits(cross_task_annot_dir, cross_task_feat_dir, train_ratio=0.8):
    with open(os.path.join(cross_task_annot_dir, 'crosstask_release/tasks_primary.txt'), 'r') as f:
        task_primary_lines = f.readlines() 
        
    task_meta = dict()
    lines_of_one_task = []
    steps = set()
    for i in range(len(task_primary_lines)):
        if not task_primary_lines[i] == '\n':
            lines_of_one_task.append(task_primary_lines[i])
        else:
            assert len(lines_of_one_task) == 5
            task_id = lines_of_one_task[0].split('\n')[0]
            task_name = lines_of_one_task[1].split('\n')[0]
            wikihow_url = lines_of_one_task[2].split('\n')[0]
            num_steps = int(lines_of_one_task[3].split('\n')[0])
            ordered_steps = lines_of_one_task[4].split('\n')[0].split(',')
            assert len(ordered_steps) == num_steps
            task_meta[task_id] = {
                'task_id': task_id,
                'task_name': task_name,
                'wikihow_url': wikihow_url,
                'num_steps': num_steps,
                'ordered_steps': ordered_steps
            }
            lines_of_one_task = []
            
            for step in ordered_steps:
                steps.add(step)
                
    
    task_video_combinations_with_annos = list(
        glob.glob(os.path.join(cross_task_annot_dir, 'crosstask_release/annotations/*.csv')))
    
    with open(os.path.join(cross_task_feat_dir, 'corrupted_videos.pickle'), 'rb') as f:
        corrupted_videos = pickle.load(f)
          
            
    task_video_combinations_with_annos_new = []
    for video_annot_file in task_video_combinations_with_annos:
        video_sid = video_annot_file.split('/')[-1].split('_', 1)[1].split('.csv')[0]
        if (os.path.exists(os.path.join(cross_task_feat_dir, video_sid))) and (video_sid not in corrupted_videos):
            task_video_combinations_with_annos_new.append(video_annot_file)
    
    task_video_combinations_with_annos = task_video_combinations_with_annos_new
    
    def splitting_step_cls_train_test(task_meta, steps, task_video_combinations_with_annos):
        train_split = np.random.choice(
                task_video_combinations_with_annos, 
            int(len(task_video_combinations_with_annos)*train_ratio), 
            replace=False).tolist()
        test_split = [
            vid for vid in task_video_combinations_with_annos if vid not in train_split]
    
        
        train_steps = set()
        for video_annot_file in train_split:
            anno_csv = pd.read_csv(
                video_annot_file, names=['step class id', 'start in seconds', 'end in seconds'])
            step_ids = anno_csv['step class id'].tolist()
            steps_of_this_task = task_meta[video_annot_file.split('/')[-1].split('_', 1)[0]]['ordered_steps']
            
            for step_id in step_ids:
                train_steps.add(steps_of_this_task[step_id-1])
                
        test_steps = set()
        for video_annot_file in test_split:
            anno_csv = pd.read_csv(
                video_annot_file, names=['step class id', 'start in seconds', 'end in seconds'])
            step_ids = anno_csv['step class id'].tolist()
            steps_of_this_task = task_meta[video_annot_file.split('/')[-1].split('_', 1)[0]]['ordered_steps']
            
            for step_id in step_ids:
                test_steps.add(steps_of_this_task[step_id-1])
        
        if len(train_steps) == len(steps) and len(test_steps) == len(steps):
            split_result_ok = True
        else:
            split_result_ok = False
        return train_split, test_split, split_result_ok

    split_result_ok = False
    while not split_result_ok:
        train_split, test_split, split_result_ok = splitting_step_cls_train_test(
            task_meta, steps, task_video_combinations_with_annos)
        
    return task_meta, steps, train_split, test_split


class CrossTask_Task_CLS(Dataset):
    def __init__(self, args, logger, split='train'):
        
        self.args = args
        self.logger = logger
        
        self.task_sid2iid = defaultdict()
        self.task_iid2sid = defaultdict()
        self.get_task_ids()
        
        with open(os.path.join(
            args.cross_task_s3d_feat_dir, 
            'task_cls_{}_split.pickle'.format(split)), 'rb') as f:
            split_samples = pickle.load(f)
        self.sample_video_paths = []
        for task in split_samples:
            self.sample_video_paths += split_samples[task] 
        
    def __len__(self):
        return len(self.sample_video_paths)
        
    def get_task_ids(self):
        task_sids = os.listdir(self.args.cross_task_video_dir)
        for i in range(len(task_sids)):
            task_sid = task_sids[i]
            self.task_sid2iid[task_sid] = i
            self.task_iid2sid[i] = task_sid
        return
    
    @staticmethod  
    def custom_collate(batch):
        # https://androidkt.com/create-dataloader-with-collate_fn-for-variable-length-input-in-pytorch/
        segments_list, label_list = [], []
        max_length = 0
        sequence_dim = None
        for (_segments, _label) in batch:
            if not sequence_dim:
                sequence_dim = _segments.shape[-1]
            max_length = max(max_length, len(_segments))
            segments_list.append(_segments)
            label_list.append(_label)
             
        mask_list = []
        for i in range(len(segments_list)):
            if len(segments_list[i]) < max_length:
                pad_length = max_length - len(segments_list[i])
                mask_list.append(torch.tensor([0]*len(segments_list[i])+[1]*pad_length))
                segments_list[i] = torch.cat(
                    [segments_list[i], torch.zeros((pad_length, sequence_dim))], dim=0)
            else:
                mask_list.append(torch.tensor([0]*max_length))
        return torch.stack(segments_list), torch.LongTensor(label_list), torch.stack(mask_list).bool()

    def __getitem__(self, index):
        sample_video_path = self.sample_video_paths[index]
        task_id = self.task_sid2iid[sample_video_path.split('/')[-2]]
        video_id = sample_video_path.split('/')[-1].split('.')[0]
       
        video_feats = np.load(
            os.path.join(self.args.cross_task_s3d_feat_dir, video_id, 'video.npy'))
        video_feats = np.mean(video_feats, axis=1)
        
        return torch.FloatTensor(video_feats), task_id
        
        
class CrossTask_Step_CLS(Dataset):
    def __init__(self, args, logger, split='train'):
        
        self.args = args
        self.logger = logger
        
        with open(os.path.join(
            args.cross_task_s3d_feat_dir, 'step_cls_task2step_meta.pickle'), 'rb') as f:
            self.task_meta = pickle.load(f)
        with open(os.path.join(
            args.cross_task_s3d_feat_dir, 'step_cls_set_of_steps.pickle'), 'rb') as f:
            self.steps = pickle.load(f)
            
        self.cls_sid2iid = dict()
        self.cls_iid2sid = dict()
        for step in self.steps:
            self.cls_sid2iid[step] = len(self.cls_sid2iid)
            self.cls_iid2sid[self.cls_sid2iid[step]] = step
        
        with open(os.path.join(
            args.cross_task_s3d_feat_dir, 
            'step_cls_{}_split.pickle'.format(split)), 'rb') as f:
            split_samples = pickle.load(f)
            
        # obtain the samples
        self.samples = []
        for video_annot_file in split_samples:
            anno_csv = pd.read_csv(
                video_annot_file, 
                names=['step class id', 'start in seconds', 'end in seconds'])

            for step_idx in range(len(anno_csv['step class id'])):
                self.samples.append((video_annot_file, step_idx))
            
    def __len__(self):
        return len(self.samples)
        
    @staticmethod  
    def custom_collate(batch):
        # https://androidkt.com/create-dataloader-with-collate_fn-for-variable-length-input-in-pytorch/
        segments_list, label_list = [], []
        max_length = 0
        sequence_dim = None
        for (_segments, _label) in batch:
            if not sequence_dim:
                sequence_dim = _segments.shape[-1]
            max_length = max(max_length, len(_segments))
            segments_list.append(_segments)
            label_list.append(_label)
             
        mask_list = []
        for i in range(len(segments_list)):
            if len(segments_list[i]) < max_length:
                pad_length = max_length - len(segments_list[i])
                mask_list.append(torch.tensor([0]*len(segments_list[i])+[1]*pad_length))
                segments_list[i] = torch.cat(
                    [segments_list[i], torch.zeros((pad_length, sequence_dim))], dim=0)
            else:
                mask_list.append(torch.tensor([0]*max_length))
        return torch.stack(segments_list), torch.LongTensor(label_list), torch.stack(mask_list).bool()

    def __getitem__(self, index):
        video_annot_file, step_idx = self.samples[index]
        anno_csv = pd.read_csv(video_annot_file, names=['step class id', 'start in seconds', 'end in seconds'])
        steps_of_this_task = self.task_meta[video_annot_file.split('/')[-1].split('_', 1)[0]]['ordered_steps']

        step_names_this_video = [steps_of_this_task[step_id-1] for step_id in anno_csv['step class id'].tolist()]
        step_ids_this_video = [self.cls_sid2iid[step_name] for step_name in step_names_this_video]

        step_starttimes_this_video = anno_csv['start in seconds'].tolist()
        step_endtimes_this_video = anno_csv['end in seconds'].tolist()

        video_sid = video_annot_file.split('/')[-1].split('_', 1)[1].split('.csv')[0]
        video_feats = np.load(
            os.path.join(self.args.cross_task_s3d_feat_dir, video_sid, 'video.npy'))
        video_segments_time = np.load(
            os.path.join(self.args.cross_task_s3d_feat_dir, video_sid, 'segment_time.npy'))
        segment_elapse = video_segments_time[0][1] - video_segments_time[0][0]
       
        sample_label = step_ids_this_video[step_idx]
        step_starttime = step_starttimes_this_video[step_idx]
        step_endtime = step_endtimes_this_video[step_idx]
        
        step_feats = []
        for j in range(len(video_feats)):
            segment_starttime, segment_endtime = video_segments_time[j][0], video_segments_time[j][-1] + segment_elapse
            if segment_endtime > step_starttime:
                if segment_starttime < step_endtime:
                    step_feats.append(video_feats[j])
        step_feats = np.array(step_feats)  # (num_segments, 3, 512)
        step_feats = np.mean(step_feats, axis=1)  # (num_segments, 512)
        
        return torch.FloatTensor(step_feats), sample_label
        
        
        
class CrossTask_Step_Forecast(Dataset):
    def __init__(self, args, logger, split='train'):
        
        self.args = args
        self.logger = logger
        
        with open(os.path.join(
            args.cross_task_s3d_feat_dir, 
            'step_forecasting_task2step_meta.pickle'), 'rb') as f:
            self.task_meta = pickle.load(f)
        with open(os.path.join(
            args.cross_task_s3d_feat_dir, 
            'step_forecasting_set_of_steps.pickle'), 'rb') as f:
            self.steps = pickle.load(f)
            
        self.cls_sid2iid = dict()
        self.cls_iid2sid = dict()
        for step in self.steps:
            self.cls_sid2iid[step] = len(self.cls_sid2iid)
            self.cls_iid2sid[self.cls_sid2iid[step]] = step
        
        with open(os.path.join(
            args.cross_task_s3d_feat_dir, 
            'step_forecasting_{}_split.pickle'.format(split)), 'rb') as f:
            split_samples = pickle.load(f)
            
        # obtain the samples
        self.samples = []
        for video_annot_file in split_samples:
            anno_csv = pd.read_csv(
                video_annot_file, 
                names=['step class id', 'start in seconds', 'end in seconds'])

            for step_idx in range(len(anno_csv['step class id']) - args.coin_step_forecasting_history):
                self.samples.append((video_annot_file, step_idx + args.coin_step_forecasting_history))
            
    def __len__(self):
        return len(self.samples)
        
    @staticmethod  
    def custom_collate(batch):
        # https://androidkt.com/create-dataloader-with-collate_fn-for-variable-length-input-in-pytorch/
        segments_list, label_list = [], []
        max_length = 0
        sequence_dim = None
        for (_segments, _label) in batch:
            if not sequence_dim:
                sequence_dim = _segments.shape[-1]
            max_length = max(max_length, len(_segments))
            segments_list.append(_segments)
            label_list.append(_label)
             
        mask_list = []
        for i in range(len(segments_list)):
            if len(segments_list[i]) < max_length:
                pad_length = max_length - len(segments_list[i])
                mask_list.append(torch.tensor([0]*len(segments_list[i])+[1]*pad_length))
                segments_list[i] = torch.cat(
                    [segments_list[i], torch.zeros((pad_length, sequence_dim))], dim=0)
            else:
                mask_list.append(torch.tensor([0]*max_length))
        return torch.stack(segments_list), torch.LongTensor(label_list), torch.stack(mask_list).bool()

    
    def __getitem__(self, index):
        video_annot_file, step_idx = self.samples[index]
        anno_csv = pd.read_csv(
            video_annot_file,
            names=['step class id', 'start in seconds', 'end in seconds'])
        steps_of_this_task = self.task_meta[video_annot_file.split('/')[-1].split('_', 1)[0]]['ordered_steps']

        step_names_this_video = [steps_of_this_task[step_id-1] for step_id in anno_csv['step class id'].tolist()]
        step_ids_this_video = [self.cls_sid2iid[step_name] for step_name in step_names_this_video]

        step_starttimes_this_video = anno_csv['start in seconds'].tolist()
        step_endtimes_this_video = anno_csv['end in seconds'].tolist()

        video_sid = video_annot_file.split('/')[-1].split('_', 1)[1].split('.csv')[0]
        video_feats = np.load(
            os.path.join(self.args.cross_task_s3d_feat_dir, video_sid, 'video.npy'))
        video_segments_time = np.load(
            os.path.join(self.args.cross_task_s3d_feat_dir, video_sid, 'segment_time.npy'))
        segment_elapse = video_segments_time[0][1] - video_segments_time[0][0]
       
        sample_label = step_ids_this_video[step_idx]
        
        sample_feats = []
        i = 0
        j = 0
        while i < step_idx and j < len(video_feats):
            step_starttime, step_endtime = step_starttimes_this_video[step_idx], step_endtimes_this_video[step_idx]
            segment_starttime, segment_endtime = video_segments_time[j][0], video_segments_time[j][-1] + segment_elapse
            if segment_endtime <= step_starttime:
                j += 1
            else:
                if segment_starttime >= step_endtime:
                    i += 1
                else:
                    # print('keep j: {}'.format(j))
                    sample_feats.append(video_feats[j])
                    j += 1
        # print(' ')
        
        sample_feats = np.array(sample_feats)  # (num_segments, 3, 512)
        sample_feats = np.mean(sample_feats, axis=1)  # (num_segments, 512)
        
        return torch.FloatTensor(sample_feats), sample_label
    
    
    
