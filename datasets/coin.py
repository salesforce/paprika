import os
import numpy as np
import json
from collections import defaultdict

import torch
from torch.utils.data import Dataset


class COIN_Task_CLS(Dataset):
    def __init__(self, args, logger, split='train'):
        
        self.args = args
        self.logger = logger
        
        with open(args.coin_annoataion_json, 'r') as f:
            self.coin_json = json.load(f)
        
        self.sample_video_paths = []
        self.cls_sid2iid = defaultdict()
        self.cls_iid2sid = defaultdict()
        self.sample_video_to_task = defaultdict()
        
        for video_sid in self.coin_json['database']:
            cls_sid = self.coin_json['database'][video_sid]['class']
            if cls_sid not in self.cls_sid2iid:
                self.cls_sid2iid[cls_sid] = len(self.cls_sid2iid)
                self.cls_iid2sid[self.cls_sid2iid[cls_sid]] = cls_sid
            self.sample_video_to_task[video_sid] = self.cls_sid2iid[cls_sid]
            
            video_feat_path = os.path.join(args.coin_s3d_feat_dir, video_sid)
            if split == 'train' and self.coin_json['database'][video_sid]['subset'] == 'training':
                self.sample_video_paths.append(video_feat_path)
            elif split == 'test' and self.coin_json['database'][video_sid]['subset'] == 'testing':
                self.sample_video_paths.append(video_feat_path)
                
        assert len(self.cls_iid2sid) == args.model_task_cls_num_classes
        
    def __len__(self):
        return len(self.sample_video_paths)
        
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
        video_sid = sample_video_path.split('/')[-1]
        task_iid = self.sample_video_to_task[video_sid]
        
        
        video_feats = np.load(
            os.path.join(sample_video_path, 'video.npy'))
        video_feats = np.mean(video_feats, axis=1)
        
        return torch.FloatTensor(video_feats), task_iid
        

class COIN_Step_CLS(Dataset):
    def __init__(self, args, logger, split='train'):
        
        self.args = args
        self.logger = logger
        
        with open(args.coin_annoataion_json, 'r') as f:
            self.coin_json = json.load(f)
            
        self.samples = []
        self.cls_sid2iid = defaultdict()
        self.cls_iid2sid = defaultdict()
        for video_sid in self.coin_json['database']:
            video_metadata = self.coin_json['database'][video_sid]
            num_steps_this_video = len(video_metadata['annotation'])
            
            for step_idx in range(num_steps_this_video):
                cls_sid = (video_metadata['annotation'][step_idx]['id'] 
                            + '-' + 
                            video_metadata['annotation'][step_idx]['label'])
                if cls_sid not in self.cls_sid2iid:
                    self.cls_sid2iid[cls_sid] = len(self.cls_sid2iid)
                    self.cls_iid2sid[self.cls_sid2iid[cls_sid]] = cls_sid
                
                if split == 'train' and self.coin_json['database'][video_sid]['subset'] == 'training':
                    self.samples.append((video_sid, step_idx))
                elif split == 'test' and self.coin_json['database'][video_sid]['subset'] == 'testing':
                    self.samples.append((video_sid, step_idx))
        
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
        video_sid, step_idx = self.samples[index]
        video_step_annotations = self.coin_json['database'][video_sid]['annotation']

        cls_sid = (video_step_annotations[step_idx]['id'] 
                   + '-' + 
                   video_step_annotations[step_idx]['label'])
        sample_label = self.cls_sid2iid[cls_sid]
        
        video_feats = np.load(
            os.path.join(self.args.coin_s3d_feat_dir, video_sid, 'video.npy'))
        video_segments_time = np.load(
            os.path.join(self.args.coin_s3d_feat_dir, video_sid, 'segment_time.npy'))
        segment_elapse = video_segments_time[0][1] - video_segments_time[0][0]
        
        step_feats = []
        step_starttime, step_endtime = video_step_annotations[step_idx]['segment'] 
        for j in range(len(video_feats)):
            segment_starttime, segment_endtime = video_segments_time[j][0], video_segments_time[j][-1] + segment_elapse
            if segment_endtime > step_starttime:
                if segment_starttime < step_endtime:
                    step_feats.append(video_feats[j])
        step_feats = np.array(step_feats)  # (num_segments, 3, 512)
        step_feats = np.mean(step_feats, axis=1)  # (num_segments, 512)
        
        return torch.FloatTensor(step_feats), sample_label
        
        
class COIN_Step_Forecast(Dataset):
    def __init__(self, args, logger, split='train'):
        
        self.args = args
        self.logger = logger
        
        with open(args.coin_annoataion_json, 'r') as f:
            self.coin_json = json.load(f)
            
        self.samples = []
        self.cls_sid2iid = defaultdict()
        self.cls_iid2sid = defaultdict()
        for video_sid in self.coin_json['database']:
            video_metadata = self.coin_json['database'][video_sid]
            num_steps_this_video = len(video_metadata['annotation'])
            
            for step_idx in range(num_steps_this_video):
                cls_sid = (video_metadata['annotation'][step_idx]['id'] 
                            + '-' + 
                            video_metadata['annotation'][step_idx]['label'])
                if cls_sid not in self.cls_sid2iid:
                    self.cls_sid2iid[cls_sid] = len(self.cls_sid2iid)
                    self.cls_iid2sid[self.cls_sid2iid[cls_sid]] = cls_sid
                
                if step_idx < num_steps_this_video - args.coin_step_forecasting_history:
                    if split == 'train' and self.coin_json['database'][video_sid]['subset'] == 'training':
                        self.samples.append((video_sid, step_idx + args.coin_step_forecasting_history))
                    elif split == 'test' and self.coin_json['database'][video_sid]['subset'] == 'testing':
                        self.samples.append((video_sid, step_idx + args.coin_step_forecasting_history))
                
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
        video_sid, step_idx = self.samples[index]
        video_step_annotations = self.coin_json['database'][video_sid]['annotation']
        
        cls_sid = (video_step_annotations[step_idx]['id'] 
                   + '-' + 
                   video_step_annotations[step_idx]['label'])
        sample_label = self.cls_sid2iid[cls_sid]
        
        video_feats = np.load(
            os.path.join(self.args.coin_s3d_feat_dir, video_sid, 'video.npy'))
        video_segments_time = np.load(
            os.path.join(self.args.coin_s3d_feat_dir, video_sid, 'segment_time.npy'))
        segment_elapse = video_segments_time[0][1] - video_segments_time[0][0]
        
        sample_feats = []
        i = 0
        j = 0
        while i < step_idx and j < len(video_feats):
            step_starttime, step_endtime = video_step_annotations[i]['segment']
            segment_starttime, segment_endtime = video_segments_time[j][0], video_segments_time[j][-1] + segment_elapse
            if segment_endtime <= step_starttime:
                j += 1
            else:
                if segment_starttime >= step_endtime:
                    i += 1
                else:
                    sample_feats.append(video_feats[j])
                    j += 1
        
        sample_feats = np.array(sample_feats)  # (num_segments, 3, 512)
        sample_feats = np.mean(sample_feats, axis=1)  # (num_segments, 512)
        
        return torch.FloatTensor(sample_feats), sample_label
        