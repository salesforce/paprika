import os

import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer 

from models.misc import build_mlp


class Task_Head(nn.Module):
    def __init__(self, args, logger):
        super(Task_Head, self).__init__()
        
        self.args = args
        self.logger = logger
        
        #### embedding layers
        self.cls_embed_layer = nn.Embedding(1, args.model_task_cls_segment_hidden_dim)
        
        if args.model_task_cls_time_pos_embed_type == 'absolute_learned_1D':
            
            from models.position_encoding import PositionEmbeddingAbsoluteLearned_1D
            self.time_embed_layer = PositionEmbeddingAbsoluteLearned_1D(
                args.model_task_cls_max_time_ids_embed, args.model_task_cls_segment_hidden_dim)
        
        else:
            raise ValueError(f"not supported {self.args.model_task_cls_time_pos_embed_type}")
            
        
        if self.args.model_task_cls_head_name == 'downstream_transformer':
            self.long_term_model = TransformerEncoderLayer(
                args.model_task_cls_segment_hidden_dim, 
                args.model_task_cls_tx_nhead, 
                args.model_task_cls_tx_dim_feedforward,
                args.model_task_cls_tx_dropout, 
                args.model_task_cls_tx_activation)
        
            
        self.classifier = build_mlp(
            input_dim=args.model_task_cls_segment_hidden_dim, 
            hidden_dims=[args.model_task_cls_classifier_hidden_dim], 
            output_dim=args.model_task_cls_num_classes)
            
        
    def forward(self, video_feats, video_mask):
        """
        - video_feats: (B, num_segments, 512)
        - video_mask: (B, num_segments)
        """
        B = video_feats.shape[0]
        T = video_feats.shape[1]
        device = self.args.device
        
        # CLS initial embedding
        CLS_id = torch.arange(1, device=device).repeat(B, 1)
        CLS = self.cls_embed_layer(CLS_id)
        
        # time positional encoding
        if self.args.model_task_cls_time_pos_embed_type == 'absolute_learned_1D':
            time_ids = torch.arange(1, T+1, device=device).repeat(B, 1)
            time_seq = self.time_embed_layer(time_ids) 
        elif self.args.model_task_cls_time_pos_embed_type == 'fixed_sinusoidal_1D':
            time_seq = self.time_embed_layer(T, device=device).unsqueeze(0).unsqueeze(0).repeat(B, N, J, 1, 1)
        else:
            raise ValueError(f"not supported {self.args.model_task_cls_time_pos_embed_type}")
      
    
        if self.args.model_task_cls_head_name == 'downstream_transformer':
            tx_updated_sequence = self.long_term_model(
                torch.cat([CLS, video_feats + time_seq], dim=1).transpose(0, 1),
                src_key_padding_mask = torch.cat([torch.zeros((B, 1)).bool().to(device), video_mask], dim=1)
            )
            
            fine_cls = tx_updated_sequence[0]
            pred_logits = self.classifier(fine_cls)
        
        elif self.args.model_task_cls_head_name == 'downstream_mlp':
             
            pred_logits = self.classifier(torch.mean(video_feats + time_seq, dim=1))
            
        else:
            self.logger.info('The model_task_cls_head_name is not implemented!\nFunc: {}\nFile:{}'.format(
                    __name__, __file__))
            os._exit(0)
        
        return pred_logits
    