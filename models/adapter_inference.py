import torch.nn as nn


class Adapter(nn.Module):
    def __init__(self, args, logger):
        super(Adapter, self).__init__()
        
        self.args, self.logger = args, logger
        
        assert self.args.adapter_refined_feat_dim == self.args.s3d_hidden_dim
        
        adapter_layers = []
        adapter_layers.append(
            nn.Linear(args.s3d_hidden_dim, args.bottleneck_dim))
        adapter_layers.append(
            nn.ReLU(inplace=True))
        adapter_layers.append(
            nn.Linear(args.bottleneck_dim, args.adapter_refined_feat_dim))
        self.adapter = nn.Sequential(*adapter_layers)
        
        
    def forward(self, segment_feat, prediction=False):
        """
        - segment_feat: (B, 512)
        """
        
        refined_segment_feat = self.adapter(segment_feat)
        
        return refined_segment_feat
        