"""
Various positional encodings. Code is written based on the following refs:
[1] https://github.com/facebookresearch/detr/blob/a54b77800eb8e64e3ad0d8237789fcbf2f8350c5/models/position_encoding.py
"""
import os
import math
import numpy as np
import torch
from torch import nn
from torch.nn.parameter import Parameter
import pdb


class PositionEmbeddingAbsoluteLearned_1D(nn.Module):
    """
    Absolute pos embedding, learned (for 1D sequence).
    """
    def __init__(self, max_num_positions=50, num_pos_feats=256):
        super().__init__()
        self.x_embed = nn.Embedding(max_num_positions, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.x_embed.weight)

    def forward(self, x, batch_size=1):
        # x: in shape [..., num_pos]. E.g., in shape [B, N, J, T]
        pos = self.x_embed(x) # [B, N, J, T, dim]
        return pos
    
    
class PositionEmbeddingAbsoluteLearned_2D(nn.Module):
    """
    Absolute pos embedding, learned (for 2D sequence) and accepts ALL possible image loacations as input.
    """
    def __init__(self, max_num_positions_r=50, max_num_positions_c=50, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(max_num_positions_r, num_pos_feats)
        self.col_embed = nn.Embedding(max_num_positions_c, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, h, w, device):
        # x: in shape [..., h, w]. 
        # h, w = x.shape[-2] # this is assume x is an image
        i = torch.arange(w, device=device)
        j = torch.arange(h, device=device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1)
        return pos
    
    
class PositionEmbeddingAbsoluteLearned_2D_only(nn.Module):
    """
    Absolute pos embedding, learned (for 2D sequence) and ONLY accepts image loacations that have joints as input .
    """
    def __init__(self, max_num_positions_r=50, max_num_positions_c=50, num_pos_feats=256):
        super().__init__()
        self.max_num_positions_r = max_num_positions_r
        self.max_num_positions_c = max_num_positions_c
        
        self.row_embed = nn.Embedding(max_num_positions_r, num_pos_feats) # for height
        self.col_embed = nn.Embedding(max_num_positions_c, num_pos_feats) # for width
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, i, j):
        assert torch.max(i) < self.max_num_positions_c
        assert torch.max(j) < self.max_num_positions_r
        
        x_emb = self.col_embed(i)  # for a location along the 'width' axis
        y_emb = self.row_embed(j)  # for a location along the 'height' axis
        
        pos = torch.cat([x_emb, y_emb], dim=-1)
        return pos
    

class PositionEmbeddingFixedSine_1D(nn.Module):
    """
    Sinusoidal positional encoding (for 1D sequence) from Section 3.5 in https://arxiv.org/pdf/1706.03762.pdf)
    """
    def __init__(self, num_pos_feats=128, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, num_positions, device, mask=None, batch_size=1):
        """
        - mask: Shape (B, N)
                False means that nth position has non-padding values, 
                True measn that position has just the padding value.
                Mask can be used to allow for sequences in different length in one batch.
        - batch_size: Int
                If mask is None, then batch_size determines the batch dim in output.
                Mask is None means sequences in this batch share the same length.
        """
        if mask is None:
            mask = torch.zeros((batch_size, num_positions), dtype=torch.uint8, device=device)
        
        not_mask = ~mask
        x_embed = not_mask.cumsum(1, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            x_embed = x_embed / (x_embed[:, -1:] + eps) * self.scale  # the normalized "pos" vector, [B, N]
            # normalize so that x values are between 0 and 1, and then scale by 2 * pi 

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)  # denominator part, [d_model, ]

        pos_x = x_embed[:, :, None] / dim_t  # pos_x: [B, N, d_model]
        
        pos = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        
        return pos



class PositionEmbeddingFixedSine_2D(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is All Your Need paper
    (Section 3.5 in https://arxiv.org/pdf/1706.03762.pdf), 
    generalized to work on images.
    See https://github.com/facebookresearch/detr/issues/25#issuecomment-636452464 for code explanation
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        if num_pos_feats % 2 == 0:
            self.num_pos_feats = num_pos_feats 
            self.combine_xy_fn = 'cat'
        else:
            self.num_pos_feats = num_pos_feats*2
            self.combine_xy_fn = 'sum'

        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        

    def forward(self, num_positions_h, num_positions_w, device, mask=None, batch_size=1):
        """
        - mask: Shape (B, H/X, W/Y)
                False means that image position has pixel values, 
                True measn that image position has just the padding value.
                Mask can be used to allow for different image sizes in one batch.
        - batch_size: Int
                If mask is None, then batch_size determines the batch dim in output.
                Mask is None means images in this batch share the same size.
        """
        if mask is None:
            mask = torch.zeros((batch_size, num_positions_h, num_positions_w)).type(torch.ByteTensor).to(device)
        
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
            # normalize so that x/y values are between 0 and 1, and then scale by 2 * pi 
        
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        if self.combine_xy_fn == 'cat':
            pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        else:
            pos = (pos_y + pos_x).permute(0, 3, 1, 2)
        return pos



class GaussianFourierFeatureTransform(torch.nn.Module):
    """
    An implementation of Gaussian Fourier feature mapping.
    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html
    Given an input of size [batches, num_input_channels, width, height],
     returns a tensor of size [batches, mapping_size*2, width, height].
    """

    def __init__(self, num_input_channels, mapping_size=256, scale=10):
        super().__init__()

        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size
        self._B = torch.randn((num_input_channels, mapping_size)) * scale

    def forward(self, x):
        assert x.dim() == 4, 'Expected 4D input (got {}D input)'.format(x.dim())

        batches, channels, width, height = x.shape

        assert channels == self._num_input_channels,\
            "Expected input to have {} channels (got {} channels)".format(self._num_input_channels, channels)

        # Make shape compatible for matmul with _B.
        # From [B, C, W, H] to [(B*W*H), C].
        x = x.permute(0, 2, 3, 1).reshape(batches * width * height, channels)
        x = x @ self._B.to(x.device)

        # From [(B*W*H), C] to [B, W, H, C]
        x = x.view(batches, width, height, self._mapping_size)
        # From [B, W, H, C] to [B, C, W, H]
        x = x.permute(0, 3, 1, 2)

        x = 2 * np.pi * x  # [1, 128, 512, 512]
        return torch.cat([torch.sin(x), torch.cos(x)], dim=1)  # [1, 256, 512, 512]



class LearnedFourierFeatureTransform(torch.nn.Module):
    """
    Modified from GaussianFourierFeatureTransform class.
    Given an input of size [batches, num_input_channels, width, height],
     returns a tensor of size [batches, mapping_size*2, width, height].
    """

    def __init__(self, num_input_channels, mapping_size=256):
        super().__init__()

        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size
        self._B = nn.Parameter(torch.zeros(size=(num_input_channels, mapping_size)))
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self._B.data)
        # nn.init.xavier_normal_(self._B.data, gain=math.sqrt(2)) 

    def forward(self, x):
        assert x.dim() == 4, 'Expected 4D input (got {}D input)'.format(x.dim())

        batches, channels, width, height = x.shape

        assert channels == self._num_input_channels,\
            "Expected input to have {} channels (got {} channels)".format(self._num_input_channels, channels)

        # Make shape compatible for matmul with _B.
        # From [B, C, W, H] to [(B*W*H), C].
        x = x.permute(0, 2, 3, 1).reshape(batches * width * height, channels)
        x = x @ self._B.to(x.device)

        # From [(B*W*H), C] to [B, W, H, C]
        x = x.view(batches, width, height, self._mapping_size)
        # From [B, W, H, C] to [B, C, W, H]
        x = x.permute(0, 3, 1, 2)

        x = 2 * np.pi * x  # [1, 128, 512, 512]
        return torch.cat([torch.sin(x), torch.cos(x)], dim=1)  # [1, 256, 512, 512]
