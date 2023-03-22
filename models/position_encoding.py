from torch import nn


class PositionEmbeddingAbsoluteLearned_1D(nn.Module):
    def __init__(self, max_num_positions=50, num_pos_feats=256):
        super().__init__()
        self.x_embed = nn.Embedding(max_num_positions, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.x_embed.weight)

    def forward(self, x, batch_size=1):
        pos = self.x_embed(x)
        return pos
