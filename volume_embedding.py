import torch
import torch.nn as nn
from einops import rearrange, repeat

def expand_to_batch(tensor, desired_size):
    tile = desired_size // tensor.shape[0]
    return repeat(tensor, 'b ... -> (b tile) ...', tile=tile)

class AbsPositionalEncoding1D(nn.Module):
    def __init__(self, tokens, dim):
        super(AbsPositionalEncoding1D, self).__init__()
        self.abs_pos_enc = nn.Parameter(torch.randn(1,tokens, dim))

    def forward(self, x):
        batch = x.size()[0]
        return x + expand_to_batch(self.abs_pos_enc, desired_size=batch)

class Embeddings3D(nn.Module):
    def __init__(self, input_dim, embed_dim, cube_size, patch_size=16, dropout=0.1):
        super().__init__()
        self.n_patches = int((cube_size[0] * cube_size[1] * cube_size[2]) / (patch_size * patch_size * patch_size))
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_embeddings = nn.Conv3d(in_channels=input_dim, out_channels=embed_dim,
                                          kernel_size=patch_size, stride=patch_size, bias=False)
        self.position_embeddings = AbsPositionalEncoding1D(self.n_patches, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x is a 5D tensor
        """
        x = rearrange(self.patch_embeddings(x), 'b d x y z -> b (x y z) d')
        embeddings = self.dropout(self.position_embeddings(x))
        return embeddings
