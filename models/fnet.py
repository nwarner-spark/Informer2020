from nbm_bench.models.fft_ae import FNetAutoencoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def default(val, default_val):
    return val if val is not None else default_val

def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
#     tensor.uniform_(-std, std)
    tensor.normal_(0, std)
    return tensor


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, out_dim = None, dropout = 0.):
        out_dim = default(out_dim, dim)
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FNetBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.fft.fft(torch.fft.fft(x, dim=-1), dim=-2).real
        return x

class MaskedFNetBlock(nn.Module):
    def __init__(self, forc_len):
        self.forc_len = forc_len
        super().__init__()

    def forward(self, x, x_mask=None, cross_mask=None):
        n = x.shape[1] - self.forc_len
        x = torch.fft.fftn(torch.fft.fft(x, dim=-1), n, dim=-2).real
        x = F.pad(x, (0, 0, 0, self.forc_len)) #dims correct?
        return x

class CrossAttention(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, x_mask=None, cross_mask=None):
        x = torch.fft.fft(torch.fft.fft(x, dim=-1), dim=-2).real #just mixes the zeros in with the rest of them
        return x
    

class FNetEncoder(nn.Module):
    def __init__(self, dim, depth, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(2 * depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, FNetBlock()),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x, attn_mask=None):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x, attn_mask

#only for forecasting :) (causal masking off the table)
class FNetDecoder(nn.Module):
    def __init__(self, dim, depth, mlp_dim, forc_len, seq_len, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.forc_len = forc_len
        self.seq_len = seq_len
        self.proj = nn.Parameter(init_(torch.zeros(seq_len, forc_len)))
        for _ in range(2 * depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, FNetBlock()),
#                 PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)),
                PreNorm(dim, FNetBlock()),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))   #operating on forc_len does too much complexity damage
            ]))
    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for attn, cattn, cff in self.layers:
            x = attn(x) + x #decoder queries 'attend' to decoder keys
#             x = ff(x) + x
            cross = cattn(cross)
            forc = torch.einsum('bnd,nf->bfd', cross, self.proj)
#             perm = forc.permute(0, 2, 1)  #applying linear layers to seq_len dimension balloons params, ruins complexity?
            x = cff(x + F.pad(forc, (0, 0, self.seq_len, 0)))
            
        return x

    
# lower performance decoder with constant n_params    
# class FNetDecoder(nn.Module):
#     def __init__(self, dim, depth, mlp_dim, forc_len, seq_len, dropout = 0.):
#         super().__init__()
#         self.layers = nn.ModuleList([])
#         self.forc_len = forc_len
#         self.seq_len = seq_len
#         self.proj = nn.Parameter(init_(torch.zeros(seq_len, mlp_dim)), requires_grad=False)
#         self.reproj = nn.Parameter(init_(torch.zeros(mlp_dim, forc_len)), requires_grad=False)
#         for _ in range(2 * depth):
#             self.layers.append(nn.ModuleList([
#                 PreNorm(dim, FNetBlock()),
#                 PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)),
#                 PreNorm(dim, FNetBlock()),
#                 PreNorm(mlp_dim, FeedForward(mlp_dim, mlp_dim, dropout = dropout))   #operating on forc_len does too much complexity damage
#             ]))
#     def forward(self, x, cross, x_mask=None, cross_mask=None):
#         for attn, ff, cattn, cff in self.layers:
#             x = attn(x) + x #decoder queries 'attend' to decoder keys
#             x = ff(x) + x #maybe save for after?
#             cross = cattn(cross)
#             proj = torch.einsum('bnd,nm->bdm', cross, self.proj)
#             proj = cff(proj)
#             forc = torch.einsum('bdm,mf->bfd', proj, self.reproj)
# #             forc = torch.einsum('bnd,nf->bfd', cross, self.proj)
# #             perm = forc.permute(0, 2, 1)  #applying linear layers to seq_len dimension balloons params, ruins complexity?
#             x = x + F.pad(forc, (0, 0, self.seq_len, 0))
            
#         return x
