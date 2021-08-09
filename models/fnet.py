import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
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

# class MaskedFNetBlock(nn.Module):
#     def __init__(self, forc_len):
#         self.forc_len = forc_len
#         super().__init__()

#     def forward(self, x):
#         n = x.shape[1] - self.forc_len
#         x = torch.fft.fftn(torch.fft.fft(x, dim=-1), n, dim=-2).real
#         x = F.pad(x, (0, 0, 0, self.forc_len)) #dims correct?
#         return x

class FNetEncoder(nn.Module):
    def __init__(self, dim, depth, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(2 * depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, FNetBlock()),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x, attn_mask=None, x_mask=None, cross_mask=None):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x, attn_mask

class FNetDecoder(nn.Module):
    def __init__(self, dim, depth, mlp_dim, forc_len, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.forc_len = forc_len
        for _ in range(2 * depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, FNetBlock()),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x