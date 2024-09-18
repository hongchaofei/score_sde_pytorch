from __future__ import annotations

import math
import torch
from torch import Tensor
from torch import nn, pi, from_numpy
from torch.nn import Module, ModuleList
import torch.nn.functional as F

import einx
from einops import einsum, reduce, rearrange, repeat
from einops.layers.torch import Rearrange


class GaussianFourierProjection(nn.Module):
  """Gaussian Fourier embeddings for noise levels."""

  def __init__(self, embedding_size=256, scale=1.0):
    super().__init__()
    self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)

  def forward(self, x):
    x_proj = x[:, None] * self.W[None, :] * 2 * torch.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class MLPnet(Module):

    def __init__(self, in_channel=2, unit_channel=16, layer_num=2):
        super().__init__()
        self.layer_num = layer_num
        self.up_layers = ModuleList()
        self.down_layers = []
        self.in_layer = nn.Linear(in_channel, unit_channel)
        self.out_layer = nn.Linear(unit_channel, in_channel)
        self.time_layers = ModuleList()
        in_c = unit_channel
        out_c = 2*in_c
        pos_emb = GaussianFourierProjection(unit_channel)
        time_dim = 4*unit_channel

        for ii in range(layer_num):
            self.up_layers.append(nn.Linear(in_c, out_c))
            self.down_layers.append(nn.Linear(out_c, in_c))
            self.time_layers.append(nn.Sequential(
                    pos_emb,
                    nn.Linear(unit_channel, time_dim),
                    nn.GELU(),
                    nn.Linear(time_dim, in_c),))
            in_c = out_c
            out_c = 2 * in_c
        self.time_layers.append(nn.Sequential(
                    pos_emb,
                    nn.Linear(unit_channel, time_dim),
                    nn.GELU(),
                    nn.Linear(time_dim, in_c),))
        self.mid_layer = nn.Linear(in_c, in_c)
        self.down_layers.reverse()
        self.down_layers = ModuleList(self.down_layers)



    def forward(self, x, times):
        times = times.unsqueeze(-1)
        x = F.silu(self.in_layer(x))
        hs = []
        for i, layer in enumerate(self.up_layers):
            x = F.relu(layer(F.layer_norm(x+self.time_layers[i](times), [x.shape[1],])))
            hs.append(x)
        hs.reverse()
        x = F.relu(self.mid_layer(F.layer_norm(x+self.time_layers[-1](times), [x.shape[1],])))
        for i, (h, layer) in enumerate(zip(hs, self.down_layers)):
            x = F.relu(layer(F.layer_norm(x+h+self.time_layers[-1-i](times), [x.shape[1],])))

        out = self.out_layer(x)
        return out





