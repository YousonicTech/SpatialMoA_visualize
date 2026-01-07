from typing import Optional, Union, Tuple

import torch
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax

from utils import init_weights
import math

class StructureKeyDropGraphAttention(MessagePassing):

    def __init__(self,
                 hidden_dim: int,
                 num_heads: int,
                 dropout: float,
                 has_edge_attr: bool,
                 if_self_attention: bool,
                 mask_ratio:float,
                 num_r_bins: int,
                num_theta_bins: int,
                 **kwargs) -> None:
        super(StructureKeyDropGraphAttention, self).__init__(aggr='add', node_dim=0, **kwargs)
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.has_edge_attr = has_edge_attr
        self.if_self_attention = if_self_attention
        self.mask_ratio = mask_ratio

        self.num_r_bins= num_r_bins
        self.num_theta_bins = num_theta_bins

        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        if has_edge_attr:
            self.edge_k = nn.Linear(hidden_dim, hidden_dim)
            self.edge_v = nn.Linear(hidden_dim, hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.attn_drop = nn.Dropout(0.01)
        if if_self_attention:
            self.mha_prenorm_src = nn.LayerNorm(hidden_dim)
        else:
            self.mha_prenorm_src = nn.LayerNorm(hidden_dim)
            self.mha_prenorm_dst = nn.LayerNorm(hidden_dim)
        if has_edge_attr:
            self.mha_prenorm_edge = nn.LayerNorm(hidden_dim)
        self.ffn_prenorm = nn.LayerNorm(hidden_dim)
        self.apply(init_weights)

    def forward(self,
                x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None,
                att_mask= None,
                raw_attr = None) -> torch.Tensor:
        if self.if_self_attention:
            x_src = x_dst = self.mha_prenorm_src(x)
        else:
            x_src, x_dst = x
            x_src = self.mha_prenorm_src(x_src)
            x_dst = self.mha_prenorm_dst(x_dst)
        if self.has_edge_attr:
            edge_attr = self.mha_prenorm_edge(edge_attr)
        x_dst = x_dst + self._mha_layer(x_src, x_dst, edge_index, edge_attr,att_mask,raw_attr)
        x_dst = x_dst + self._ffn_layer(self.ffn_prenorm(x_dst))
        return x_dst

    def message(self,
                x_dst_i: torch.Tensor,
                x_src_j: torch.Tensor,
                edge_attr: Optional[torch.Tensor],
                index: torch.Tensor,
                ptr: Optional[torch.Tensor],
                att_mask,raw_attr) -> torch.Tensor:
        query_i = self.q(x_dst_i).view(-1, self.num_heads, self.head_dim)
        key_j = self.k(x_src_j).view(-1, self.num_heads, self.head_dim)
        value_j = self.v(x_src_j).view(-1, self.num_heads, self.head_dim)
        if self.has_edge_attr:
            key_j = key_j + self.edge_k(edge_attr).view(-1, self.num_heads, self.head_dim)
            value_j = value_j + self.edge_v(edge_attr).view(-1, self.num_heads, self.head_dim)
        scale = self.head_dim ** 0.5
        weight = (query_i * key_j).sum(dim=-1) / scale



        if self.training:
            # m_r = torch.ones_like(weight) * self.mask_ratio
            # weight = weight + torch.bernoulli(m_r) * (-1e-12)
            edge_mask = self._structured_keydrop_mask(
                weight=weight,
                raw_attr=raw_attr
            )
            weight = weight + edge_mask * (-1e-12)


        weight = softmax(weight, index, ptr)
        weight = self.attn_drop(weight)
        return (value_j * weight.unsqueeze(-1)).view(-1, self.num_heads*self.head_dim)

    def _mha_layer(self,
                   x_src: torch.Tensor,
                   x_dst: torch.Tensor,
                   edge_index: torch.Tensor,
                   edge_attr: Optional[torch.Tensor]=None,
                   att_mask=None,
                   raw_attr=None) -> torch.Tensor:
        return self.propagate(edge_index=edge_index, edge_attr=edge_attr, x_dst=x_dst, x_src=x_src, att_mask=att_mask,raw_attr = raw_attr)

    def _ffn_layer(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)

    def _structured_keydrop_mask(self,weight,raw_attr):
        device = weight.device
        dtype = weight.dtype

        # 1. 取出半径和角度
        r = raw_attr[:, 0]  # [E]
        theta = raw_attr[:, 1]  # [E]

        # 2. 把角度统一到 [0, 2π)
        theta = (theta + 2 * math.pi) % (2 * math.pi)

        # 3. 构造半径和角度的 bin 边界
        r_min, r_max = r.min(), r.max()
        # +1e-6 为了让最大值落在最后一个 bin
        r_edges = torch.linspace(r_min, r_max + 1e-6,
                                 steps=self.num_r_bins + 1,
                                 device=device,
                                 dtype=dtype)
        theta_edges = torch.linspace(0.0, 2 * math.pi + 1e-6,
                                     steps=self.num_theta_bins + 1,
                                     device=device,
                                     dtype=dtype)

        # 4. 每条边量化成 (r_bin, theta_bin)
        # bucketize 返回 [1, ..., K]，减 1 变成 [0, ..., K-1]
        r_bin = torch.bucketize(r, r_edges) - 1
        theta_bin = torch.bucketize(theta, theta_edges) - 1

        r_bin = r_bin.clamp(min=0, max=self.num_r_bins - 1)
        theta_bin = theta_bin.clamp(min=0, max=self.num_theta_bins - 1)

        # 5. 在 (r_bin, theta_bin, head) 网格上采样 Bernoulli mask
        E, H = weight.shape
        p = torch.full(
            (self.num_r_bins, self.num_theta_bins, H),
            fill_value=self.mask_ratio,
            device=device,
            dtype=dtype
        )
        grid_mask = torch.bernoulli(p)  # [R, T, H]

        # 6. 展开回每条 edge：[E, H]
        edge_mask = grid_mask[r_bin, theta_bin]  # 利用高级索引

        return edge_mask
