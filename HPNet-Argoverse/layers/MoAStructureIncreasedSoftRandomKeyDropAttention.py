from typing import Optional, Union, Tuple

import torch
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax

from utils import init_weights
import math

class ExpertGather(nn.Module):
    def __init__(self, E: int, I: int, J: int):
        super().__init__()

        self.E, self.I, self.J = E, I, J
        self.W = nn.Parameter(torch.empty(E, I, J))

        self.reset_parameters()

    def forward(self, X: torch.Tensor, ind: torch.Tensor):
        # output has shape [B,E,K,J]
        T, I = X.shape
        E, K = ind.shape
        index = ind.reshape(E * K)[:, None].expand(-1, I)
        X_gathered = torch.gather(X, dim=0, index=index).reshape(E, K, I)
        Y = torch.matmul(X_gathered, self.W)  # [E, K, I] @ [E, I, J] -> [E, K, J]
        # test_W = self.W.unsqueeze(0)
        #
        # # [B, E, K, I] @ [1, E, I, J] -> [B, E, K, J]
        # test_Y = torch.matmul(X_gathered, test_W)

        return Y

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.J)
        nn.init.uniform_(self.W, -bound, bound)

class MoAStructureIncreasedSoftKeyDropGraphAttention(MessagePassing):

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
        super(MoAStructureIncreasedSoftKeyDropGraphAttention, self).__init__(aggr='add', node_dim=0, **kwargs)
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.has_edge_attr = has_edge_attr
        self.if_self_attention = if_self_attention
        self.mask_ratio = mask_ratio

        self.num_r_bins= num_r_bins
        self.num_theta_bins = num_theta_bins


        self.r = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, num_heads, bias=False),
            torch.nn.Sigmoid()
        )
        self.sparsity = 16

        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)


        self.Q = ExpertGather(num_heads, hidden_dim, hidden_dim)
        self.K = ExpertGather(num_heads, hidden_dim, hidden_dim)
        self.V = ExpertGather(num_heads, hidden_dim, hidden_dim)

        self.cogate_feature_extractor = nn.Sequential(
            nn.Linear(self.num_heads * self.head_dim, self.head_dim),
            nn.LayerNorm(self.head_dim),
            nn.ReLU(),
            nn.Linear(self.head_dim, 1),
            nn.LayerNorm(1),
            nn.ReLU()
        )

        if has_edge_attr:
            self.edge_k = nn.Linear(hidden_dim, hidden_dim)
            self.edge_v = nn.Linear(hidden_dim, hidden_dim)

            self.edge_K = ExpertGather(num_heads, hidden_dim, hidden_dim)
            self.edge_V = ExpertGather(num_heads, hidden_dim, hidden_dim)


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

    def get_topk(self, x: torch.Tensor):
        """
        Selects tokens for the experts
        Input:
            x - inputs shape [B, T, E]
        Output (3-tuple):
            - scores of the tokens for given router [B, E, k]
            - indices of selected tokens in the original sequence [B, E, k]
            - selected number of tokens
        """
        M, h = x.shape

        logits = self.r(x)
        #
        # if self.include_first:
        #     return self.get_topk_includefirst(logits)

        k = int(M // self.sparsity)
        k = min(max(k, 2), M)  # 2 is the minimum number of tokens to select from

        logits_topk = logits.topk(dim=0, k=k)  # [b, k, E]
        topk_I = logits_topk.indices.transpose(1, 0)  # [b, E, k]  #时间维度的索引
        topk_vals = logits_topk.values.transpose(1, 0)  # [b, E, k]

        return topk_vals, topk_I, k


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

            edge_mask = self._structured_keydrop_mask(
                weight=weight,
                raw_attr=raw_attr
            )

            weight = weight + edge_mask * (-1e-12)

        weight = softmax(weight, index, ptr)
        weight = self.attn_drop(weight)
        att_dense = (value_j * weight.unsqueeze(-1)).view(-1, self.num_heads*self.head_dim)

        topk_vals, topk_I, k = self.get_topk(x_dst_i)
        query_i_new = self.Q(x_dst_i, topk_I).view(self.num_heads, -1, self.num_heads, self.head_dim)
        key_j_new = self.K(x_src_j, topk_I).view(self.num_heads, -1, self.num_heads, self.head_dim)
        value_j_new = self.V(x_src_j, topk_I).view(self.num_heads, -1, self.num_heads, self.head_dim)
        if self.has_edge_attr:
            key_j_new = key_j_new + self.edge_K(edge_attr,topk_I).view(self.num_heads, -1, self.num_heads, self.head_dim)
            value_j_new = value_j_new + self.edge_V(edge_attr,topk_I).view(self.num_heads, -1, self.num_heads, self.head_dim)

        new_weight = (query_i_new * key_j_new).sum(dim=-1) / scale
        new_index = index[topk_I]
        new_weights_list = []
        for c in range(self.num_heads):
            idx_h = new_index[c]
            w_h = softmax(new_weight[c], idx_h, ptr=None)
            new_weights_list.append(w_h)
        new_weight = torch.stack(new_weights_list, dim=0)
        new_weight = self.attn_drop(new_weight)

        att = value_j_new * new_weight.unsqueeze(-1)
        att = att * topk_vals.unsqueeze(-1).unsqueeze(-1)  # [H, k, H, D]
        att_flat = att.reshape(-1, k, 128)
        num_edges = value_j.shape[0]  # 134342
        out_flat = torch.zeros((num_edges, 128),
                               device=value_j_new.device,
                               dtype=value_j_new.dtype)
        for c in range(self.num_heads):
            idx = topk_I[c].contiguous()  # [k]
            # out_flat.index_add_(0, idx, att_flat[c])

            with torch.cuda.amp.autocast(enabled=False):
                of = out_flat.float()
                src = att_flat[c].float()
                of.index_add_(0, idx, src)
            out_flat = of.to(out_flat.dtype)
        cogate_feature = self.cogate_feature_extractor(edge_attr)

        return cogate_feature * out_flat +  att_dense

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
        base_ratio = self.mask_ratio
        r_idx = torch.arange(self.num_r_bins, device=device, dtype=dtype)

        r_ratio = torch.full(
            (self.num_r_bins,),
            fill_value=base_ratio,
            device=device,
            dtype=dtype,
        )
        half_bins = self.num_r_bins // 2
        outer_mask = r_idx >= half_bins
        outer_bins = max(self.num_r_bins - half_bins, 1)

        if outer_bins > 0:
            outer_idx = r_idx[outer_mask] - half_bins
            norm_pos = (outer_idx + 1).float() / outer_bins
            delta_max = base_ratio * 0.1
            r_ratio_outer = base_ratio + delta_max * norm_pos
            r_ratio[outer_mask] = r_ratio_outer

        r_ratio = r_ratio.clamp(0.0, 1.0)
        p = r_ratio.view(self.num_r_bins, 1, 1).expand(
            self.num_r_bins,
            self.num_theta_bins,
            H
        )

        # p = torch.full(
        #     (self.num_r_bins, self.num_theta_bins, H),
        #     fill_value=self.mask_ratio,
        #     device=device,
        #     dtype=dtype
        # )
        grid_mask = torch.bernoulli(p)  # [R, T, H]

        # 6. 展开回每条 edge：[E, H]
        edge_mask = grid_mask[r_bin, theta_bin]  # 利用高级索引

        return edge_mask
