from typing import Optional, Union, Tuple

import torch
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax

from utils import init_weights


class GraphAttentionReturenW(MessagePassing):

    def __init__(self,
                 hidden_dim: int,
                 num_heads: int,
                 dropout: float,
                 has_edge_attr: bool,
                 if_self_attention: bool,
                 **kwargs) -> None:
        super(GraphAttentionReturenW, self).__init__(aggr='add', node_dim=0, **kwargs)
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.has_edge_attr = has_edge_attr
        self.if_self_attention = if_self_attention

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
        self.attn_drop = nn.Dropout(dropout)

        if if_self_attention:
            self.mha_prenorm_src = nn.LayerNorm(hidden_dim)
        else:
            self.mha_prenorm_src = nn.LayerNorm(hidden_dim)
            self.mha_prenorm_dst = nn.LayerNorm(hidden_dim)

        if has_edge_attr:
            self.mha_prenorm_edge = nn.LayerNorm(hidden_dim)

        self.ffn_prenorm = nn.LayerNorm(hidden_dim)

        # ---- 用于缓存 attention 的内部状态（每次 forward 会覆盖）----
        self._return_attention: bool = False
        self._cached_attention: Optional[torch.Tensor] = None  # [E, H]

        self.apply(init_weights)

    def forward(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        detach_attention: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            return_attention: True 时返回 (out, (edge_index, alpha))，其中 alpha shape=[E, H]
            detach_attention: True 时 alpha.detach()，更省显存/更适合日志与可视化
        """
        if self.has_edge_attr and edge_attr is None:
            raise ValueError("has_edge_attr=True 但 forward 未传入 edge_attr")

        if self.if_self_attention:
            x_src = x_dst = self.mha_prenorm_src(x)
        else:
            x_src, x_dst = x
            x_src = self.mha_prenorm_src(x_src)
            x_dst = self.mha_prenorm_dst(x_dst)

        if self.has_edge_attr:
            edge_attr = self.mha_prenorm_edge(edge_attr)

        # 开启/关闭本次 forward 的 attention 收集
        self._return_attention = return_attention
        self._cached_attention = None

        x_dst = x_dst + self._mha_layer(x_src, x_dst, edge_index, edge_attr)
        x_dst = x_dst + self._ffn_layer(self.ffn_prenorm(x_dst))

        if not return_attention:
            return x_dst

        alpha = self._cached_attention
        # 理论上不应为 None；保险起见做个兜底
        if alpha is None:
            alpha = torch.empty((edge_index.size(1), self.num_heads), device=edge_index.device)

        if detach_attention:
            alpha = alpha.detach()

        # 清理状态，避免外部误用上一次 forward 的缓存
        self._return_attention = False
        self._cached_attention = None

        return x_dst, alpha

    def message(
        self,
        x_dst_i: torch.Tensor,
        x_src_j: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
    ) -> torch.Tensor:
        query_i = self.q(x_dst_i).view(-1, self.num_heads, self.head_dim)
        key_j = self.k(x_src_j).view(-1, self.num_heads, self.head_dim)
        value_j = self.v(x_src_j).view(-1, self.num_heads, self.head_dim)

        if self.has_edge_attr:
            key_j = key_j + self.edge_k(edge_attr).view(-1, self.num_heads, self.head_dim)
            value_j = value_j + self.edge_v(edge_attr).view(-1, self.num_heads, self.head_dim)

        scale = self.head_dim ** 0.5
        weight = (query_i * key_j).sum(dim=-1) / scale  # [E, H]
        weight = softmax(weight, index, ptr)            # [E, H]  (softmax 后)

        # 缓存 attention（softmax 后、dropout 前）用于返回
        if self._return_attention:
            self._cached_attention = weight

        weight = self.attn_drop(weight)                # dropout 后实际用于加权
        # 如果你想返回 dropout 后的 attention，把上面缓存那行挪到 dropout 之后即可：
        # if self._return_attention:
        #     self._cached_attention = weight

        return (value_j * weight.unsqueeze(-1)).view(-1, self.num_heads * self.head_dim)

    def _mha_layer(
        self,
        x_src: torch.Tensor,
        x_dst: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.propagate(edge_index=edge_index, edge_attr=edge_attr, x_dst=x_dst, x_src=x_src)

    def _ffn_layer(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)
