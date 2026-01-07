import torch
import torch.nn as nn

class MyModule(nn.Module):
    def __init__(self, mask_ratio=0.2):
        super().__init__()
        self.mask_ratio = mask_ratio

    def forward(self, weight, training=True):
        if training:
            m_r = torch.ones_like(weight) * self.mask_ratio
            weight = weight + torch.bernoulli(m_r) * (-1e-12)
        return weight

# 测试
mod = MyModule(mask_ratio=0.1)
w = torch.randn(3,4)  # 随机一个 weight 张量
print("原始 weight：\n", w)

w2 = mod.forward(w, training=True)
print("training=True 时修改后的 weight：\n", w2)

w3 = mod.forward(w, training=False)
print("training=False 时（未修改）weight：\n", w3)
