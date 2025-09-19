import torch
import torch.nn as nn
import math
class WFE(nn.Module):
    def __init__(self):
        super(WFE, self).__init__()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H % 2 == 0 and W % 2 == 0, "Height and width must be divisible by 2."


        x = x.view(B, C, H // 2, 2, W // 2, 2)
        x = x.permute(0, 1, 2, 4, 3, 5).contiguous()
        sqrt2 = math.sqrt(2)

        LL = (x[..., 0, 0] + x[..., 1, 0] + x[..., 0, 1] + x[..., 1, 1]) / sqrt2
        LH = (x[..., 0, 0] - x[..., 1, 0] + x[..., 0, 1] - x[..., 1, 1]) / sqrt2
        HL = (x[..., 0, 0] + x[..., 1, 0] - x[..., 0, 1] - x[..., 1, 1]) / sqrt2
        HH = (x[..., 0, 0] - x[..., 1, 0] - x[..., 0, 1] + x[..., 1, 1]) / sqrt2


        out = torch.cat([LL, LH, HL, HH], dim=1)
        return out