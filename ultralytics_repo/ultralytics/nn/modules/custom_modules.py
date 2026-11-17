# ====================================================================
# MODUL ARSITEKTUR KUSTOM SKRIPSI - REFAEL JUARI SIAGIAN (231402055)
# Implementasi Hibrid ConvNeXt V2 & CBAM terintegrasi RT-DETR
# ====================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "CBAM",
    "ChannelAttention",
    "SpatialAttention",
    "GRN",
    "ConvNeXtV2_Block",
    "ConvNeXtV2_Stem",
    "ConvNeXtV2_Stage",
]

# --------------------------------------------------------------------
# 1. MODUL CBAM (Convolutional Block Attention Module)
# --------------------------------------------------------------------

class ChannelAttention(nn.Module):
    """Channel Attention Module CBAM (Woo et al., 2018)."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        reduced_ch = max(channels // reduction, 8)
        self.fc1 = nn.Conv2d(channels, reduced_ch, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(reduced_ch, channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = F.adaptive_avg_pool2d(x, (1, 1))
        max_pool = F.adaptive_max_pool2d(x, (1, 1))
        avg_out = self.fc2(self.relu(self.fc1(avg_pool)))
        max_out = self.fc2(self.relu(self.fc1(max_pool)))
        scale = self.sigmoid(avg_out + max_out)
        return x * scale


class SpatialAttention(nn.Module):
    """Spatial Attention Module CBAM (Woo et al., 2018)."""
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), "Kernel size harus 3 atau 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        scale = self.sigmoid(self.conv(concat))
        return x * scale


class CBAM(nn.Module):
    """CBAM terpadu (Channel Attention diikuti Spatial Attention)."""
    def __init__(self, c1, c2=None, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_att = ChannelAttention(c1, reduction)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


# --------------------------------------------------------------------
# 2. MODUL CONVNEXT V2 & GLOBAL RESPONSE NORMALIZATION (GRN)
# --------------------------------------------------------------------

class GRN(nn.Module):
    """Global Response Normalization (GRN) (Woo et al., 2023 - ConvNeXt V2).
    Mencegah feature collapse dan memperkuat pemodelan representasi spasial."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.eps = eps

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + self.eps)
        return self.gamma * (x * Nx) + self.beta + x


class ConvNeXtV2_Block(nn.Module):
    """ConvNeXt V2 Block: 7x7 Depthwise Conv -> 1x1 Conv -> GELU -> GRN -> 1x1 Conv + Residual."""
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        return residual + x


class ConvNeXtV2_Stem(nn.Module):
    """Stem ConvNeXt V2: Patchify 4x4 konvolusi dengan stride 4."""
    def __init__(self, c1=3, c2=64):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, kernel_size=4, stride=4)
        self.norm = nn.BatchNorm2d(c2)

    def forward(self, x):
        return self.norm(self.conv(x))


class ConvNeXtV2_Stage(nn.Module):
    """Stage ConvNeXt V2: Opsional downsampling 2x2 stride 2 diikuti kumpulan ConvNeXtV2 Blocks."""
    def __init__(self, c1, c2, depth=2, downsample=True):
        super().__init__()
        layers = []
        if downsample:
            layers.append(nn.Conv2d(c1, c2, kernel_size=2, stride=2))
            layers.append(nn.BatchNorm2d(c2))
            curr_dim = c2
        else:
            if c1 != c2:
                layers.append(nn.Conv2d(c1, c2, kernel_size=1))
                layers.append(nn.BatchNorm2d(c2))
                curr_dim = c2
            else:
                curr_dim = c1

        for _ in range(depth):
            layers.append(ConvNeXtV2_Block(curr_dim))
        self.stage = nn.Sequential(*layers)

    def forward(self, x):
        return self.stage(x)

