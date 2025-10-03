import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Blocks ---
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=3, stride=1, padding=None, bias=False, dilation=1):
        if padding is None:
            padding = ((k - 1) // 2) * dilation  # "same" for odd k
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, stride=stride, padding=padding, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

class DWSeparableConv(nn.Module):
    """
    Depthwise 3x3 (stride/dilation as given) -> BN -> ReLU
    Pointwise 1x1 -> BN -> ReLU
    """
    def __init__(self, in_ch, out_ch, k=3, stride=1, dilation=1, bias=False):
        super().__init__()
        pad = ((k - 1) // 2) * dilation
        self.dw = nn.Conv2d(in_ch, in_ch, k, stride=stride, padding=pad,
                            dilation=dilation, groups=in_ch, bias=bias)
        self.bn_dw = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=bias)
        self.bn_pw = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = F.relu(self.bn_dw(self.dw(x)), inplace=True)
        x = F.relu(self.bn_pw(self.pw(x)), inplace=True)
        return x

# --- Model (≈36k params) ---
class NetCIFAR10_Tiny(nn.Module):
    """
    C1: 3x3 -> 3x3
    C2: DW-Sep + DW-Sep (dilated d=2)
    C3: DW-Sep + DW-Sep
    C4: DW-Sep (stride=2) + DW-Sep(d=2) + DW-Sep(d=3) + DW-Sep + DW-Sep
    GAP -> 1x1 head -> log_softmax
    Receptive field before GAP ≈ 45
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # C1: (B,3,32,32) -> (B,32,32,32)
        self.c1 = nn.Sequential(
            ConvBNReLU(3, 16, k=3, stride=1),      # light standard conv (helps early features)
            ConvBNReLU(16, 24, k=3, stride=1),
        )

        # C2: (B,24,32,32) -> (B,36,32,32)
        self.c2 = nn.Sequential(
            DWSeparableConv(24, 36, k=3, stride=1),            # depthwise-separable
            DWSeparableConv(36, 36, k=3, stride=1, dilation=2) # dilated
        )

        # C3: (B,36,32,32) -> (B,48,32,32)
        self.c3 = nn.Sequential(
            DWSeparableConv(36, 48, k=3, stride=1),
            DWSeparableConv(48, 48, k=3, stride=1),
        )

        # C4: downsample only here (no maxpool)
        self.c4 = nn.Sequential(
            DWSeparableConv(48, 64, k=3, stride=2),            # ↓ 32->16
            DWSeparableConv(64, 64, k=3, stride=1, dilation=2),
            DWSeparableConv(64, 64, k=3, stride=1, dilation=3),
            DWSeparableConv(64, 64, k=3, stride=1),
            DWSeparableConv(64, 64, k=3, stride=1),
        )

        self.gap  = nn.AdaptiveAvgPool2d(1)                    # (B,64,1,1)
        self.head = nn.Conv2d(64, num_classes, kernel_size=1)  # (B,C,1,1)

    def forward(self, x):
        # Expect CIFAR-10: [B,3,32,32]
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)    # [B,64,16,16]
        x = self.gap(x)   # [B,64,1,1]
        x = self.head(x)  # [B,num_classes,1,1]
        x = x.view(x.size(0), -1)
        return F.log_softmax(x, dim=1)

# quick sanity for params
if __name__ == "__main__":
    m = NetCIFAR10_Tiny()
    total = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print("Trainable params:", total)  # ~36k