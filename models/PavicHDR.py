import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
import numpy as np
from torchvision.ops import DeformConv2d

class Mutual_Attention(nn.Module):
    def __init__(self, embbed_dim, num_heads, bias=False):
        super(Mutual_Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(embbed_dim, embbed_dim, kernel_size=1, bias=bias)
        self.k = nn.Conv2d(embbed_dim, embbed_dim, kernel_size=1, bias=bias)
        self.v = nn.Conv2d(embbed_dim, embbed_dim, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(embbed_dim, embbed_dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        assert x.shape == y.shape
        b, c, h, w = x.shape

        q = self.q(x)
        k = self.k(y)
        v = self.v(y)

        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = rearrange(
            out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w
        )
        out = self.project_out(out)
        return out


class ExposureMask(nn.Module):
    def __init__(self, dim_x, dim_y, num_heads):
        super(ExposureMask, self).__init__()
        self.attention = Mutual_Attention(dim_y, num_heads)
        self.conv_1 = nn.Conv2d(dim_x + dim_y, dim_y, 3, 1, 1)
        self.conv_2 = nn.Conv2d(dim_y, 1, 3, 1, 1)

    def forward(self, x, y):
        map = self.conv_1(torch.cat((x, y), axis=1))
        map = self.conv_2(map)
        mapped = x * map
        att = self.attention(y, mapped)
        return att

class ResBlock(nn.Module):
    def __init__(self, channels, dilation=1, bias=True):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=bias), 
            nn.PReLU(channels)
        )
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=bias)
        self.prelu = nn.PReLU(channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.prelu(x + out)
        return out


class SFTLayer(nn.Module):
    def __init__(self, in_nc=32, out_nc=64, nf=32):
        super(SFTLayer, self).__init__()
        self.SFT_scale_conv0 = nn.Conv2d(in_nc, nf, 1)
        self.SFT_scale_conv1 = nn.Conv2d(nf, out_nc, 1)
        self.SFT_shift_conv0 = nn.Conv2d(in_nc, nf, 1)
        self.SFT_shift_conv1 = nn.Conv2d(nf, out_nc, 1)

    def forward(self, x):
        # x[0]: fea; x[1]: cond
        scale = self.SFT_scale_conv1(
            F.leaky_relu(self.SFT_scale_conv0(x[1]), 0.1, inplace=True)
        )
        shift = self.SFT_shift_conv1(
            F.leaky_relu(self.SFT_shift_conv0(x[1]), 0.1, inplace=True)
        )
        return x[0] * (scale + 1) + shift

class ResBlock_with_SFT(nn.Module):
    def __init__(self, ch_x):
        super(ResBlock_with_SFT, self).__init__()
        self.conv1 = nn.Conv2d(ch_x, ch_x, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(ch_x, ch_x, 3, 1, 1, bias=True)

        dch = 2 * 2 * 3 * 3
        self.offset = nn.Conv2d(ch_x, ch_x, 3, 1, 1, bias=True)

        self.sft1 = SFTLayer(in_nc=30, out_nc=90, nf=30)
        self.dconv1 = DeformConv2d(ch_x, ch_x, 3, 1, 1)  # Todo :DCONV
        self.sft2 = SFTLayer(in_nc=30, out_nc=90, nf=30)
        self.dconv2 = DeformConv2d(ch_x, ch_x, 3, 1, 1)  # Todo :DCONV

    def forward(self, x):
        # x[0]: fea; x[1]: cond
        off = self.offset(x[0])
        fea = self.sft1(x)
        fea = F.relu(self.dconv1(fea, off), inplace=True)
        fea = self.sft2((fea, x[1]))
        fea = self.dconv2(fea, off)
        return (x[0] + fea, x[1])

class PavicHDR(nn.Module):
    r""" """

    def __init__(
        self,
        ch_in=6,
        ch_out=3,
        dim_f=30,
        dim_c=30,
        num_heads=5,
        window_size=8,
    ):
        super(PavicHDR, self).__init__()
        self.window_size = window_size
        self.feat_1 = nn.Conv2d(ch_in, dim_f, 3, 1, 1)
        self.feat_3 = nn.Conv2d(ch_in, dim_f, 3, 1, 1)
        self.mask_l = ExposureMask(dim_f, dim_c, num_heads)
        self.mask_h = ExposureMask(dim_f, dim_c, num_heads)
        self.mask_est = nn.Sequential(
            nn.Conv2d(ch_in, dim_c, 3, 1, 1),
            nn.PReLU(dim_c),
            nn.Conv2d(dim_c, dim_c, 3, 1, 1),
            nn.PReLU(dim_c),
            nn.Conv2d(dim_c, dim_c, 3, 1, 1),
            nn.PReLU(dim_c),
            nn.Conv2d(dim_c, dim_c, 3, 1, 1),
            nn.PReLU(dim_c),
        )

        
        self.resblock1 = ResBlock(dim_c + dim_f + dim_f, 1)
        self.resblock2 = ResBlock(dim_c + dim_f + dim_f, 2)
        self.resblock3 = ResBlock(dim_c + dim_f + dim_f, 4)
        self.resblock4 = ResBlock(dim_c + dim_f + dim_f, 8)
        self.resblock5 = ResBlock(dim_c + dim_f + dim_f, 4)
        self.resblock6 = ResBlock(dim_c + dim_f + dim_f, 2)
        self.resblock7 = ResBlock(dim_c + dim_f + dim_f, 1)
        
        self.resblock_sft_1 = ResBlock_with_SFT(dim_c + dim_f + dim_f)
        self.resblock_sft_2 = ResBlock_with_SFT(dim_c + dim_f + dim_f)
        self.resblock_sft_3 = ResBlock_with_SFT(dim_c + dim_f + dim_f)
        
        self.conv_end = nn.Conv2d(dim_c + dim_f + dim_f, ch_out, 3, 1, 1)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2, x3):
        
        x = torch.cat((x1, x2, x3), axis=1)
        
        H, W = x.shape[2:]

        x1, x2, x3 = x.chunk(3, dim=1)
        x1 = self.feat_1(x1)
        x2 = self.mask_est(x2)
        x3 = self.feat_3(x3)

        x1 = self.mask_l(x1, x2)
        x3 = self.mask_h(x3, x2)

        x_m = torch.cat((x1, x2, x3), axis=1)
        
        feat = self.resblock1(x_m)
        feat = self.resblock2(feat)
        feat, _ = self.resblock_sft_1((feat, x2))
        feat = self.resblock3(feat)
        feat = self.resblock4(feat)
        feat, _ = self.resblock_sft_2((feat, x2))
        feat = self.resblock5(feat)
        feat = self.resblock6(feat)
        feat, _ = self.resblock_sft_3((feat, x2))
        feat = self.resblock7(feat)
        
        x = self.conv_end(feat)
        
        return self.sigmoid(x[:, :, :H, :W])


if __name__ == "__main__":
    '''
    import pynvml
    import time
    from thop import profile

    model = PavicHDR().cuda().eval()

    w = 256
    h = 256
    img0_c = torch.randn(1, 6, h, w).cuda()
    img1_c = torch.randn(1, 6, h, w).cuda()
    img2_c = torch.randn(1, 6, h, w).cuda()

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    gpuName = pynvml.nvmlDeviceGetName(handle)
    print(gpuName)

    with torch.no_grad():
        for i in range(2):
            out = model(img0_c, img1_c, img2_c)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        time_stamp = time.time()
        for i in range(10):
            out = model(img0_c, img1_c, img2_c)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print("Time: {:.3f}s of 0.776s".format((time.time() - time_stamp) / 10))

    flops, params = profile(model, inputs=(img0_c, img1_c, img2_c), verbose=False)
    print(
        "FLOPs: {:.3f}T of 1.12T\nParams: {:.2f}M of 1.12M".format(
            flops / 1000 / 1000 / 1000 / 1000, params / 1000 / 1000
        )
    )
    print(out.shape)
##SAFNET 0.776 seconds, 0.976 TFlops, 1.12 MParams 1500x1000
##
    '''
    model = PavicHDR()
    x = torch.randn((1, 6, 128, 128))
    x = model(x, x, x)
    print(x.shape)

