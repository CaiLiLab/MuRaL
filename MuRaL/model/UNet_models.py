import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class ConvBlock(nn.Module):
    def __init__(self, inp, oup, expand_ratio=2, fused=True):
        super(ConvBlock, self).__init__()
        hidden_dim = round(inp * expand_ratio)
        self.conv = nn.Sequential(
            nn.Conv1d(inp, hidden_dim, 5, 1, padding=2, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(inplace=False),
            nn.Conv1d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm1d(oup),
        )

    def forward(self, x):
        return x + self.conv(x)

class UNet_pro(nn.Module):
    def __init__(self, distal_radius, n_class, out_channels, kernel_size, downsize, num_layers=5):
        super(UNet_pro, self).__init__()
        length = distal_radius*2
        self.channels = [out_channels+2**(i+3) for i in range(num_layers)]
        # self.channels = [out_channels,out_channels+32,out_channels+64,out_channels+96,out_channels+96]
        self.uplblocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(4, self.channels[0], stride=downsize[0], kernel_size=kernel_size, padding=(kernel_size-1)//2), nn.BatchNorm1d(self.channels[0])
                ),
                nn.Sequential(
                    nn.Conv1d(self.channels[0], self.channels[1], stride=downsize[1], kernel_size=kernel_size, padding=(kernel_size-1)//2),
                    nn.BatchNorm1d(self.channels[1]),
                ),
                nn.Sequential(
                    nn.Conv1d(self.channels[1], self.channels[2], stride=downsize[2], kernel_size=kernel_size, padding=(kernel_size-1)//2),
                    nn.BatchNorm1d(self.channels[2]),
                ),
                nn.Sequential(
                    nn.Conv1d(self.channels[2], self.channels[3], stride=downsize[3], kernel_size=kernel_size, padding=(kernel_size-1)//2),
                    nn.BatchNorm1d(self.channels[3]),
                ),
                nn.Sequential(
                    nn.Conv1d(self.channels[3], self.channels[4], stride=downsize[4], kernel_size=kernel_size, padding=(kernel_size-1)//2),
                    nn.BatchNorm1d(self.channels[4]),
                ),
            ]
        )

        self.upblocks = nn.ModuleList(
            [
                nn.Sequential(
                    ConvBlock(self.channels[0], self.channels[0], fused=True), ConvBlock(self.channels[0], self.channels[0], fused=True)
                ),
                nn.Sequential(
                    ConvBlock(self.channels[1], self.channels[1], fused=True), ConvBlock(self.channels[1], self.channels[1], fused=True)
                ),
                nn.Sequential(
                    ConvBlock(self.channels[2], self.channels[2], fused=True), ConvBlock(self.channels[2], self.channels[2], fused=True)
                ),
                nn.Sequential(
                    ConvBlock(self.channels[3], self.channels[3], fused=True), ConvBlock(self.channels[3], self.channels[3], fused=True)
                ),
                nn.Sequential(
                    ConvBlock(self.channels[4], self.channels[4], fused=True), ConvBlock(self.channels[4], self.channels[4], fused=True)
                ),
            ]
        )

        self.downlblocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Upsample(scale_factor=downsize[4]),
                    nn.Conv1d(self.channels[4], self.channels[3], kernel_size=kernel_size, padding=(kernel_size-1)//2),
                    nn.BatchNorm1d(self.channels[3]),
                ),
                nn.Sequential(
                    nn.Upsample(scale_factor=downsize[3]),
                    nn.Conv1d(self.channels[3], self.channels[2], kernel_size=kernel_size, padding=(kernel_size-1)//2),
                    nn.BatchNorm1d(self.channels[2]),
                ),
                nn.Sequential(
                    nn.Upsample(scale_factor=downsize[2]),
                    nn.Conv1d(self.channels[2], self.channels[1], kernel_size=kernel_size, padding=(kernel_size-1)//2),
                    nn.BatchNorm1d(self.channels[1]),
                ),
                nn.Sequential(
                    nn.Upsample(scale_factor=downsize[1]),
                    nn.Conv1d(self.channels[1], self.channels[0], kernel_size=kernel_size, padding=(kernel_size-1)//2),
                    nn.BatchNorm1d(self.channels[0]),
                ),
            ]
        )

        self.downblocks = nn.ModuleList(
            [
                nn.Sequential(
                    ConvBlock(self.channels[3], self.channels[3], fused=True), ConvBlock(self.channels[3], self.channels[3], fused=True)
                ),
                nn.Sequential(
                    ConvBlock(self.channels[2], self.channels[2], fused=True), ConvBlock(self.channels[2], self.channels[2], fused=True)
                ),
                nn.Sequential(
                    ConvBlock(self.channels[1], self.channels[1], fused=True), ConvBlock(self.channels[1], self.channels[1], fused=True)
                ),
                nn.Sequential(
                    ConvBlock(self.channels[0], self.channels[0], fused=True), ConvBlock(self.channels[0], self.channels[0], fused=True)
                ),
            ]
        )

        self.uplblocks2 = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(self.channels[0], self.channels[1], stride=downsize[1], kernel_size=kernel_size, padding=(kernel_size-1)//2),
                    nn.BatchNorm1d(self.channels[1]),
                ),
                nn.Sequential(
                    nn.Conv1d(self.channels[1], self.channels[2], stride=downsize[2], kernel_size=kernel_size, padding=(kernel_size-1)//2),
                    nn.BatchNorm1d(self.channels[2]),
                ),
                nn.Sequential(
                    nn.Conv1d(self.channels[2], self.channels[3], stride=downsize[3], kernel_size=kernel_size, padding=(kernel_size-1)//2),
                    nn.BatchNorm1d(self.channels[3]),
                ),
                nn.Sequential(
                    nn.Conv1d(self.channels[3], self.channels[4], stride=downsize[4], kernel_size=kernel_size, padding=(kernel_size-1)//2),
                    nn.BatchNorm1d(self.channels[4]),
                ),
            ]
        )

        self.upblocks2 = nn.ModuleList(
            [
                nn.Sequential(
                    ConvBlock(self.channels[1], self.channels[1], fused=True), ConvBlock(self.channels[1], self.channels[1], fused=True)
                ),
                nn.Sequential(
                    ConvBlock(self.channels[2], self.channels[2], fused=True), ConvBlock(self.channels[2], self.channels[2], fused=True)
                ),
                nn.Sequential(
                    ConvBlock(self.channels[3], self.channels[3], fused=True), ConvBlock(self.channels[3], self.channels[3], fused=True)
                ),
                nn.Sequential(
                    ConvBlock(self.channels[4], self.channels[4], fused=True), ConvBlock(self.channels[4], self.channels[4], fused=True)
                ),
            ]
        )

        self.downlblocks2 = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Upsample(scale_factor=downsize[4]),
                    nn.Conv1d(self.channels[4], self.channels[3], kernel_size=kernel_size, padding=(kernel_size-1)//2),
                    nn.BatchNorm1d(self.channels[3]),
                ),
                nn.Sequential(
                    nn.Upsample(scale_factor=downsize[3]),
                    nn.Conv1d(self.channels[3], self.channels[2], kernel_size=kernel_size, padding=(kernel_size-1)//2),
                    nn.BatchNorm1d(self.channels[2]),
                ),
                nn.Sequential(
                    nn.Upsample(scale_factor=downsize[2]),
                    nn.Conv1d(self.channels[2], self.channels[1], kernel_size=kernel_size, padding=(kernel_size-1)//2),
                    nn.BatchNorm1d(self.channels[1]),
                ),
                nn.Sequential(
                    nn.Upsample(scale_factor=downsize[1]),
                    nn.Conv1d(self.channels[1], self.channels[0], kernel_size=kernel_size, padding=(kernel_size-1)//2),
                    nn.BatchNorm1d(self.channels[0]),
                ),
            ]
        )

        self.downblocks2 = nn.ModuleList(
            [
                nn.Sequential(
                    ConvBlock(self.channels[3], self.channels[3], fused=True), ConvBlock(self.channels[3], self.channels[3], fused=True)
                ),
                nn.Sequential(
                    ConvBlock(self.channels[2], self.channels[2], fused=True), ConvBlock(self.channels[2], self.channels[2], fused=True)
                ),
                nn.Sequential(
                    ConvBlock(self.channels[1], self.channels[1], fused=True), ConvBlock(self.channels[1], self.channels[1], fused=True)
                ),
                nn.Sequential(
                    ConvBlock(self.channels[0], self.channels[0], fused=True), ConvBlock(self.channels[0], self.channels[0], fused=True)
                ),
            ]
        )

        self.out_conv = nn.Sequential(
            nn.Conv1d(self.channels[0], self.channels[0], kernel_size=1),
            nn.BatchNorm1d(self.channels[0]),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.channels[0], self.channels[0], kernel_size=1),
            nn.Softplus(),
        )

        self.out_fc = nn.Sequential(
            nn.BatchNorm1d(self.channels[0]),
            nn.Dropout(0.1),
            nn.Linear(self.channels[0], n_class), 
            nn.Softplus(),
        )

    def forward(self, local_input, distal_input):
        """Forward propagation of a batch."""
        out = distal_input
        encodings = []
        for i, lconv, conv in zip(
            np.arange(len(self.uplblocks)), self.uplblocks, self.upblocks
        ):
            lout = lconv(out)
            out = conv(lout)
            encodings.append(out)

        encodings2 = [out]
        for enc, lconv, conv in zip(
            reversed(encodings[:-1]), self.downlblocks, self.downblocks
        ):
            lout = lconv(out)
            out = conv(lout)
            out = enc + out
            encodings2.append(out)

        encodings3 = [out]
        for enc, lconv, conv in zip(
            reversed(encodings2[:-1]), self.uplblocks2, self.upblocks2
        ):
            lout = lconv(out)
            out = conv(lout)
            out = enc + out
            encodings3.append(out)

        for enc, lconv, conv in zip(
            reversed(encodings3[:-1]), self.downlblocks2, self.downblocks2
        ):
            lout = lconv(out)
            out = conv(lout)
            out = enc + out

        out = self.out_conv(out)
        out, _ = torch.max(out, dim=2)
        out = self.out_fc(out)

        return out


class ConvBlock1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, expand_ratio=2):
        super(ConvBlock1, self).__init__()
        hidden_dim = round(in_channels * expand_ratio)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size, stride, kernel_size-1, dilation=2, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(inplace=False),
            nn.Conv1d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm1d(out_channels),
        )
    def forward(self, x):
        x = self.conv(x)

        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, dilation=1):
        """Residual block ('bottleneck' version)"""
        super(ResidualBlock, self).__init__()
        
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(in_channels)
        self.conv2 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, dilation=dilation)        
        
        self.layer = nn.Sequential(nn.ELU(),self.bn1, self.conv1, nn.ELU(), self.bn2, self.conv2)

    def forward(self, x):
        out = self.layer(x)
        d = x.shape[2] - out.shape[2]
        out = x[:,:,0:x.shape[2]-d] + out
        
        return out

class UNet_Small(nn.Module):
    def __init__(self, n_class, out_channels, kernel_size, downsize):
        super(UNet_Small, self).__init__()

        # self.channels = [out_channels+2**(i+3) for i in range(num_layers)]
        # self.channels = [out_channels,out_channels+16,out_channels+32,out_channels+64,out_channels+96,out_channels+128]
        self.channels = [out_channels,out_channels*2,out_channels*3,out_channels*4,out_channels*5,out_channels*6]
        self.uplblocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(4, self.channels[0], stride=downsize[0], kernel_size=kernel_size, padding=(kernel_size-1)//2), nn.BatchNorm1d(self.channels[0])
                ),
                nn.Sequential(
                    nn.Conv1d(self.channels[0], self.channels[1], stride=downsize[1], kernel_size=kernel_size, padding=(kernel_size-1)//2),
                    nn.BatchNorm1d(self.channels[1]),
                ),
                nn.Sequential(
                    nn.Conv1d(self.channels[1], self.channels[2], stride=downsize[2], kernel_size=kernel_size, padding=(kernel_size-1)//2),
                    nn.BatchNorm1d(self.channels[2]),
                ),
                nn.Sequential(
                    nn.Conv1d(self.channels[2], self.channels[3], stride=downsize[3], kernel_size=kernel_size, padding=(kernel_size-1)//2),
                    nn.BatchNorm1d(self.channels[3]),
                ),
                nn.Sequential(
                    nn.Conv1d(self.channels[3], self.channels[4], stride=downsize[4], kernel_size=kernel_size, padding=(kernel_size-1)//2),
                    nn.BatchNorm1d(self.channels[4]),
                ),
                nn.Sequential(
                    nn.Conv1d(self.channels[4], self.channels[5], stride=downsize[5], kernel_size=kernel_size, padding=(kernel_size-1)//2),
                    nn.BatchNorm1d(self.channels[5]),
                ),
            ]
        )

        self.upblocks = nn.ModuleList(
            [
                nn.Sequential(
                    ConvBlock(self.channels[0], self.channels[0], fused=True),
                ),
                nn.Sequential(
                    ConvBlock(self.channels[1], self.channels[1], fused=True)
                ),
                nn.Sequential(
                    ConvBlock(self.channels[2], self.channels[2], fused=True),
                ),
                nn.Sequential(
                    ConvBlock(self.channels[3], self.channels[3], fused=True),
                ),
                nn.Sequential(
                    ConvBlock(self.channels[4], self.channels[4], fused=True),
                ),
                nn.Sequential(
                    ConvBlock(self.channels[5], self.channels[5], fused=True),
                ),
            ]
        )

        self.downlblocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Upsample(scale_factor=downsize[5]),
                    nn.Conv1d(self.channels[5], self.channels[4], kernel_size=kernel_size, padding=(kernel_size-1)//2),
                    nn.BatchNorm1d(self.channels[4]),
                ),
                nn.Sequential(
                    nn.Upsample(scale_factor=downsize[4]),
                    nn.Conv1d(self.channels[4], self.channels[3], kernel_size=kernel_size, padding=(kernel_size-1)//2),
                    nn.BatchNorm1d(self.channels[3]),
                ),
                nn.Sequential(
                    nn.Upsample(scale_factor=downsize[3]),
                    nn.Conv1d(self.channels[3], self.channels[2], kernel_size=kernel_size, padding=(kernel_size-1)//2),
                    nn.BatchNorm1d(self.channels[2]),
                ),
                nn.Sequential(
                    nn.Upsample(scale_factor=downsize[2]),
                    nn.Conv1d(self.channels[2], self.channels[1], kernel_size=kernel_size, padding=(kernel_size-1)//2),
                    nn.BatchNorm1d(self.channels[1]),
                ),
                nn.Sequential(
                    nn.Upsample(scale_factor=downsize[1]),
                    nn.Conv1d(self.channels[1], self.channels[0], kernel_size=kernel_size, padding=(kernel_size-1)//2),
                    nn.BatchNorm1d(self.channels[0]),
                ),
            ]
        )

        self.downblocks = nn.ModuleList(
            [
                nn.Sequential(
                    ConvBlock(self.channels[4], self.channels[4], fused=True),
                ),
                nn.Sequential(
                    ConvBlock(self.channels[3], self.channels[3], fused=True),
                ),
                nn.Sequential(
                    ConvBlock(self.channels[2], self.channels[2], fused=True),
                ),
                nn.Sequential(
                    ConvBlock(self.channels[1], self.channels[1], fused=True),
                ),
                nn.Sequential(
                    ConvBlock(self.channels[0], self.channels[0], fused=True),
                ),
            ]
        )

        self.out_conv = nn.Sequential(
            nn.Conv1d(self.channels[0], self.channels[0], kernel_size=1),
            nn.BatchNorm1d(self.channels[0]),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.channels[0], self.channels[0], kernel_size=1),
            nn.Softplus(),
        )

        self.out_fc = nn.Sequential(
            nn.BatchNorm1d(self.channels[0]),
            nn.Dropout(0.1),
            nn.Linear(self.channels[0], n_class), 
            nn.Softplus(),
        )

    def forward(self, local_input, distal_input):
        """Forward propagation of a batch."""
        out = distal_input
        encodings = []
        for i, lconv, conv in zip(
            np.arange(len(self.uplblocks)), self.uplblocks, self.upblocks
        ):
            lout = lconv(out)
            out = conv(lout)
            encodings.append(out)

        for enc, lconv, conv in zip(
            reversed(encodings[:-1]), self.downlblocks, self.downblocks
        ):
            lout = lconv(out)
            out = conv(lout)
            out = enc + out

        out = self.out_conv(out)
        out, _ = torch.max(out, dim=2)
        out = self.out_fc(out)

        return out

class UNet_Dilation(nn.Module):
    def __init__(self, n_class, out_channels, kernel_size, downsize):
        super(UNet_Dilation, self).__init__()

        # self.channels = [out_channels+2**(i+3) for i in range(num_layers)]
        # self.channels = [out_channels,out_channels+32,out_channels+64,out_channels+96,out_channels+96]
        self.channels = [out_channels,out_channels+16,out_channels+32,out_channels+64,out_channels+96,out_channels+128]
        self.uplblocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(4, self.channels[0], stride=downsize[0], kernel_size=kernel_size, padding=(kernel_size-1)//2), nn.BatchNorm1d(self.channels[0])
                ),
                nn.Sequential(
                    nn.Conv1d(self.channels[0], self.channels[1], stride=downsize[1], kernel_size=kernel_size, padding=(kernel_size-1)//2),
                    nn.BatchNorm1d(self.channels[1]),
                ),
                nn.Sequential(
                    nn.Conv1d(self.channels[1], self.channels[2], stride=downsize[2], kernel_size=kernel_size, padding=(kernel_size-1)//2),
                    nn.BatchNorm1d(self.channels[2]),
                ),
                nn.Sequential(
                    nn.Conv1d(self.channels[2], self.channels[3], stride=downsize[3], kernel_size=kernel_size, padding=(kernel_size-1)//2),
                    nn.BatchNorm1d(self.channels[3]),
                ),
                nn.Sequential(
                    nn.Conv1d(self.channels[3], self.channels[4], stride=downsize[4], kernel_size=kernel_size, padding=(kernel_size-1)//2),
                    nn.BatchNorm1d(self.channels[4]),
                ),
                nn.Sequential(
                    nn.Conv1d(self.channels[4], self.channels[5], stride=downsize[5], kernel_size=kernel_size, padding=(kernel_size-1)//2),
                    nn.BatchNorm1d(self.channels[5]),
                ),
            ]
        )

        self.upblocks = nn.ModuleList(
            [
                nn.Sequential(
                    ConvBlock1(self.channels[0], self.channels[0]),
                ),
                nn.Sequential(
                    ConvBlock1(self.channels[1], self.channels[1])
                ),
                nn.Sequential(
                    ConvBlock1(self.channels[2], self.channels[2]),
                ),
                nn.Sequential(
                    ConvBlock1(self.channels[3], self.channels[3]),
                ),
                nn.Sequential(
                    ConvBlock1(self.channels[4], self.channels[4]),
                ),
                nn.Sequential(
                    ConvBlock1(self.channels[5], self.channels[5]),
                ),
            ]
        )

        self.downlblocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Upsample(scale_factor=downsize[5]),
                    nn.Conv1d(self.channels[5], self.channels[4], kernel_size=kernel_size, padding=(kernel_size-1)//2),
                    nn.BatchNorm1d(self.channels[4]),
                ),
                nn.Sequential(
                    nn.Upsample(scale_factor=downsize[4]),
                    nn.Conv1d(self.channels[4], self.channels[3], kernel_size=kernel_size, padding=(kernel_size-1)//2),
                    nn.BatchNorm1d(self.channels[3]),
                ),
                nn.Sequential(
                    nn.Upsample(scale_factor=downsize[3]),
                    nn.Conv1d(self.channels[3], self.channels[2], kernel_size=kernel_size, padding=(kernel_size-1)//2),
                    nn.BatchNorm1d(self.channels[2]),
                ),
                nn.Sequential(
                    nn.Upsample(scale_factor=downsize[2]),
                    nn.Conv1d(self.channels[2], self.channels[1], kernel_size=kernel_size, padding=(kernel_size-1)//2),
                    nn.BatchNorm1d(self.channels[1]),
                ),
                nn.Sequential(
                    nn.Upsample(scale_factor=downsize[1]),
                    nn.Conv1d(self.channels[1], self.channels[0], kernel_size=kernel_size, padding=(kernel_size-1)//2),
                    nn.BatchNorm1d(self.channels[0]),
                ),
            ]
        )

        self.downblocks = nn.ModuleList(
            [
                nn.Sequential(
                    ConvBlock1(self.channels[4], self.channels[4]),
                ),
                nn.Sequential(
                    ConvBlock1(self.channels[3], self.channels[3]),
                ),
                nn.Sequential(
                    ConvBlock1(self.channels[2], self.channels[2]),
                ),
                nn.Sequential(
                    ConvBlock1(self.channels[1], self.channels[1]),
                ),
                nn.Sequential(
                    ConvBlock1(self.channels[0], self.channels[0]),
                ),
            ]
        )

        self.out_conv = nn.Sequential(
            nn.Conv1d(self.channels[0], self.channels[0], kernel_size=1),
            nn.BatchNorm1d(self.channels[0]),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.channels[0], self.channels[0], kernel_size=1),
            nn.Softplus(),
        )

        self.out_fc = nn.Sequential(
            nn.BatchNorm1d(self.channels[0]),
            nn.Dropout(0.1),
            nn.Linear(self.channels[0], n_class), 
            nn.Softplus(),
        )

    def forward(self, local_input, distal_input):
        """Forward propagation of a batch."""
        out = distal_input
        encodings = []
        for i, lconv, conv in zip(
            np.arange(len(self.uplblocks)), self.uplblocks, self.upblocks
        ):
            lout = lconv(out)
            out = conv(lout)
            encodings.append(out)

        for enc, lconv, conv in zip(
            reversed(encodings[:-1]), self.downlblocks, self.downblocks
        ):
            lout = lconv(out)
            out = conv(lout)
            out = enc + out

        out = self.out_conv(out)
        out, _ = torch.max(out, dim=2)
        out = self.out_fc(out)

        return out

class UNet_Res(nn.Module):
    def __init__(self, n_class, out_channels, kernel_size, downsize):
        super(UNet_Res, self).__init__()

        # self.channels = [out_channels+2**(i+3) for i in range(num_layers)]
        self.channels = [out_channels,out_channels+16,out_channels+32,out_channels+64,out_channels+96,out_channels+128]
        self.uplblocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(4, self.channels[0], stride=downsize[0], kernel_size=kernel_size, padding=(kernel_size-1)//2), nn.BatchNorm1d(self.channels[0])
                ),
                nn.Sequential(
                    nn.Conv1d(self.channels[0], self.channels[1], stride=downsize[1], kernel_size=kernel_size, padding=(kernel_size-1)//2),
                    nn.BatchNorm1d(self.channels[1]),
                ),
                nn.Sequential(
                    nn.Conv1d(self.channels[1], self.channels[2], stride=downsize[2], kernel_size=kernel_size, padding=(kernel_size-1)//2),
                    nn.BatchNorm1d(self.channels[2]),
                ),
                nn.Sequential(
                    nn.Conv1d(self.channels[2], self.channels[3], stride=downsize[3], kernel_size=kernel_size, padding=(kernel_size-1)//2),
                    nn.BatchNorm1d(self.channels[3]),
                ),
                nn.Sequential(
                    nn.Conv1d(self.channels[3], self.channels[4], stride=downsize[4], kernel_size=kernel_size, padding=(kernel_size-1)//2),
                    nn.BatchNorm1d(self.channels[4]),
                ),
                nn.Sequential(
                    nn.Conv1d(self.channels[4], self.channels[5], stride=downsize[5], kernel_size=kernel_size, padding=(kernel_size-1)//2),
                    nn.BatchNorm1d(self.channels[5]),
                ),
            ]
        )

        self.upblocks = nn.ModuleList(
            [
                nn.Sequential(
                    ResidualBlock(self.channels[0], self.channels[0]),
                ),
                nn.Sequential(
                    ResidualBlock(self.channels[1], self.channels[1])
                ),
                nn.Sequential(
                    ResidualBlock(self.channels[2], self.channels[2]),
                ),
                nn.Sequential(
                    ResidualBlock(self.channels[3], self.channels[3]),
                ),
                nn.Sequential(
                    ResidualBlock(self.channels[4], self.channels[4]),
                ),
                nn.Sequential(
                    ResidualBlock(self.channels[5], self.channels[5]),
                ),
            ]
        )

        self.downlblocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Upsample(scale_factor=downsize[5]),
                    nn.Conv1d(self.channels[5], self.channels[4], kernel_size=kernel_size, padding=(kernel_size-1)//2),
                    nn.BatchNorm1d(self.channels[4]),
                ),
                nn.Sequential(
                    nn.Upsample(scale_factor=downsize[4]),
                    nn.Conv1d(self.channels[4], self.channels[3], kernel_size=kernel_size, padding=(kernel_size-1)//2),
                    nn.BatchNorm1d(self.channels[3]),
                ),
                nn.Sequential(
                    nn.Upsample(scale_factor=downsize[3]),
                    nn.Conv1d(self.channels[3], self.channels[2], kernel_size=kernel_size, padding=(kernel_size-1)//2),
                    nn.BatchNorm1d(self.channels[2]),
                ),
                nn.Sequential(
                    nn.Upsample(scale_factor=downsize[2]),
                    nn.Conv1d(self.channels[2], self.channels[1], kernel_size=kernel_size, padding=(kernel_size-1)//2),
                    nn.BatchNorm1d(self.channels[1]),
                ),
                nn.Sequential(
                    nn.Upsample(scale_factor=downsize[1]),
                    nn.Conv1d(self.channels[1], self.channels[0], kernel_size=kernel_size, padding=(kernel_size-1)//2),
                    nn.BatchNorm1d(self.channels[0]),
                ),
            ]
        )

        self.downblocks = nn.ModuleList(
            [
                nn.Sequential(
                    ResidualBlock(self.channels[4], self.channels[4]),
                ),
                nn.Sequential(
                    ResidualBlock(self.channels[3], self.channels[3]),
                ),
                nn.Sequential(
                    ResidualBlock(self.channels[2], self.channels[2]),
                ),
                nn.Sequential(
                    ResidualBlock(self.channels[1], self.channels[1]),
                ),
                nn.Sequential(
                    ResidualBlock(self.channels[0], self.channels[0]),
                ),
            ]
        )

        self.out_conv = nn.Sequential(
            nn.Conv1d(self.channels[0], self.channels[0], kernel_size=1),
            nn.BatchNorm1d(self.channels[0]),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.channels[0], self.channels[0], kernel_size=1),
            nn.Softplus(),
        )

        self.out_fc = nn.Sequential(
            nn.BatchNorm1d(self.channels[0]),
            nn.Dropout(0.1),
            nn.Linear(self.channels[0], n_class), 
            nn.Softplus(),
        )

    def forward(self, local_input, distal_input):
        """Forward propagation of a batch."""
        out = distal_input
        encodings = []
        for i, lconv, conv in zip(
            np.arange(len(self.uplblocks)), self.uplblocks, self.upblocks
        ):
            lout = lconv(out)
            out = conv(lout)
            encodings.append(out)

        for enc, lconv, conv in zip(
            reversed(encodings[:-1]), self.downlblocks, self.downblocks
        ):
            lout = lconv(out)
            out = conv(lout)
            out = enc + out

        out = self.out_conv(out)
        out, _ = torch.max(out, dim=2)
        out = self.out_fc(out)

        return out

class UNet3(nn.Module):
    """ UNet_3Plus with Deep Supervision """
    def __init__(self, n_class, in_channels=16, kernel_size=3, num_layers=5):
        super(UNet3, self).__init__()
        self.channels = [in_channels*2**i for i in range(num_layers)]
        cat_channels = self.channels[0]
        cat_blocks = len(self.channels)
        upsample_channels = cat_blocks * cat_channels

        self.downblocks = nn.ModuleList([
            nn.Sequential(
                    nn.Conv1d(4, self.channels[0], stride=2, kernel_size=9, padding=4), nn.BatchNorm1d(self.channels[0]),
                ),
            nn.Sequential(
                    nn.Conv1d(self.channels[0], self.channels[1], stride=2, kernel_size=9, padding=4), nn.BatchNorm1d(self.channels[1]),
                ),
            nn.Sequential(
                    nn.Conv1d(self.channels[1], self.channels[2], stride=2, kernel_size=9, padding=4), nn.BatchNorm1d(self.channels[2]),
                ),
            nn.Sequential(
                    nn.Conv1d(self.channels[2], self.channels[3], stride=5, kernel_size=9, padding=4), nn.BatchNorm1d(self.channels[3]),
                ),
            nn.Sequential(
                    nn.Conv1d(self.channels[3], self.channels[4], stride=5, kernel_size=9, padding=4), nn.BatchNorm1d(self.channels[4]),
                ),
        ])
        self.downblocks1 = nn.ModuleList([
            nn.Sequential(
                    ConvBlock1(self.channels[0], self.channels[0]), ConvBlock1(self.channels[0], self.channels[0])
                ),
            nn.Sequential(
                    ConvBlock1(self.channels[1], self.channels[1]), ConvBlock1(self.channels[1], self.channels[1])
                ),
            nn.Sequential(
                    ConvBlock1(self.channels[2], self.channels[2]), ConvBlock1(self.channels[2], self.channels[2])
                ),
            nn.Sequential(
                    ConvBlock1(self.channels[3], self.channels[3]), ConvBlock1(self.channels[3], self.channels[3])
                ),
            nn.Sequential(
                    ConvBlock1(self.channels[4], self.channels[4]), ConvBlock1(self.channels[4], self.channels[4])
                ),
        ])

        self.out_conv = nn.Sequential(
            nn.Conv1d(upsample_channels, cat_channels, kernel_size=1),
            nn.BatchNorm1d(cat_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(cat_channels, cat_channels, kernel_size=1),
        )

        self.out_fc = nn.Sequential(
            nn.BatchNorm1d(cat_channels),
            nn.Dropout(0.1), 
            nn.Linear(cat_channels, n_class), 
            nn.Softplus(),
        )

        # decode layer4
        self.e1d4 = nn.Sequential(
            nn.Conv1d(self.channels[0], self.channels[0], stride=20, kernel_size=21, padding=10),
            nn.Conv1d(self.channels[0], self.channels[0], kernel_size=9, padding=4),
            nn.BatchNorm1d(self.channels[0]),
            )
        self.e1d4_conv = ConvBlock1(self.channels[0], cat_channels)

        self.e2d4 = nn.Sequential(
            nn.Conv1d(self.channels[1], self.channels[1], stride=10, kernel_size=11, padding=5),
            nn.Conv1d(self.channels[1], self.channels[1], kernel_size=9, padding=4),
            nn.BatchNorm1d(self.channels[1]),
            )
        self.e2d4_conv = ConvBlock1(self.channels[1], cat_channels)

        self.e3d4 = nn.Sequential(
            nn.Conv1d(self.channels[2], self.channels[2], stride=5, kernel_size=9, padding=4),
            nn.Conv1d(self.channels[2], self.channels[2], kernel_size=9, padding=4),
            nn.BatchNorm1d(self.channels[2]),
            )
        self.e3d4_conv = ConvBlock1(self.channels[2], cat_channels)

        self.e4d4 = nn.Sequential(
            nn.Conv1d(self.channels[3], self.channels[3], kernel_size=9, padding=4),
            nn.BatchNorm1d(self.channels[3]),
            )
        self.e4d4_conv = ConvBlock1(self.channels[3], cat_channels)

        self.e5d4 = nn.Sequential(
            nn.Upsample(scale_factor=5),
            nn.Conv1d(self.channels[4], self.channels[4], kernel_size=9, padding=4),
            nn.BatchNorm1d(self.channels[4]),
            )
        self.e5d4_conv = ConvBlock1(self.channels[4], cat_channels)

        # decode layer3
        self.e1d3 = nn.Sequential(
            nn.Conv1d(self.channels[0], self.channels[0], stride=4, kernel_size=9, padding=4),
            nn.Conv1d(self.channels[0], self.channels[0], kernel_size=9, padding=4),
            nn.BatchNorm1d(self.channels[0]),
            )
        self.e1d3_conv = ConvBlock1(self.channels[0], cat_channels)

        self.e2d3 = nn.Sequential(
            nn.Conv1d(self.channels[1], self.channels[1], stride=2, kernel_size=9, padding=4),
            nn.Conv1d(self.channels[1], self.channels[1], kernel_size=9, padding=4),
            nn.BatchNorm1d(self.channels[1]),
            )
        self.e2d3_conv = ConvBlock1(self.channels[1], cat_channels)

        self.e3d3 = nn.Sequential(
            nn.Conv1d(self.channels[2], self.channels[2], kernel_size=9, padding=4),
            nn.BatchNorm1d(self.channels[2]),
            )
        self.e3d3_conv = ConvBlock1(self.channels[2], cat_channels)

        self.e4d3 = nn.Sequential(
            nn.Upsample(scale_factor=5),
            nn.Conv1d(self.channels[3], self.channels[3], kernel_size=9, padding=4),
            nn.BatchNorm1d(self.channels[3]),
            )
        self.e4d3_conv = ConvBlock1(self.channels[3], cat_channels)

        self.e5d3 = nn.Sequential(
            nn.Upsample(scale_factor=25),
            nn.Conv1d(self.channels[4], self.channels[4], kernel_size=9, padding=4),
            nn.BatchNorm1d(self.channels[4]),
            )
        self.e5d3_conv = ConvBlock1(self.channels[4], cat_channels)

        # decode layer2
        self.e1d2 = nn.Sequential(
            nn.Conv1d(self.channels[0], self.channels[0], stride=2, kernel_size=9, padding=4),
            nn.Conv1d(self.channels[0], self.channels[0], kernel_size=9, padding=4),
            nn.BatchNorm1d(self.channels[0]),
            )
        self.e1d2_conv = ConvBlock1(self.channels[0], cat_channels)

        self.e2d2 = nn.Sequential(
            nn.Conv1d(self.channels[1], self.channels[1], kernel_size=9, padding=4),
            nn.BatchNorm1d(self.channels[1]),
            )
        self.e2d2_conv = ConvBlock1(self.channels[1], cat_channels)

        self.d3d2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv1d(upsample_channels, upsample_channels, kernel_size=9, padding=4),
            nn.BatchNorm1d(upsample_channels),
            )
        self.d3d2_conv = ConvBlock1(upsample_channels, cat_channels)

        self.e4d2 = nn.Sequential(
            nn.Upsample(scale_factor=10),
            nn.Conv1d(self.channels[3], self.channels[3], kernel_size=9, padding=4),
            nn.BatchNorm1d(self.channels[3]),
            )
        self.e4d2_conv = ConvBlock1(self.channels[3], cat_channels)

        self.e5d2 = nn.Sequential(
            nn.Upsample(scale_factor=50),
            nn.Conv1d(self.channels[4], self.channels[4], kernel_size=9, padding=4),
            nn.BatchNorm1d(self.channels[4]),
            )
        self.e5d2_conv = ConvBlock1(self.channels[4], cat_channels)

        # decode layer1
        self.e1d1 = nn.Sequential(
            nn.Conv1d(self.channels[0], self.channels[0], kernel_size=9, padding=4),
            nn.BatchNorm1d(self.channels[0]),
            )
        self.e1d1_conv = ConvBlock1(self.channels[0], cat_channels)

        self.d2d1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv1d(upsample_channels, upsample_channels, kernel_size=9, padding=4),
            nn.BatchNorm1d(upsample_channels),
            )
        self.d2d1_conv = ConvBlock1(upsample_channels, cat_channels)

        self.d3d1 = nn.Sequential(
            nn.Upsample(scale_factor=4),
            nn.Conv1d(upsample_channels, upsample_channels, kernel_size=9, padding=4),
            nn.BatchNorm1d(upsample_channels),
            )
        self.d3d1_conv = ConvBlock1(upsample_channels, cat_channels)

        self.e4d1 = nn.Sequential(
            nn.Upsample(scale_factor=20),
            nn.Conv1d(self.channels[3], self.channels[3], kernel_size=9, padding=4),
            nn.BatchNorm1d(self.channels[3]),
            )
        self.e4d1_conv = ConvBlock1(self.channels[3], cat_channels)

        self.e5d1 = nn.Sequential(
            nn.Upsample(scale_factor=100),
            nn.Conv1d(self.channels[4], self.channels[4], kernel_size=9, padding=4),
            nn.BatchNorm1d(self.channels[4]),
            )
        self.e5d1_conv = ConvBlock1(self.channels[4], cat_channels)

        # last layer does not have batch norm and relu
        self.d_conv = ConvBlock1(upsample_channels, upsample_channels)

    def forward(self, local_input, distal_input):
        """Forward propagation of a batch."""
        out = distal_input   # 128,4,2000
        """ Encoder"""
        # block1
        e1 = self.downblocks[0](out)  # 128,32,2000
        e1 = self.downblocks1[0](e1)
        # block2
        e2 = self.downblocks[1](e1)  # 128,64,1000
        e2 = self.downblocks1[1](e2)
        # block3
        e3 = self.downblocks[2](e2)  # 128,128,500
        e3 = self.downblocks1[2](e3)
        # block4
        # bottleneck layer
        e4 = self.downblocks[3](e3)  # 128,256,250
        e4 = self.downblocks1[3](e4)

        e5 = self.downblocks[4](e4)  # 128,512,50
        e5 = self.downblocks1[4](e5)

        """ d4 """
        e1_d4 = self.e1d4(e1)  # 128,32,2000  --> 128,32,250
        e1_d4 = self.e1d4_conv(e1_d4)  # 128,32,250  --> 128,32,250

        e2_d4 = self.e2d4(e2)  # 128,64,1000 --> 128,64,250
        e2_d4 = self.e2d4_conv(e2_d4)  # 128,64,250 --> 128,32,250

        e3_d4 = self.e3d4(e3)  # 128,128,500  --> 128,128,250
        e3_d4 = self.e3d4_conv(e3_d4)  # 128,128,250  --> 128,32,250

        e4_d4 = self.e4d4(e4)
        e4_d4 = self.e4d4_conv(e4_d4)  # 128,256,250  --> 128,32,250

        e5_d4 = self.e5d4(e5)  # 128,512,50  --> 128,512,250
        e5_d4 = self.e5d4_conv(e5_d4)  # 128,512,250  --> 128,32,250

        d4 = torch.cat([e1_d4, e2_d4, e3_d4, e4_d4, e5_d4], dim=1)  # 128,32,250  -->  128,160,250
        d4 = self.d_conv(d4)  # 128,160,250  --> 128,160,250

        """ d3 """
        e1_d3 = self.e1d3(e1)  # 128,32,2000  --> 128,32,500
        e1_d3 = self.e1d3_conv(e1_d3)  # 128,32,500  --> 128,32,500

        e2_d3 = self.e2d3(e2)  # 128,64,1000 --> 128,64,500
        e2_d3 = self.e2d3_conv(e2_d3)  # 128,64,500 --> 128,32,500

        e3_d3 = self.e3d3(e3)
        e3_d3 = self.e3d3_conv(e3_d3)  # 128,128,500  --> 128,128,500

        e4_d3 = self.e4d3(e4)  # 128,256,250  --> 128,256,500
        e4_d3 = self.e4d3_conv(e4_d3)  # 128,256,500  --> 128,32,500

        e5_d3 = self.e5d3(e5)  # 128,512,50  --> 128,512,500
        e5_d3 = self.e5d3_conv(e5_d3)  # 128,512,500  --> 128,32,500

        d3 = torch.cat([e1_d3, e2_d3, e3_d3, e4_d3, e5_d3], dim=1)  # 128,32,500  -->  128,160,500
        d3 = self.d_conv(d3)  # 128,160,500  --> 128,160,500

        """ d2 """
        e1_d2 = self.e1d2(e1)  # 128,32,2000  --> 128,32,1000
        e1_d2 = self.e1d2_conv(e1_d2)  # 128,32,1000  --> 128,32,1000

        e2_d2 = self.e2d2(e2)
        e2_d2 = self.e2d2_conv(e2_d2)  # 128,64,1000 --> 128,32,1000

        d3_d2 = self.d3d2(d3)  # 128,128,500  --> 128,128,1000
        d3_d2 = self.d3d2_conv(d3_d2)  # 128,128,1000  --> 128,32,1000

        e4_d2 = self.e4d2(e4)  # 128,256,250  --> 128,256,1000
        e4_d2 = self.e4d2_conv(e4_d2)  # 128,256,1000  --> 128,32,1000

        e5_d2 = self.e5d2(e5)  # 128,512,50  --> 128,512,1000
        e5_d2 = self.e5d2_conv(e5_d2)  # 128,512,1000  --> 128,32,1000

        d2 = torch.cat([e1_d2, e2_d2, d3_d2, e4_d2, e5_d2], dim=1)  # 128,32,1000  -->  128,160,1000
        d2 = self.d_conv(d2)  # 128,160,1000  --> 128,160,1000

        """ d1 """
        e1_d1 = self.e1d1(e1)
        e1_d1 = self.e1d1_conv(e1_d1)  # 128,32,2000  --> 128,32,2000

        d2_d1 = self.d2d1(d2)  # 128,128,1000  --> 128,128,2000
        d2_d1 = self.d2d1_conv(d2_d1)  # 128,128,2000 --> 128,32,2000

        d3_d1 = self.d3d1(d3)  # 128,128,500  --> 128,128,2000
        d3_d1 = self.d3d1_conv(d3_d1)  # 128,128,2000  --> 128,32,2000

        e4_d1 = self.e4d1(e4)  # 128,256,250  --> 128,256,2000
        e4_d1 = self.e4d1_conv(e4_d1)  # 128,256,2000  --> 128,32,2000

        e5_d1 = self.e5d1(e5)  # 128,512,50  --> 128,512,2000
        e5_d1 = self.e5d1_conv(e5_d1)  # 128,512,2000  --> 128,32,2000

        d1 = torch.cat([e1_d1, d2_d1, d3_d1, e4_d1, e5_d1], dim=1)  # 128,32,2000  -->  128,160,2000
        d1 = self.d_conv(d1)  # 128,160,2000  -->  128,160,2000

        out = self.out_conv(d1)
        out, _ = torch.max(out, dim=2)
        out = self.out_fc(out)

        return out