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

class UNet_Small(nn.Module):
    def __init__(self, n_class, out_channels, kernel_size, downsize, use_reverse=None):
        super(UNet_Small, self).__init__()

        # self.channels = [out_channels+2**(i+3) for i in range(num_layers)]
        # self.channels = [out_channels,out_channels+16,out_channels+32,out_channels+64,out_channels+96,out_channels+128]
        self.use_reverse = use_reverse
        in_channels = 4
        if self.use_reverse:
            self.conv = nn.Sequential(
                nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, padding=(kernel_size-1)//2),
                nn.BatchNorm1d(in_channels))
        
        self.channels = [out_channels,out_channels*2,out_channels*3,out_channels*4,out_channels*5,out_channels*6]
        self.uplblocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(in_channels, self.channels[0], stride=downsize[0], kernel_size=kernel_size, padding=(kernel_size-1)//2), nn.BatchNorm1d(self.channels[0])
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

    def forward(self, distal_input):
        """Forward propagation of a batch."""
        out = distal_input
        if self.use_reverse:
            out = torch.add(self.conv(out), self.conv(out.flip([1, 2])).flip([2]))

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
    
    def reverse_input(self,distal_input):
        return distal_input.flip([1, 2])