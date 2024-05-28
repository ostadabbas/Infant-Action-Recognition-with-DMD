# The based unit of graph convolutional networks.

import torch
import torch.nn as nn

class ConvTemporalGraphical(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, 
                 temporal_kernel_size=1, temporal_stride=1, temporal_padding=0, temporal_dilation=1, bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size = (temporal_kernel_size, 1),
            padding = (temporal_padding, 0),
            stride = (temporal_stride, 1),
            dilation = (temporal_dilation, 1),
            bias=bias)

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size, f"{A.size(0)}, {self.kernel_size}"

        x = self.conv(x)

        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))

        return x.contiguous(), A