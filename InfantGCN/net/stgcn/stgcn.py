import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from net.stgcn.tgcn import ConvTemporalGraphical
from graphs import Graph

class STGCNBlock(nn.Module):

    def __init__(self,in_channels, out_channels, temporal_kernel_size, kernel_size, stride=1, dropout=0, residual=True):
        super(STGCNBlock, self).__init__()

        assert temporal_kernel_size % 2 == 1
        padding = ((temporal_kernel_size - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size)

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (temporal_kernel_size, 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res

        return self.relu(x), A
    

class STGCN(nn.Module):
    def __init__(self, in_channels, num_class, graph_args, edge_importance_weighting, **kwargs):
        super().__init__()

        # load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList((
            STGCNBlock(in_channels, 64, temporal_kernel_size, spatial_kernel_size, 1, residual=False, **kwargs0),
            STGCNBlock(64, 64, temporal_kernel_size, spatial_kernel_size, 1, **kwargs),
            STGCNBlock(64, 64, temporal_kernel_size, spatial_kernel_size, 1, **kwargs),
            STGCNBlock(64, 64, temporal_kernel_size, spatial_kernel_size, 1, **kwargs),
            STGCNBlock(64, 128, temporal_kernel_size, spatial_kernel_size, 2, **kwargs),
            STGCNBlock(128, 128, temporal_kernel_size, spatial_kernel_size, 1, **kwargs),
            STGCNBlock(128, 128, temporal_kernel_size, spatial_kernel_size, 1, **kwargs),
            STGCNBlock(128, 256, temporal_kernel_size, spatial_kernel_size, 2, **kwargs),
            STGCNBlock(256, 256, temporal_kernel_size, spatial_kernel_size, 1, **kwargs),
            STGCNBlock(256, 256, temporal_kernel_size, spatial_kernel_size, 1, **kwargs),
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # fcn for prediction
        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)

    def forward(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)

        # prediction
        x = self.fcn(x)
        x = x.view(x.size(0), -1)

        return x

    def extract_features(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        feats = x.view(N, M, -1, 1, 1).mean(dim=1)

        # prediction
        x = self.fcn(feats)

        feats = feats.view(feats.size(0), -1)
        x = x.view(x.size(0), -1)

        return x, feats