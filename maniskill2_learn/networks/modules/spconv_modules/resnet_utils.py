import torch.nn as nn
import torchsparse
import torchsparse.nn as spnn


def build_sparse_norm(channels, use_ln=True):
    return spnn.LayerNorm(channels, eps=1e-6) if use_ln else spnn.BatchNorm(channels)


class BasicConvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, use_ln=False):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, transposed=False),
            build_sparse_norm(out_channels, use_ln),
            spnn.ReLU(True),
        )

    def forward(self, x):
        return self.net(x)


class BasicDeconvolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, use_ln=False):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, transposed=True),
            build_sparse_norm(out_channels, use_ln),
            spnn.ReLU(True),
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, use_ln=False):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation, stride=stride),
            build_sparse_norm(out_channels, use_ln),
            spnn.ReLU(True),
            spnn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, dilation=dilation, stride=1),
            build_sparse_norm(out_channels, use_ln),
        )

        if in_channels == out_channels * self.expansion and stride == 1:
            self.downsample = nn.Sequential()
        else:
            if stride == 1:
                self.downsample = nn.Sequential(
                    spnn.Conv3d(in_channels, out_channels, kernel_size=1, dilation=1, stride=stride),
                    build_sparse_norm(out_channels, use_ln),
                )
            else:
                self.downsample = nn.Sequential(
                    spnn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, dilation=1, stride=stride),
                    build_sparse_norm(out_channels, use_ln),
                )

        self.relu = spnn.ReLU(True)

    def forward(self, x):
        return self.relu(self.net(x) + self.downsample(x))


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, use_ln=False):
        super(Bottleneck, self).__init__()

        self.net = nn.Sequential(
            spnn.Conv3d(in_channels, out_channels, kernel_size=1),
            build_sparse_norm(out_channels, use_ln),
            spnn.ReLU(True),
            spnn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, dilation=dilation, stride=stride),
            build_sparse_norm(out_channels, use_ln),
            spnn.ReLU(True),
            spnn.Conv3d(out_channels, out_channels * self.expansion, kernel_size=1),
            build_sparse_norm(out_channels * self.expansion, use_ln),
        )

        if in_channels == out_channels * self.expansion and stride == 1:
            self.downsample = nn.Sequential()
        else:
            if stride == 1:
                self.downsample = nn.Sequential(
                    spnn.Conv3d(in_channels, out_channels * self.expansion, kernel_size=1, dilation=1, stride=stride),
                    build_sparse_norm(out_channels * self.expansion, use_ln),
                )
            else:
                self.downsample = nn.Sequential(
                    spnn.Conv3d(in_channels, out_channels * self.expansion, kernel_size=kernel_size, dilation=1, stride=stride),
                    build_sparse_norm(out_channels * self.expansion, use_ln),
                )

        self.relu = spnn.ReLU(True)

    def forward(self, x):
        return self.relu(self.net(x) + self.downsample(x))
