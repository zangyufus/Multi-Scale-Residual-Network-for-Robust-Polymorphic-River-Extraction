import torch
import torch.nn as nn
from .mobilevit import MobileViTBlock
from torch.nn import functional as F
from functools import partial
from torch.nn.functional import upsample as Up

nonlinearity = partial(F.relu, inplace=True)

def shortcut(in_channels, out_channels, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False),
        nn.BatchNorm2d(out_channels)
    )

class Shortcut_depth4(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Shortcut_depth4, self).__init__()
        if in_channels != out_channels:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
            self.bn = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)
        else:
            self.conv = nn.Identity()
            self.bn = nn.Identity()
            self.relu = nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class DUC(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor=2):
        super(DUC, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pixel_shuffle(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True, dilation=1, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity
        self.dropout = nn.Dropout2d(p=dropout)  # 添加 Dropout

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        # # 判断是否为训练模式，只有在训练时才应用 Dropout
        # if self.training:  # 判断当前是否处于训练模式
        #     x = self.dropout(x)  # 训练时应用 Dropout
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.hdc = nn.Sequential(
                ConvBlock(in_channels, out_channels, padding=1, dilation=1),
                ConvBlock(out_channels, out_channels, padding=2, dilation=2),
                ConvBlock(out_channels, out_channels, padding=5, dilation=5, with_nonlinearity=False)
            )
        self.shortcut = shortcut(in_channels, out_channels) if in_channels != out_channels else nn.Identity()
        self.relu = nn.ReLU()
        self.se = SE_Block(c=out_channels)

    def forward(self, x):
        res = self.shortcut(x)
        x   = self.se(self.hdc(x))
        x   = self.relu(res + x)
        return x

class DownBlockwithVit(nn.Module):
    def __init__(self, in_channels, out_channels, dim, L, kernel_size=3, patch_size=(4,4)):
        super().__init__()
        self.downsample = nn.MaxPool2d(2,2)
        self.convblock  = ResidualBlock(in_channels, out_channels)
        self.vitblock   = MobileViTBlock(dim, L, out_channels, kernel_size, patch_size, int(dim*2))

    def forward(self, x):
        x = self.downsample(x)
        x = self.convblock(x)
        x = self.vitblock(x)
        return x

class Bridge(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.downsample = nn.MaxPool2d(2,2)
        self.bridge = ResidualBlock(in_channels, out_channels)

    def forward(self, x):
        return self.bridge(self.downsample(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = DUC(in_channels, in_channels*2)
        self.residualblock = ResidualBlock(in_channels, out_channels)

    def forward(self, up_x, down_x):
        x = self.upsample(up_x)
        x = torch.cat([x, down_x], 1) 
        x = self.residualblock(x)
        return x

class UpBlock_depth_4(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = DUC(in_channels, 516)
        self.residualblock = ResidualBlock(in_channels + 516, out_channels)
        self.shortcut = Shortcut_depth4(in_channels, 516)

    def forward(self, up_x, down_x):
        x = self.upsample(up_x)
        up_x_shortcut = self.shortcut(up_x)
        x = torch.cat([x, down_x], 1)
        x = self.residualblock(x)
        
        return x

class SE_Block(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        x = x * y.expand_as(x)
        return x

class MSDC(nn.Module):
    def __init__(self, channel, features, out_features=512, sizes=(1, 2, 3, 6)):
        super(MSDC, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList([self.create(features, size) for size in sizes])
        self.bottle = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        self.pooling = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def create(self, features, size):
        pool = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(pool, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        fusion = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottle(torch.cat(fusion, 1))
        x = self.relu(bottle)
        dilate1_out = nonlinearity(self.conv1x1(self.dilate1(x)))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate4(self.dilate3(self.dilate2(self.dilate1(x))))))
        dilate5_out = nonlinearity(self.conv1x1(self.dilate5(self.dilate4(self.dilate3(self.dilate2(self.dilate1(x)))))))
        dilate6_out = self.pooling(x)
        
        out_feature = dilate1_out + dilate2_out + dilate3_out + dilate4_out + dilate5_out + dilate6_out
        
        return out_feature
    
class MKMP(nn.Module):
    def __init__(self, channels):
        super(MKMP, self).__init__()
        self.maxpool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)
        self.maxpool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)
        self.maxpool4 = nn.MaxPool2d(kernel_size=[6, 6], stride=6)
        
        self.conv2d = nn.Conv2d(in_channels=channels, out_channels=1, kernel_size=1, padding=0)
        self.reduce_channels = nn.Conv2d(in_channels=516, out_channels=512, kernel_size=1)

    def forward(self, x):
        height = x.size(2)
        width = x.size(3)
        
        self.subpart1 = self.conv2d(self.maxpool1(x))
        self.sub1 = Up(self.subpart1, size=(height, width), mode='bilinear', align_corners=False)
        
        self.subpart2 = self.conv2d(self.maxpool2(x))
        self.sub2 = Up(self.subpart2, size=(height, width), mode='bilinear', align_corners=False)
        
        self.subpart3 = self.conv2d(self.maxpool3(x))
        self.sub3 = Up(self.subpart3, size=(height, width), mode='bilinear', align_corners=False)
        
        self.subpart4 = self.conv2d(self.maxpool4(x))
        self.sub4 = Up(self.subpart4, size=(height, width), mode='bilinear', align_corners=False)
        
        out_feature = torch.cat([self.sub1, self.sub2, self.sub3, self.sub4, x], 1)
        out_feature = self.reduce_channels(out_feature)
        
        return out_feature

# class BoundaryEnhancement(nn.Module):
#     def __init__(self, in_channels=2, out_channels=2):
#         super(BoundaryEnhancement, self).__init__()

#         self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)  # 提取边界信息
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(16, out_channels, kernel_size=1)  # 1x1 卷积，恢复到原通道数

#     def forward(self, x):
#         x = self.conv1(x)  # 3x3 卷积
#         x = self.relu(x)
#         x = self.conv2(x)  # 3x3 卷积
#         x = self.relu(x)
#         x = self.conv3(x)  # 1x1 卷积
#         return x

class Net(nn.Module):
    DEPTH = 4
        
    def __init__(self, n_classes=1, dims=[144, 240, 320]):
        super().__init__()

        self.n_classes = n_classes
        # self.boundary = BoundaryEnhancement(in_channels=2, out_channels=2)

        down_blocks = []
        up_blocks = []

        down_blocks.append(ResidualBlock(in_channels=16, out_channels=32))
        # down_blocks.append(ConvBlock(in_channels=16, out_channels=32))
        down_blocks.append(DownBlockwithVit(in_channels=32, out_channels=64, dim=dims[0], L=2)) 
        down_blocks.append(DownBlockwithVit(in_channels=64, out_channels=128, dim=dims[1], L=4))
        down_blocks.append(DownBlockwithVit(in_channels=128, out_channels=256, dim=dims[2], L=3))

        self.down_blocks = nn.ModuleList(down_blocks)

        self.bridge = Bridge(256, 512)
        self.msdc = MSDC(channel=512, features=512)
        self.mkmp = MKMP(channels=512)

        up_blocks.append(UpBlock(in_channels=256*2, out_channels=256))
        up_blocks.append(UpBlock(in_channels=128*2, out_channels=128))
        up_blocks.append(UpBlock(in_channels=64*2, out_channels=64))
        up_blocks.append(UpBlock(in_channels=32*2, out_channels=32))

        self.up_blocks = nn.ModuleList(up_blocks)

        self.se = SE_Block(c=32, r=4)

        self.out = nn.Conv2d(32, n_classes, kernel_size=1, stride=1)

    def forward(self, x, istrain=False):
        stages = dict()
        stages[f"layer_0"] = x

        # encoder
        for i, block in enumerate(self.down_blocks, 1):
            x = block(x)
            stages[f"layer_{i}"] = x

        x = self.bridge(x)
        x = self.msdc(x)
        x = self.mkmp(x)

        # decoder
        for i, block in enumerate(self.up_blocks, 1):
            key = f"layer_{Net.DEPTH + 1 - i}"
            x = block(x, stages[key])

        x = self.se(x)
        x = self.out(x)
        del stages

        if istrain:
            return x, None
        else:
            return x
