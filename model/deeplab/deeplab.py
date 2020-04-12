import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from torchsummary import summary

from . import torchvision_resnet
from .deeplab_utils import initialize_weights
# import torchvision_resnet
# from deeplab_utils import *
import torch.nn.functional as F

BACKBONE = 'resnet50'
NUM_CLASSES = 16


def conv3x3(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)


class ChDecrease(nn.Module):
    def __init__(self, inplanes, times):
        super().__init__()
        self.conv1x1 = nn.Conv2d(inplanes, inplanes // times, kernel_size=1)

    def forward(self, x):
        x = self.conv1x1(x)
        return x


class Resnet(nn.Module):
    def __init__(self, backbone=BACKBONE, in_channels=3, pretrained=True,
                 zero_init_residual=False):
        super().__init__()
        model = getattr(torchvision_resnet, backbone)(pretrained)
        if in_channels != 3:
            self.layer0 = nn.Sequential(
                nn.Conv2d(in_channels, 64, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(inplace=True),
                # nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
            initialize_weights(self.layer0)
        else:
            self.layer0 = nn.Sequential(*list(model.children())[:4])

        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        for m in self.layer4.modules():
            if isinstance(m, nn.Conv2d):
                if m.kernel_size == (3, 3):
                    m.padding = (2, 2)
                    m.dilation = (2, 2)

        if not pretrained:
            initialize_weights(self)
            for m in self.modules():
                if isinstance(m, torchvision_resnet.Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, torchvision_resnet.BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        self.chd = ChDecrease(2048, 4)

    def forward(self, x):
        low_level_features = self.layer0(x)
        x = self.layer1(low_level_features)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        output = self.chd(x)

        return output, low_level_features


class ASPP(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.aspp1 = nn.Sequential(
            nn.Conv2d(in_ch, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.aspp2 = nn.Sequential(
            nn.Conv2d(in_ch, 256, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.aspp3 = nn.Sequential(
            nn.Conv2d(in_ch, 256, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.aspp4 = nn.Sequential(
            nn.Conv2d(in_ch, 256, 3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.aspp5 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_ch, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.aspp_out = nn.Sequential(
            nn.Conv2d(1280, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        size = (x.shape[2], x.shape[3])
        aspp1 = self.aspp1(x)
        aspp2 = self.aspp2(x)
        aspp3 = self.aspp3(x)
        aspp4 = self.aspp4(x)
        aspp5 = F.interpolate(self.aspp5(x),
                              size=size,
                              mode='bilinear',
                              align_corners=True)
        all_aspp = torch.cat((aspp1, aspp2, aspp3, aspp4, aspp5), dim=1)
        return self.aspp_out(all_aspp)


class Pred_Fore_Rate(nn.Module):
    def __init__(self):
        super().__init__()
        self.de_ratio = ChDecrease(512, 32)
        self.conv = nn.Sequential(
            nn.Conv2d(16, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.de_ratio(x)
        x = self.conv(x)
        x = self.pool(x)
        x = x.reshape(x.shape[0], x.shape[1])

        return x


class DeepLabV3Plus(nn.Module):
    def __init__(self, inplanes=4, num_classes=16, backbone=BACKBONE, use_threshold=False, use_gcn=False):
        assert backbone in [
            'vgg16', 'resnet34', 'resnet50', 'se_resnet34', 'se_resnet50',
            'mobilenet_v1', 'mobilenet_v2', 'xception'
        ]
        super().__init__()
        self.backbone = Resnet(backbone, inplanes)
        self.aspp = ASPP(512)
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(64, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.concat_conv = nn.Sequential(
            conv3x3(304, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            conv3x3(256, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1, bias=False),
        )
        self.ratios = Pred_Fore_Rate()
        self.use_threshold = use_threshold

    def forward(self, x):
        x, low_level_features = self.backbone(x)
        if self.use_threshold:
            output_ratios = self.ratios(x)
        aspp_out = self.aspp(x)
        aspp_out = F.interpolate(aspp_out,
                                 scale_factor=8,
                                 mode='bilinear',
                                 align_corners=True)
        low_level_features = self.low_level_conv(low_level_features)
        out = torch.cat((aspp_out, low_level_features), dim=1)
        out = self.concat_conv(out)
        out = F.interpolate(out,
                            scale_factor=2,
                            mode='bilinear',
                            align_corners=True)
        return out, output_ratios

    def reset_classes(self, num_classes):
        self.num_classes = num_classes
        self.concat_conv[-1] = nn.Conv2d(256, num_classes, 1, bias=False)

# net = DeepLabV3Plus()
# summary(net.cuda(), (4, 256, 256))