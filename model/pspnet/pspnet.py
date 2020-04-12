import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from torchsummary import summary

from . import torchvision_resnet
from .pspnet_utils import initialize_weights
# import torchvision_resnet
# from pspnet_utils import *
import torch.nn.functional as F

BACKBONE = 'resnet50'
NUM_CLASSES = 16


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

        if not pretrained:
            initialize_weights(self)
            for m in self.modules():
                if isinstance(m, torchvision_resnet.Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, torchvision_resnet.BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        self.chd = ChDecrease(2048, 4)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        output = self.chd(x)

        return output


class PPM(nn.Module):
    def __init__(self, inplanes):
        super().__init__()
        planes = int(inplanes / 4)
        self.ppm1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, planes, 1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
        )
        self.ppm2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Conv2d(inplanes, planes, 1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
        )
        self.ppm3 = nn.Sequential(
            nn.AdaptiveAvgPool2d((3, 3)),
            nn.Conv2d(inplanes, planes, 1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
        )
        self.ppm4 = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Conv2d(inplanes, planes, 1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        size = (x.shape[2], x.shape[3])
        ppm1 = F.interpolate(self.ppm1(x),
                             size=size,
                             mode='bilinear',
                             align_corners=True)
        ppm2 = F.interpolate(self.ppm2(x),
                             size=size,
                             mode='bilinear',
                             align_corners=True)
        ppm3 = F.interpolate(self.ppm3(x),
                             size=size,
                             mode='bilinear',
                             align_corners=True)
        ppm4 = F.interpolate(self.ppm4(x),
                             size=size,
                             mode='bilinear',
                             align_corners=True)
        ppm = torch.cat([x, ppm1, ppm2, ppm3, ppm4], dim=1)
        return ppm


class Double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, inplanes, planes):
        super(Double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inplanes, planes, 3, padding=1),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, 3, padding=1),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


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


class PSPNet(nn.Module):
    def __init__(self, inplanes, num_classes, backbone, use_threshold, use_gcn):
        super().__init__()
        self.backbone = Resnet(backbone, inplanes)
        self.ppm = PPM(512)
        self.ppm_conv = nn.Sequential(
            Double_conv(1024, 512),
            nn.Dropout(p=0.1),
        )
        self.ratios = Pred_Fore_Rate()
        self.out_conv = nn.Conv2d(512, num_classes, 1, bias=False)
        self.use_threshold = use_threshold

    def forward(self, x):
        x = self.backbone(x)
        if self.use_threshold:
            output_ratios = self.ratios(x)

        x = self.ppm(x)
        x = self.ppm_conv(x)
        x = self.out_conv(x)

        out = F.interpolate(x,
                            scale_factor=16,
                            mode='bilinear',
                            align_corners=True)

        return out, output_ratios


# net = Dt_PSPNet(4, 16, 'resnet50', False, False)
# summary(net.cuda(), (4, 256, 256))