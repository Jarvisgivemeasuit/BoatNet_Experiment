import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from torchsummary import summary

from . import torchvision_resnet
from .boat_resunet_utils import initialize_weights
# import torchvision_resnet
# from boat_resunet_utils import *
import torch.nn.functional as F

BACKBONE = 'resnet50'


class ResDown(nn.Module):
    def __init__(self, backbone=BACKBONE, in_channels=3, pretrained=True,
                 zero_init_residual=False):
        super(ResDown, self).__init__()
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

    def forward(self, x):
        x = self.layer0(x)
        output0 = x
        x = self.layer1(x)
        output1 = x
        x = self.layer2(x)
        output2 = x
        x = self.layer3(x)
        output3 = x
        output4 = self.layer4(x)

        return output0, output1, output2, output3, output4


class Pred_Fore_Rate(nn.Module):
    def __init__(self, inplanes, planes):
        super().__init__()
        self.de_ratio = ChDecrease(512, 32)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.double_conv = Double_conv(16, 16)

    def forward(self, x):
        x = self.de_ratio(x)
        x = self.double_conv(x)
        x = self.pool(x)
        x = x.reshape(x.shape[0], x.shape[1])

        return x


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


class Up(nn.Module):
    def __init__(self, u_inplanes, d_inplanes, d_planes, bilinear=False, last_cat=False):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(u_inplanes, u_inplanes, 2, stride=2)
            # self.up = nn.ConvTranspose2d(512, 512, 2, stride=8)
        self.conv = Double_conv(d_inplanes, d_planes)
        self.last_cat = last_cat

    def forward(self, x1, x2):
        if not self.last_cat:
            x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class ChDecrease(nn.Module):
    def __init__(self, inplanes, times):
        super().__init__()
        self.conv1x1 = nn.Conv2d(inplanes, inplanes // times, kernel_size=1)

    def forward(self, x):
        x = self.conv1x1(x)
        return x


class Dy_UNet(nn.Module):
    def __init__(self, inplanes, num_classes, backbone):
        super().__init__()
        self.down = ResDown(in_channels=inplanes, backbone=backbone)
        self.backbone = backbone
        self.num_classes = num_classes

        if self.backbone not in ['resnet18', 'resnet34']:
            self.de1 = ChDecrease(256, 4)
            self.de2 = ChDecrease(512, 4)
            self.de3 = ChDecrease(1024, 4)
            self.de4 = ChDecrease(2048, 4)

        self.fore_pred = Pred_Fore_Rate(512, self.num_classes)

        self.up1 = Up(512, 768, 256)
        self.up2 = Up(256, 384, 128)
        self.up3 = Up(128, 192, 64)
        self.up4 = Up(64, 128, 64, last_cat=True)
        self.up5 = Up(64, 68, 64)

        self.outconv = Double_conv(64, self.num_classes)

    def forward(self, x):
        ori_x = x
        x0, x1, x2, x3, x4 = self.down(x)

        if self.backbone not in ['resnet18', 'resnet34']:
            x1 = self.de1(x1)
            x2 = self.de2(x2)
            x3 = self.de3(x3)
            x4 = self.de4(x4)
        # print(x0.shape, x1.shape, x2.shape, x3.shape, x4.shape)

        x4_ = x4.detach()
        ratios = self.fore_pred(x4_).float()

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        x = self.up5(x, ori_x)

        output = self.outconv(x)

        return ratios, output


# net = Boat_UNet(4, 16, 'resnet50')
# summary(net.cuda(), (4, 256, 256))

# net = Boat_UNet(4, 16, 'resnet50', 'resnet18')
# summary(net.cuda(), (4, 256, 256))

# net = Boat_UNet(4, 16, 'resnet50', 'resnet18').cuda()
# test_data = torch.randn([2, 4, 256, 256]).cuda()
# aa, bb, cc = net(test_data)
# print(aa.shape, bb.shape, cc.shape)