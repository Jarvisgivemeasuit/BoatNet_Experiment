import torch.nn as nn
import torch
import torchvision_resnet
import numpy as np
from torchsummary import summary

from boat_resunet_utils import initialize_weights
import torch.nn.functional as F

BACKBONE = 'resnet50'


class ResDown(nn.Module):
    def __init__(self, backbone=BACKBONE, in_channels=3, pretrained=True,
                 zero_init_residual=False):
        super(ResDown, self).__init__()
        model = getattr(torchvision_resnet, backbone)(pretrained)
        if in_channels != 3:
            self.layer0 = nn.Sequential(
                nn.Conv2d(in_channels, 64, 3, stride=1, padding=1, bias=False),
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
        self.conv1x1 = nn.Conv2d(inplanes, planes, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1x1(x)
        x = self.pool(x)

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
    def __init__(self, inplanes, planes, bilinear=False, last_cat=False):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(inplanes // 3 * 2, inplanes // 3 * 2, 2, stride=2)
            # self.up = nn.ConvTranspose2d(512, 512, 2, stride=8)
        self.conv = Double_conv(inplanes, planes)
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
    def __init__(self, inplanes):
        super().__init__()
        self.conv1x1 = nn.Conv2d(inplanes, inplanes // 4, kernel_size=1)

    def forward(self, x):
        x = self.conv1x1(x)
        return x


class Boat_UNet_Part1(nn.Module):
    def __init__(self, inplanes, num_classes, backbone):
        super().__init__()
        self.down = ResDown(in_channels=inplanes, backbone=backbone)
        self.backbone = backbone

        if self.backbone not in ['resnet18', 'resnet34']:
            self.de1 = ChDecrease(256)
            self.de2 = ChDecrease(512)
            self.de3 = ChDecrease(1024)
            self.de4 = ChDecrease(2048)

        self.fore_pred = Pred_Fore_Rate(512, 1)

        self.up1 = Up(768, 256)
        self.up2 = Up(384, 128)
        self.up3 = Up(192, 64)
        self.up4 = Up(128, 64, last_cat=True)
        self.outconv = nn.Conv2d(64, num_classes, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0, x1, x2, x3, x4 = self.down(x)

        # print(x0.shape, x1.shape, x2.shape, x3.shape, x4.shape)
        if self.backbone not in ['resnet18', 'resnet34']:
            x1 = self.de1(x1)
            x2 = self.de2(x2)
            x3 = self.de3(x3)
            x4 = self.de4(x4)

        rate = self.fore_pred(x4)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)

        fore_output = self.outconv(x)
        fore_output_ = self.sigmoid(fore_output)
        fore_feature = (fore_output_ > (1 - rate)).byte()

        fore_size = fore_feature[0].size()[0] * fore_feature[0].size()[1] * fore_feature[0].size()[2]
        pred_rate = fore_feature.sum(dim=(1, 2, 3)) / fore_size
        
        output = torch.cat([x, fore_feature], dim=1)

        return fore_output, pred_rate, output


class Boat_UNet_Part2(nn.Module):
    def __init__(self, inplanes, num_classes, backbone):
        super().__init__()
        self.down = ResDown(in_channels=inplanes, backbone=backbone)
        self.backbone = backbone

        if self.backbone not in ['resnet18', 'resnet34']:
            self.de1 = ChDecrease(256)
            self.de2 = ChDecrease(512)
            self.de3 = ChDecrease(1024)
            self.de4 = ChDecrease(2048)

        self.up1 = Up(768, 256)
        self.up2 = Up(384, 128)
        self.up3 = Up(192, 64)
        self.up4 = Up(128, 64, last_cat=True)
        self.outconv = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        x0, x1, x2, x3, x4 = self.down(x)
        # print(x0.shape, x1.shape, x2.shape, x3.shape, x4.shape)
        if self.backbone not in ['resnet18', 'resnet34']:

            x1 = self.de1(x1)
            x2 = self.de2(x2)
            x3 = self.de3(x3)
            x4 = self.de4(x4)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        output = self.outconv(x)

        return output


class Boat_UNet(nn.Module):
    def __init__(self, inplanes, num_classes, backbone1, backbone2):
        super().__init__()
        self.part1 = Boat_UNet_Part1(inplanes, 1, backbone1)
        self.part2 = Boat_UNet_Part2(65, num_classes, backbone2)

    def forward(self, x):
        fore_output, pred_rate, x = self.part1(x)
        output = self.part2(x)

        return fore_output, pred_rate, output


# net = Boat_UNet_Part1(4, 1, 'resnet50')
# summary(net.cuda(), (4, 256, 256))

# net = Boat_UNet_Part2(65, 16, 'resnet18')
# summary(net.cuda(), (65, 256, 256))