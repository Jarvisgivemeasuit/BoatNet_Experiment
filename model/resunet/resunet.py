import torch.nn as nn
import torch
from . import torchvision_resnet
from .resunet_utils import initialize_weights
import torch.nn.functional as F

BACKBONE = 'resnet50'

class ResDown(nn.Module):
    def __init__(self, backbone=BACKBONE, in_channels=3, pretrained=True,
                 zero_init_residual=False):
        super(ResDown, self).__init__()
        model = getattr(torchvision_resnet, BACKBONE)(pretrained)
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

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
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
        # return output4


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

class UNet(nn.Module):
    def __init__(self, inplanes, num_classes, backbone):
        super().__init__()
        self.down = ResDown(in_channels=inplanes, backbone=backbone)
        self.list = None

        if not (BACKBONE == 'resnet18' or BACKBONE == 'resnet34'):
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
        self.x0, self.x1, self.x2, self.x3, self.x4 = self.down(x)
        # print(self.x0.shape, self.x1.shape, self.x2.shape, self.x3.shape, self.x4.shape)
        if not (BACKBONE == 'resnet18' or BACKBONE == 'resnet34'):
            self.x1 = self.de1(self.x1)
            self.x2 = self.de2(self.x2)
            self.x3 = self.de3(self.x3)
            self.x4 = self.de4(self.x4)

        x = self.up1(self.x4, self.x3)
        x = self.up2(x, self.x2)
        x = self.up3(x, self.x1)
        x = self.up4(x, self.x0)

        x = self.outconv(x)

        return x


class RatioNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = ResDown(in_channels=4)
        self.conv = nn.Sequential(nn.Conv2d(2048, 16, 1),
                                  nn.Conv2d(16, 16, 3, 1),
                                  nn.Conv2d(16, 16, 3, 1))
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        _1, _2, _3, _4, output = self.down(x)
        output = self.conv(output)
        output = self.pool(output)
        return output