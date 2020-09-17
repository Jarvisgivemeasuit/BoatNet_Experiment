import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from torchsummary import summary
from thop import profile

from . import torchvision_resnet
from .nonlocal_utils import initialize_weights
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

        self.change_dilation([1, 1, 1, 2, 4])

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

    def change_dilation(self, params):
        assert isinstance(params, (tuple, list))
        assert len(params) == 5
        self._change_stage_dilation(self.layer0, params[0])
        self._change_stage_dilation(self.layer1, params[1])
        self._change_stage_dilation(self.layer2, params[2])
        self._change_stage_dilation(self.layer3, params[3])
        self._change_stage_dilation(self.layer4, params[4])

    def _change_stage_dilation(self, stage, param):
        for m in stage.modules():
            if isinstance(m, nn.Conv2d):
                if m.kernel_size == (3, 3):
                    m.padding = (param, param)
                    m.dilation = (param, param)


class NonLocal(nn.Module):
    def __init__(self, in_channels, classes):
        super().__init__()
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, classes, 3, 1, 1),
            nn.BatchNorm2d(classes),
            nn.LeakyReLU(inplace=True)
        )
        self.phi = nn.Sequential(
            nn.Conv2d(classes, classes, 3, 1, 1),
            nn.BatchNorm2d(classes),
            nn.LeakyReLU(inplace=True)
        )
        self.theta = nn.Sequential(
            nn.Conv2d(classes, classes, 3, 1, 1),
            nn.BatchNorm2d(classes),
            nn.LeakyReLU(inplace=True)
        )
        self.gamma = nn.Sequential(
            nn.Conv2d(classes, classes, 3, 1, 1),
            nn.BatchNorm2d(classes),
            nn.LeakyReLU(inplace=True)
        )
        self.conv = nn.Conv2d(classes, classes, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.input_conv(x)
        phi = self.phi(x)
        theta = self.theta(x)
        gamma = self.gamma(x)

        phi = phi.reshape(phi.shape[0], phi.shape[1], -1)
        theta = theta.reshape(theta.shape[0], theta.shape[1], -1).permute(0, 2, 1)
        gamma = gamma.reshape(gamma.shape[0], gamma.shape[1], -1).permute(0, 2, 1)

        fc = self.softmax(torch.bmm(theta, phi))
        y = torch.bmm(fc, gamma).permute(0, 2, 1).contiguous()
        out = y.reshape(phi.shape[0], phi.shape[1], x.shape[2], x.shape[3])
        out = self.conv(out) + x

        return out


class Double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, inplanes, planes):
        super(Double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inplanes, planes, 3, padding=1),
            nn.BatchNorm2d(planes),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(planes, planes, 3, padding=1),
            nn.BatchNorm2d(planes),
            nn.LeakyReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class NonLocalNN(nn.Module):
    def __init__(self, inplanes, num_classes, backbone):
        super().__init__()
        self.backbone = Resnet(backbone, inplanes)
        self.nl = NonLocal(512, 16)
        self.classes_conv = nn.Sequential(
            nn.Conv2d(512, num_classes, 3, padding=1),
            nn.BatchNorm2d(num_classes),
            nn.ReLU(inplace=True))

        self.out_conv = nn.Conv2d(num_classes, num_classes, 3, padding=1, bias=True)
        
    def forward(self, x):
        size = (x.shape[2], x.shape[3])
        x = self.backbone(x)
        x = self.nl(x)

        x = F.interpolate(
            x,
            size=size,
            mode='bilinear',
            align_corners=True
        )
        out = self.out_conv(x)
        return out

    def freeze_backbone(self):
        for param in self.backbone.layer1.parameters():
            param.requires_grad = False
        for param in self.backbone.layer2.parameters():
            param.requires_grad = False
        for param in self.backbone.layer3.parameters():
            param.requires_grad = False
        for param in self.backbone.layer4.parameters():
            param.requires_grad = False

    def train_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

# net = PSPNet(4, 16, 'resnet50', False, False).cuda()
# summary(net.cuda(), (4, 256, 256))