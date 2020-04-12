import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from torchsummary import summary

from . import torchvision_resnet
from .unet_utils import initialize_weights
# import torchvision_resnet
# from dt_unet_utils import *
import torch.nn.functional as F

BACKBONE = 'resnet50'
NUM_CLASSES = 16


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


class GCN(nn.Module):
    def __init__(self, inplanes, k=(7, 7)):
        super().__init__()
        self.conv_l1 = nn.Conv2d(inplanes, NUM_CLASSES, kernel_size=(k[0], 1), padding=(int((k[0]-1)/2), 0))
        self.conv_l2 = nn.Conv2d(NUM_CLASSES, NUM_CLASSES, kernel_size=(1, k[0]), padding=(0, int((k[0]-1)/2)))
        self.conv_r1 = nn.Conv2d(inplanes, NUM_CLASSES, kernel_size=(1, k[1]), padding=(0, int((k[1]-1)/2)))
        self.conv_r2 = nn.Conv2d(NUM_CLASSES, NUM_CLASSES, kernel_size=(k[1], 1), padding=(int((k[1]-1)/2), 0))

    def forward(self, x):
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)

        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)

        x = x_l + x_r

        return x


class BR(nn.Module):
    def __init__(self):
        super().__init__()
        self.res = nn.Sequential(
            nn.Conv2d(NUM_CLASSES, NUM_CLASSES, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(NUM_CLASSES, NUM_CLASSES, 3, padding=1)
        )

    def forward(self, x):
        x_res = self.res(x)
        return x + x_res


class Pred_Fore_Rate(nn.Module):
    def __init__(self):
        super().__init__()
        self.de_ratio = ChDecrease(2048, 128)
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


class Up_Gcn(nn.Module):
    def __init__(self, inplanes, bilinear=False, last_cat=False):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(inplanes, NUM_CLASSES, 2, stride=2)
        self.br = BR()
        self.last_cat = last_cat

    def forward(self, x1, x2):
        if not self.last_cat:
            x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        x = x1 + x2
        x = self.br(x)
        return x


class ChDecrease(nn.Module):
    def __init__(self, inplanes, times):
        super().__init__()
        self.conv1x1 = nn.Conv2d(inplanes, inplanes // times, kernel_size=1)

    def forward(self, x):
        x = self.conv1x1(x)
        return x


class Dt_UNet(nn.Module):
    def __init__(self, inplanes, num_classes, backbone, use_threshold, use_gcn):
        super().__init__()
        self.down = ResDown(in_channels=inplanes, backbone=backbone)
        self.backbone = backbone
        self.num_classes = num_classes
        self.use_threshold = use_threshold
        self.use_gcn = use_gcn

        if self.backbone not in ['resnet18', 'resnet34']:
            self.de1 = ChDecrease(256, 4)
            self.de2 = ChDecrease(512, 4)
            self.de3 = ChDecrease(1024, 4)
            self.de4 = ChDecrease(2048, 4)

        self.fore_pred = Pred_Fore_Rate()

        if self.use_gcn:
            self.gcn1 = GCN(512)
            self.gcn2 = GCN(256)
            self.gcn3 = GCN(128)
            self.gcn4 = GCN(64)

            self.br = BR()

            # self.up = Up_Gcn()
            # self.up1 = Up(NUM_CLASSES, 64 + NUM_CLASSES, 64)
            # self.up2 = Up(64, 68, 64)

            self.up1 = Up_Gcn(512 + NUM_CLASSES)
            self.up2 = Up_Gcn(256 + NUM_CLASSES)
            self.up3 = Up_Gcn(128 + NUM_CLASSES)
            self.up4 = Up(64 + NUM_CLASSES, 128 + NUM_CLASSES, 64)
            self.up5 = Up(64, 68, 64)
        else:
            self.up1 = Up(512, 768, 256)
            self.up2 = Up(256, 384, 128)
            self.up3 = Up(128, 192, 64)
            self.up4 = Up(64, 128, 64, last_cat=True)
            self.up5 = Up(64, 68, 64)

        self.outconv = Double_conv(64, self.num_classes)

    def forward(self, x):
        ori_x = x
        x0, x1, x2, x3, x4 = self.down(x)
        if self.use_threshold:
            ratios = self.fore_pred(x4).float()

        if self.backbone not in ['resnet18', 'resnet34']:
            x1 = self.de1(x1)
            x2 = self.de2(x2)
            x3 = self.de3(x3)
            x4 = self.de4(x4)

        if self.use_gcn:
            # x4 = self.br(self.gcn1(x4))
            # x3 = self.br(self.gcn2(x3))
            # x2 = self.br(self.gcn3(x2))
            # x1 = self.br(self.gcn4(x1))

            # x = self.up(x4, x3)
            # x = self.up(x, x2)
            # x = self.up(x, x1)
            # x = self.br(self.up1(x, x0))
            # x = self.br(self.up2(x, ori_x))

            x4 = torch.cat([x4, self.br(self.gcn1(x4))], dim=1)
            x = self.up1(x4, self.br(self.gcn2(x3)))
            x3 = torch.cat([x3, x], dim=1)
            x = self.up2(x3, self.br(self.gcn3(x2)))
            x2 = torch.cat([x2, x], dim=1)
            x = self.up3(x2, self.br(self.gcn4(x1)))
            x1 = torch.cat([x1, x], dim=1)
            x = self.up4(x1, x0)
            x = self.up5(x, ori_x)

        else:
            x = self.up1(x4, x3)
            x = self.up2(x, x2)
            x = self.up3(x, x1)
            x = self.up4(x, x0)
            x = self.up5(x, ori_x)

        output = self.outconv(x)

        return (output, ratios) if self.use_threshold else output

    def freeze_backbone(self):
        for param in self.down.parameters():
            param.requires_grad = False

    def train_backbone(self):
        for param in self.down.parameters():
            param.requires_grad = True


# net = Pred_Fore_Rate().cuda()
# summary(net.cuda(), (512, 16, 16))

# net = Dt_UNet(4, 16, 'resnet50', False).cuda()
# summary(net, (4, 256, 256))