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

class PPM(nn.Module):
    def __init__(self, inplanes):
        super().__init__()
        planes = int(inplanes / 4)
        self.ppm1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, planes, 1, bias=False),
            nn.BatchNorm2d(planes),
            nn.LeakyReLU(inplace=True),
        )
        self.ppm2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Conv2d(inplanes, planes, 1, bias=False),
            nn.BatchNorm2d(planes),
            nn.LeakyReLU(inplace=True),
        )
        self.ppm3 = nn.Sequential(
            nn.AdaptiveAvgPool2d((3, 3)),
            nn.Conv2d(inplanes, planes, 1, bias=False),
            nn.BatchNorm2d(planes),
            nn.LeakyReLU(inplace=True),
        )
        self.ppm4 = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Conv2d(inplanes, planes, 1, bias=False),
            nn.BatchNorm2d(planes),
            nn.LeakyReLU(inplace=True),
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
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(planes, planes, 3, padding=1),
            nn.BatchNorm2d(planes),
            nn.LeakyReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class Pred_Ratios(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(512, NUM_CLASSES, 3, padding=1),
            nn.BatchNorm2d(NUM_CLASSES),
            nn.LeakyReLU(inplace=True)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.conv(x)
        # fea_map = x
        x = self.pool(x)
        # x = x.reshape(x.shape[0], x.shape[1])

        return x

class Position_Weights(nn.Module):
    def __init__(self, inplanes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inplanes, NUM_CLASSES, 3, padding=1),
            nn.BatchNorm2d(NUM_CLASSES),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(NUM_CLASSES, 1, 3, padding=1),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(inplace=True))
        self.out_conv = nn.Conv2d(2, 1, 1)

    def forward(self, x, x_weights, feats):
        x = self.conv(x)
        x = torch.cat([x, x_weights], dim=1)
        x = self.out_conv(x)
        x = torch.sigmoid(x)
        return x * feats


class PSPNet(nn.Module):
    def __init__(self, inplanes, num_classes, backbone, use_threshold, use_gcn):
        super().__init__()
        self.backbone = Resnet(backbone, inplanes)
        self.ppm = PPM(512)
        self.ppm_conv = nn.Sequential(
            Double_conv(1024, 512),
            nn.Dropout(p=0.1),
        )
        self.classes_conv = nn.Sequential(
            nn.Conv2d(512, num_classes, 3, padding=1),
            nn.BatchNorm2d(num_classes),
            nn.ReLU(inplace=True))

        self.ratios = Pred_Ratios()
        self.sigmoid = nn.Sigmoid()

        self.out_conv = nn.Conv2d(num_classes, num_classes, 3, padding=1, bias=True)
        self.weight_conv = nn.Sequential(
            nn.Conv2d(num_classes, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(1, 1, 1)
            )

        self.position_conv = Position_Weights(4)
        self.use_threshold = use_threshold

    def forward(self, x):
        ori_x = x
        size = (x.shape[2], x.shape[3])
        x = self.backbone(x)
        if self.use_threshold:
            ratios = self.ratios(x)

            x = self.ppm(x)
            x = self.ppm_conv(x)
            x = self.classes_conv(x)

            x = F.interpolate(x,
                              size=size,
                              mode='bilinear',
                              align_corners=True)

            out = self.out_conv(x)

            x_weights = self.weight_conv(x)
            ratios_ = torch.sigmoid(ratios)
            # x_weights_ = x_weights.clone().detach()
            posi_feat = self.position_conv(ori_x, x_weights, out)

            output = out * ratios_ + posi_feat

            # ratios_ = ratios.clone().detach()
            # ratios_ = F.softmax(ratios, dim=1)

            # # output = out.clone().detach()
            # mask = torch.argmax(out1, dim=1)
            # output_ratios = torch.zeros([out1.shape[0], out1.shape[1]]).cuda()
            # for cat in range(NUM_CLASSES):
            #     output_ratios[:, cat] = (mask == cat).sum(dim=(1, 2)).float() / float(out1.shape[2] * out1.shape[3])
            # output_ratios = output_ratios.reshape(ratios.shape)

            # ratios_ = F.interpolate(ratios_,
            #                         size=size,
            #                         mode='nearest')
            # output_ratios = F.interpolate(output_ratios,
            #                         size=size,
            #                         mode='nearest')
            # dist = torch.log(output_ratios * torch.exp(ratios_).sum(dim=1).reshape(ratios_.shape[0], 1, ratios_.shape[2], ratios_.shape[3]).expand(ratios_.shape) + 1e-4)
            # output = out1 +  (ratios_ - dist) * out2
            ratios = ratios.reshape(x.shape[0], x.shape[1])
            # output = output + (x_weights.expand(output.shape).permute(2, 3, 0, 1) * ratios_).permute(2, 3, 0, 1)
        else:
            x = self.ppm(x)
            x = self.ppm_conv(x)
            x = self.classes_conv(x)

            out = F.interpolate(x,
                                size=size,
                                mode='bilinear',
                                align_corners=True)

        return (output, posi_feat, x_weights, ratios) if self.use_threshold else out
        # return out1, out2

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

# net = Dt_PSPNet(4, 16, 'resnet50', False, False)
# summary(net.cuda(), (4, 256, 256))