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
        self.de_ratio = ChDecrease(512, 256)
        self.pool = nn.AdaptiveAvgPool2d(1)
        # self.fc1 = nn.Linear(64 * 16 * 16, 2048)
        # self.fc2 = nn.Linear(2048, planes)
        # self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.de_ratio(x)
        x = self.pool(x)
        x = x.reshape(x.shape[0], 2)
        # x = x.reshape(x.shape[0], 2, -1)
        # x = x.view(x.shape[0], -1)
        # x = self.fc1(x)
        # x = self.fc2(x)
        # x = self.softmax(x)
        # x = x.reshape(x.shape[:-1])
        # x = x.reshape(x.shape[0], -1)

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


class Boat_UNet_Part1(nn.Module):
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
        
        self.softmax = nn.Softmax(dim=1)
        self.outconv = Double_conv(64, 2)

    def forward(self, x):
        ori_x = x
        x0, x1, x2, x3, x4 = self.down(x)

        if self.backbone not in ['resnet18', 'resnet34']:
            x1 = self.de1(x1)
            x2 = self.de2(x2)
            x3 = self.de3(x3)
            x4 = self.de4(x4)
        # print(x0.shape, x1.shape, x2.shape, x3.shape, x4.shape)

        ratios = self.fore_pred(x4).float()

        x = self.up1(x4, x3)
        output0 = x
        x = self.up2(x, x2)
        output1 = x
        x = self.up3(x, x1)
        output2 = x
        x = self.up4(x, x0)
        output3 = x
        x = self.up5(x, ori_x)

        fg_output = self.outconv(x)

        # ratios_results = self.softmax(ratios)
        # fg_feature = self.softmax(fg_output)

        # fg_mask = torch.zeros(fg_feature.shape[0], fg_feature.shape[2], fg_feature.shape[3]).cuda()
        # for i in range(ratios.shape[0]):
        #     fg_mask[i] = (fg_feature[i] > (1 - ratios_results[i, 0])).float()[-1]
        # fg_mask = fg_mask.reshape(fg_mask.shape[0], 1, fg_mask.shape[1], fg_mask.shape[2])

        # output = torch.cat([x, fg_mask], dim=1)

        # return fg_output, ratios, output, [ori_x, x0, x1, x2, x3, x4], [output3, output2, output1, output0]
        return fg_output


class Boat_UNet_Part2(nn.Module):
    def __init__(self, inplanes, num_classes, backbone):
        super().__init__()
        self.down = ResDown(in_channels=inplanes, backbone=backbone)
        self.backbone = backbone
        
        self.co_de1 = ChDecrease(128, 2)
        self.co_de2 = ChDecrease(256, 2)
        self.co_de3 = ChDecrease(512, 2)
        self.co_de4 = ChDecrease(1024, 2)

        if self.backbone not in ['resnet18', 'resnet34']:
            self.de1 = ChDecrease(256, 4)
            self.de2 = ChDecrease(512, 4)
            self.de3 = ChDecrease(1024, 4)
            self.de4 = ChDecrease(2048, 4)


        self.up1 = Up(512, 768, 256)
        self.up2 = Up(256, 384, 128)
        self.up3 = Up(128, 192, 64)
        self.up4 = Up(64, 128, 64)
        self.up5 = Up(64, 133, 64, last_cat=True)
        self.outconv = nn.Conv2d(64, num_classes, 1)

    def forward(self, x, down_list, up_list):
        input_x = x
        x0, x1, x2, x3, x4 = self.down(x)
        ori_x, x0_0, x1_0, x2_0, x3_0, x4_0 = down_list
        x0_1, x1_1, x2_1, x3_1 = up_list
        # print(x0.shape, x1.shape, x2.shape, x3.shape, x4.shape)
        # print(x0_1.shape, x1_1.shape, x2_1.shape, x3_1.shape, x4_1.shape)
        
        x3 = torch.cat([x3, x3_1], dim=1)
        x3 = self.co_de3(x3)
        x2 = torch.cat([x2, x2_1], dim=1)
        x2 = self.co_de2(x2)
        x1 = torch.cat([x1, x1_1], dim=1)
        x1 = self.co_de1(x1)
        x0 = torch.cat([x0, x0_1], dim=1)
        x0 = self.co_de1(x0)
        
        if self.backbone not in ['resnet18', 'resnet34']:

            x1 = self.de1(x1)
            x2 = self.de2(x2)
            x3 = self.de3(x3)
            x4 = self.de4(x4)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        ori_x = torch.cat([input_x, ori_x], dim=1)
        # print(ori_x.shape, x.shape)
        x = self.up5(x, ori_x)
        output = self.outconv(x)

        return output


class Boat_UNet(nn.Module):
    def __init__(self, inplanes, num_classes, backbone1, backbone2):
        super().__init__()
        self.part1 = Boat_UNet_Part1(inplanes, num_classes, backbone1)
        self.part2 = Boat_UNet_Part2(65, num_classes, backbone2)

    def forward(self, x):
        fg_mask, output_ratios, x, down_list, up_list = self.part1(x)
        output = self.part2(x, down_list, up_list)

        return fg_mask, output_ratios, output


# net = Boat_UNet_Part1(4, 1, 'resnet50')
# summary(net.cuda(), (4, 256, 256))

# net = Boat_UNet(4, 16, 'resnet50', 'resnet18')
# summary(net.cuda(), (4, 256, 256))

# net = Boat_UNet(4, 16, 'resnet50', 'resnet18').cuda()
# test_data = torch.randn([2, 4, 256, 256]).cuda()
# aa, bb, cc = net(test_data)
# print(aa.shape, bb.shape, cc.shape)