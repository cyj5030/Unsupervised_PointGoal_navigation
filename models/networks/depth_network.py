import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101
import numpy as np

class DepthEncoder(nn.Module):
    def __init__(self, num_layers, num_inputs=1):
        super(DepthEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        resnets = {18: resnet18,
                   34: resnet34,
                   50: resnet50,
                   101: resnet101,}

        self.encoder = resnets[num_layers](True)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        features = []
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        features.append(self.encoder.relu(x))
        features.append(self.encoder.layer1(self.encoder.maxpool(features[-1])))
        features.append(self.encoder.layer2(features[-1]))
        features.append(self.encoder.layer3(features[-1]))
        features.append(self.encoder.layer4(features[-1]))

        return features


class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), kernel_size=1, stride=1, bias=bias)
    def forward(self, x):
        out = self.conv(x)
        return out

class Conv3x3(nn.Module):
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()
        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out

def nearest_interp(target, size):
    return F.interpolate(target, size, mode='nearest')

class DepthDecoder(nn.Module):
    def __init__(self,  num_ch_enc):
        super(DepthDecoder, self).__init__()
        bottleneck = [196, 128, 96, 64, 32]
        self.do = nn.Dropout(0.5)

        self.reduce4 = nn.Sequential(Conv3x3(num_ch_enc[4], bottleneck[0]), nn.ELU())
        self.reduce3 = nn.Sequential(Conv3x3(num_ch_enc[3], bottleneck[1]), nn.ELU())
        self.reduce2 = nn.Sequential(Conv3x3(num_ch_enc[2], bottleneck[2]), nn.ELU())
        self.reduce1 = nn.Sequential(Conv3x3(num_ch_enc[1], bottleneck[3]), nn.ELU())

        self.iconv4 = nn.Sequential(Conv3x3(bottleneck[0], bottleneck[0]))
        self.iconv3 = nn.Sequential(Conv3x3(sum(bottleneck[0:2])+2, bottleneck[1]), nn.ELU())
        self.iconv2 = nn.Sequential(Conv3x3(sum(bottleneck[1:3])+2, bottleneck[2]), nn.ELU())
        self.iconv1 = nn.Sequential(Conv3x3(sum(bottleneck[2:4])+2, bottleneck[3]), nn.ELU())

        # disp
        self.disp4 = nn.Sequential(Conv3x3(bottleneck[0], 2), nn.Sigmoid())
        self.disp3 = nn.Sequential(Conv3x3(bottleneck[1], 2), nn.Sigmoid())
        self.disp2 = nn.Sequential(Conv3x3(bottleneck[2], 2), nn.Sigmoid())
        self.disp1 = nn.Sequential(Conv3x3(bottleneck[3], 2), nn.Sigmoid())

    def forward(self, input_features, frame_id):
        self.outputs = {}
        l0, l1, l2, l3, l4 = input_features
        l3 = self.do(l3)
        l4 = self.do(l4)

        x4 = self.reduce4(l4)
        x4 = self.iconv4(x4)
        x4 = nearest_interp(x4, l3.shape[2:])
        disp4 = self.disp4(x4)


        x3 = self.reduce3(l3)
        x3 = torch.cat((x3, x4, disp4), 1)
        x3 = self.iconv3(x3)
        x3 = nearest_interp(x3, l2.shape[2:])
        disp3 = self.disp3(x3)


        x2 = self.reduce2(l2)
        x2 = torch.cat((x2, x3, disp3), 1)
        x2 = self.iconv2(x2)
        x2 = nearest_interp(x2, l1.shape[2:])
        disp2 = self.disp2(x2)

        x1 = self.reduce1(l1)
        x1 = torch.cat((x1, x2, disp2), 1)
        x1 = self.iconv1(x1)
        x1 = nearest_interp(x1, l0.shape[2:])
        disp1 = self.disp1(x1)

        outputs = {}

        outputs['disp', frame_id, 3] = disp4[:, 0:1, :, :]
        outputs['disp', frame_id, 2] = disp3[:, 0:1, :, :]
        outputs['disp', frame_id, 1] = disp2[:, 0:1, :, :]
        outputs['disp', frame_id, 0] = disp1[:, 0:1, :, :]

        outputs['uncert', frame_id, 3] = disp4[:, 1:2, :, :]
        outputs['uncert', frame_id, 2] = disp3[:, 1:2, :, :]
        outputs['uncert', frame_id, 1] = disp2[:, 1:2, :, :]
        outputs['uncert', frame_id, 0] = disp1[:, 1:2, :, :]

        return outputs