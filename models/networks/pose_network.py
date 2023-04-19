import torch
import torch.nn as nn
import numpy as np

from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101


def resnet_channel_input(model, total_input_channels=6):
    weight = model.conv1.weight.data
    model.conv1 = nn.Conv2d(total_input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    if total_input_channels == 6:
        model.conv1.weight.data = torch.cat([weight] * 2, 1) / 2.0
    if total_input_channels > 6:
        diff_channels = total_input_channels - 6
        model.conv1.weight.data = torch.cat([weight]*2 + [weight[:,0:1,...]]*diff_channels, 1) / 2.0
    return model

class PoseEncoder(nn.Module):
    def __init__(self, cfgs, num_layers):
        super(PoseEncoder, self).__init__()
        self.cfgs = cfgs
        if self.cfgs["pose_inputs"] == "rgb":
            inputs_channels = 6
        elif self.cfgs["pose_inputs"] == "rgbd":
            inputs_channels = 8
        elif self.cfgs["pose_inputs"] == "rgbd_d":
            inputs_channels = 6 + self.cfgs["depth_d_planes"] * 2

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        resnets = {18: resnet18,
                   34: resnet34,
                   50: resnet50,
                   101: resnet101,}

        self.encoder = resnet_channel_input(resnets[num_layers](True), total_input_channels=inputs_channels)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        self.features = []
        x = self.encoder.conv1(input_image)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features

class Flatten(nn.Module):
    def forward(self, x):
        return x.contiguous().view(x.shape[0], -1).contiguous()

class ClassPoseDecoder(nn.Module):
    def __init__(self, cfgs, num_ch_enc):
        super(ClassPoseDecoder, self).__init__()
        # self.dof = cfgs["dof"]
        # self.rot_coef = cfgs["rot_coef"]
        # self.trans_coef = cfgs["trans_coef"]

        num_reduce = 256
        spatial_hw = 4
        hidden_size = 512
        drop = 0.5

        roty_channels = 81
        transx_channels = 51
        transz_channels = 51
        roty_range = [-0.40, 0.40]
        transx_range = [-0.25, 0.25]
        transz_range = [-0.40, 0.05]
        # if action == 1:
        #     roty_channels = 81
        #     transx_channels = 51
        #     transz_channels = 51
        #     roty_range = [-0.40, 0.40]
        #     transx_range = [-0.25, 0.25]
        #     transz_range = [-0.40, 0.05]
        # elif action == 2:
        #     roty_channels = 51
        #     transx_channels = 21
        #     transz_channels = 11
        #     roty_range = [0.05, 0.40]
        #     transx_range = [-0.1, 0.1]
        #     transz_range = [-0.05, 0.05]
        # elif action == 3:
        #     roty_channels = 51
        #     transx_channels = 21
        #     transz_channels = 11
        #     roty_range = [-0.40, -0.05]
        #     transx_range = [-0.1, 0.1]
        #     transz_range = [-0.05, 0.05]

        self.roty_planes = torch.linspace(roty_range[0], roty_range[1], roty_channels)
        self.transx_planes = torch.linspace(transx_range[0], transx_range[1], transx_channels)
        self.transz_planes = torch.linspace(transz_range[0], transz_range[1], transz_channels)

        self.reduce = nn.Sequential(
            nn.Conv2d(num_ch_enc[-1], num_reduce, 1), 
            nn.ReLU(True),
            nn.Conv2d(num_reduce, num_reduce, 1), 
            nn.ReLU(True),
            nn.Conv2d(num_reduce, num_reduce, 1), 
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d([spatial_hw, spatial_hw]),
        )
        self.fc = nn.Sequential(
            Flatten(),
            nn.Linear(num_reduce*spatial_hw*spatial_hw, hidden_size),
            nn.ReLU(True),
            # nn.Dropout(drop), 
        )
        self.roty = nn.Sequential(nn.Linear(hidden_size, roty_channels), nn.Softmax(1))
        self.transx = nn.Sequential(nn.Linear(hidden_size, transx_channels), nn.Softmax(1))
        self.transz = nn.Sequential(nn.Linear(hidden_size, transz_channels), nn.Softmax(1))
        

        self.rot_axis = torch.Tensor([0,1,0]).view(1, 3).float()
        self.x_dir = torch.Tensor([1,0,0]).view(1, 3).float()
        self.z_dir = torch.Tensor([0,0,1]).view(1, 3).float()

    def forward(self, input_features):
        feat = input_features[-1]
        device = feat.device

        out = self.reduce(feat)
        out = self.fc(out)

        ry = torch.sum( (self.roty(out) * self.roty_planes[np.newaxis, ...].to(device)), dim=1, keepdim=True)
        tx = torch.sum( (self.transx(out) * self.transx_planes[np.newaxis, ...].to(device)), dim=1, keepdim=True)
        tz = torch.sum( (self.transz(out) * self.transz_planes[np.newaxis, ...].to(device)), dim=1, keepdim=True)

        rotv = ry * self.rot_axis.to(out.device)
        trans = (tx * self.x_dir.to(out.device) +  tz * self.z_dir.to(out.device))
        return rotv, trans

class PoseDecoder(nn.Module):
    def __init__(self, cfgs, num_ch_enc):
        super(PoseDecoder, self).__init__()
        self.dof = cfgs["dof"]
        self.rot_coef = cfgs["rot_coef"]
        self.trans_coef = cfgs["trans_coef"]
        
        self.reduce = nn.Conv2d(num_ch_enc[-1], 256, 1)
        self.conv1 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv3 = nn.Conv2d(256, self.dof, 1)
        self.relu = nn.ReLU()

        self.rot_axis = torch.Tensor([0,1,0]).view(1, 3).float()
        self.x_dir = torch.Tensor([1,0,0]).view(1, 3).float()
        self.z_dir = torch.Tensor([0,0,1]).view(1, 3).float()

    def forward(self, input_features):
        f = input_features[-1]
        out = self.relu(self.reduce(f))
        out = self.relu(self.conv1(out))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        out = out.mean(3).mean(2)
        out = out.view(-1, self.dof) * 0.1

        if self.dof == 6:
            rotv = self.rot_coef * out[..., :3]
            trans = self.trans_coef * out[..., 3:]

        elif self.dof == 3:
            rotv = out[..., :1] * self.rot_axis.to(out.device)
            trans = (out[..., 1:2] * self.x_dir.to(out.device) + out[..., 2:3] * self.z_dir.to(out.device))
        return rotv, trans