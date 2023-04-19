from matplotlib.animation import ImageMagickWriter
import torch
from torch import nn, einsum
import torch.nn.functional as F
import os
import numpy as np
# import mresnet as resnet
# import mdensenet as densenet
# import minception as inception

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

class MultiConv(nn.Module):
    def __init__(self, dim=512):
        super(MultiConv, self).__init__()
        self.conv1 = nn.Sequential(
            conv3x3(dim, 128),
            nn.BatchNorm2d(128)
        )
        self.conv2 = nn.Sequential(
            conv3x3(dim, 128, dilation=2),
            nn.BatchNorm2d(128)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(dim, 128, 5, 1, 2),
            nn.BatchNorm2d(128)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(dim, 64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(dim, 64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        xr = F.pad(x, (0, 1, 0, 0))[:, :, :, 1:]
        xl = F.pad(x, (1, 0, 0, 0))[:, :, :, :-1]
        xgrad = self.conv4(xr-xl)
        xd = F.pad(x, (0, 0, 0, 1))[:, :, 1:, :]
        xu = F.pad(x, (0, 0, 1, 0))[:, :, :-1, :]
        ygrad = self.conv5(xd-xu)
        x = torch.cat((x1, x2, x3, xgrad, ygrad), 1)
        x = self.relu(x)
        return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim) #defalt element-wise affine: true
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, attn_dropout=0., dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        # self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_qkv = nn.Conv1d(dim, inner_dim * 3, 1, groups=8)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attn_drop = nn.Dropout(attn_dropout)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, d, n, h = *x.shape, self.heads # b n d
        x = rearrange(x, 'b n d -> b d n')

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (d h) n -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        # self.conv = BasicBlock(dim, dim)
        self.conv = MultiConv(dim)
        self.att = PreNorm(dim, Attention(dim, heads=heads,
                        dim_head=dim_head, dropout=dropout))
        self.ff = PreNorm(dim, FeedForward(dim, mlp_dim))

    def forward(self, x):
        x = rearrange(x, 'b (h w) d -> b d h w', h = 48) # b n d
        x = self.conv(x)
        x = rearrange(x, 'b d h w -> b (h w) d') # b n d
        x1 = self.att(x)
        x = self.ff(x1) + x
        return x


class PatchEmbed(nn.Module):
    def __init__(self, input_channel=1, embed_dim=None):
        super().__init__()

        self.conv = nn.Sequential(
            conv3x3(input_channel, 32),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            conv3x3(32, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            conv3x3(64, embed_dim),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(3, 2, 1), 
        )

    def forward(self, x):
        b, p, c, h, w = x.size()
        x = rearrange(x, 'b p c h w -> (b p) c h w')

        x = self.conv(x)

        x = rearrange(x, '(b p) c h w -> b p c h w', p=p)

        return x


class mtransformer(nn.Module):
    def __init__(self, *, num_patch, stride=16, dim=512, num_input, num_classes, pool='mean'):
        super().__init__()
        embed_dim = 128
        self.pos_embed0 = nn.Parameter(torch.randn((256//stride, embed_dim)))
        self.pos_embed1 = nn.Parameter(torch.randn((256//stride, embed_dim)))
        self.pos_embed2 = nn.Parameter(torch.randn((50, embed_dim)))
        self.patch_embed = PatchEmbed(input_channel=3, embed_dim=embed_dim)
        self.proj = nn.Conv2d(embed_dim*num_patch, dim, kernel_size=1, stride=1)
        self.transformer1 = Transformer(
            dim=dim, heads=8, dim_head=dim//8, mlp_dim=512, dropout=0.1)
        self.transformer2 = Transformer(
            dim=dim, heads=8, dim_head=dim//8, mlp_dim=512, dropout=0.1)
        self.transformer3 = Transformer(
            dim=dim, heads=8, dim_head=dim//8, mlp_dim=512, dropout=0.1)
        self.transformer4 = Transformer(
            dim=dim, heads=8, dim_head=dim//8, mlp_dim=512, dropout=0.1)
        self.pool = pool
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 128)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # self.slicenet = densenet.densenet121(input_channel=num_input, num_classes=num_classes)
        # self.slicenet = inception.inception_v3(input_channel=num_input, num_classes=num_classes, aux_logits=False)
        # self.slicenet = resnet34(input_channel=num_input, num_classes=128)
        # self.fc = nn.Linear(256, num_classes)

    def forward(self, x, pos0, pos1, pos2, y):
        pos_embed0 = self.pos_embed0[pos0[0]]
        pos_embed0 = rearrange(pos_embed0, 'p c -> 1 p c 1 1')
        pos_embed1 = self.pos_embed1[pos1[0]]
        pos_embed1 = rearrange(pos_embed1, 'p c -> 1 p c 1 1')
        pos_embed2 = self.pos_embed2[pos2[0]]
        pos_embed2 = rearrange(pos_embed2, 'p c -> 1 p c 1 1')

        x = self.patch_embed(x) # b p c h w
        x = x + pos_embed0 + pos_embed1 + pos_embed2

        x = rearrange(x, 'b p c h w -> b (p c) h w')
        x = self.proj(x) # b d h w

        x = rearrange(x, 'b d h w -> b (h w) d') # b n d

        x = self.transformer1(x) # b n d
        x = self.transformer2(x) # b n d
        x = self.transformer3(x) # b n d
        x = self.transformer4(x) # b n d

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        # x = x.mean(dim=1) + x.max(dim=1) if self.pool == 'mean' else x[:, 0]

        # out1 = self.mlp_head(x)
        # out2 = self.slicenet(y)
        # out = self.fc(torch.cat((out1, out2), 1))
        return x

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    num_patch = 512
    batch_size = 1
    image = torch.rand(batch_size, num_patch, 3, 48, 48).cuda(0)
    image2 = torch.rand(batch_size, 3, 512, 512).cuda(0)
    pos0 = (torch.rand(batch_size, num_patch)*8).long().cuda(0)
    pos1 = (torch.rand(batch_size, num_patch)*8).long().cuda(0)
    pos2 = (torch.rand(batch_size, num_patch)*50).long().cuda(0)
    model = mtransformer(num_patch=num_patch, stride=16, dim=512, num_input=3, \
        num_classes=3, pool='mean')
    model = model.cuda(0)
    for i in range(100):
        out = model(image, pos0, pos1, pos2, image2)
        print(out.size())