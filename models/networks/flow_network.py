import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3_LeakyReLU(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=True),
            nn.LeakyReLU(0.1))

def coords_grid_v2(batch, h, w):
    coords = torch.meshgrid(torch.arange(h), torch.arange(w))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords.repeat(batch, 1, 1, 1) # b 2 h w

def grid_sampler_v2(img, coords, mode='bilinear', padding='zeros'):
    """ Wrapper for grid_sample, uses pixel coordinates """
    _, _, H, W = img.shape
    xgrid, ygrid = coords.permute(0, 2, 3, 1).split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True, mode=mode, padding_mode=padding)
    return img

def correlate(feat1, feat2, coords=None, r=None, mode='fast'):
    b, c, h, w = feat1.shape
    if r is None:
        feat1 = feat1.view(b, c, h*w) 
        feat2 = feat2.view(b, c, h*w) 
        corr = torch.matmul(feat1.transpose(1,2), feat2)
        corr = corr.view(b, h, w, -1).permute(0, 3, 1, 2)
    else:
        feat2 = grid_sampler_v2(feat2, coords)
        if 'fast' in mode:
            num_feats = (2*r + 1)
            num_feats2 = num_feats**2
            feat1_unflod = feat1.view(b, c, 1, h*w)
            feat2_unflod = F.unfold(feat2, num_feats, dilation=1, padding=r, stride=1).view(b, c, num_feats2, -1)
            corr = (feat2_unflod * feat1_unflod).sum(1).view(b, num_feats2, h,w)
        else:
            feat2_pad = F.pad(feat2, (r,r,r,r), value=0)
            corr = []
            for i in range(2*r + 1):
                for j in range(2*r+ 1):
                    corr.append((feat1 * feat2_pad[:, :, i:(i + h), j:(j + w)]).sum(dim=1))
            corr = torch.stack(corr, 3).contiguous().permute(0, 3, 1, 2)
            
    corr = corr / torch.sqrt(torch.tensor(c).float())
    return corr

class FlowEncoder(nn.Module):
    def __init__(self):
        super(FlowEncoder, self).__init__()

        self.feature1 = nn.Sequential(
            conv3x3_LeakyReLU(3,   16, kernel_size=3, stride=2),
            conv3x3_LeakyReLU(16,  16, kernel_size=3, stride=1),
            conv3x3_LeakyReLU(16,  16, kernel_size=3, stride=1)
        )
        self.feature2 = nn.Sequential(
            conv3x3_LeakyReLU(16,  32, kernel_size=3, stride=2),
            conv3x3_LeakyReLU(32,  32, kernel_size=3, stride=1),
            conv3x3_LeakyReLU(32,  32, kernel_size=3, stride=1)
        )
        self.feature3 = nn.Sequential(
            conv3x3_LeakyReLU(32,  64, kernel_size=3, stride=2),
            conv3x3_LeakyReLU(64,  64, kernel_size=3, stride=1),
            conv3x3_LeakyReLU(64,  64, kernel_size=3, stride=1)
        )
        self.feature4 = nn.Sequential(
            conv3x3_LeakyReLU(64,  96, kernel_size=3, stride=2),
            conv3x3_LeakyReLU(96,  96, kernel_size=3, stride=1),
            conv3x3_LeakyReLU(96,  96, kernel_size=3, stride=1)
        )
        self.feature5 = nn.Sequential(
            conv3x3_LeakyReLU(96,  128, kernel_size=3, stride=2),
            conv3x3_LeakyReLU(128, 128, kernel_size=3, stride=1),
            conv3x3_LeakyReLU(128, 128, kernel_size=3, stride=1)
        )
        self.feature6 = nn.Sequential(
            conv3x3_LeakyReLU(128, 196, kernel_size=3, stride=2),
            conv3x3_LeakyReLU(196, 196, kernel_size=3, stride=1),
            conv3x3_LeakyReLU(196, 196, kernel_size=3, stride=1)
        )

    def forward(self, inputs):
        inputs = (inputs - 0.5) * 2.0
        feature1 = self.feature1(inputs)
        feature2 = self.feature2(feature1)
        feature3 = self.feature3(feature2)
        feature4 = self.feature4(feature3)
        feature5 = self.feature5(feature4)
        feature6 = self.feature6(feature5)
        return [ feature1, feature2, feature3, feature4, feature5, feature6 ]

class Refiner(nn.Module):
    def __init__(self):
        super(Refiner, self).__init__()

        self.netMain = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=81 + 32 + 2 + 16 + 128 + 128 + 96 + 64 + 32 + 2, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=2, dilation=2),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=4, dilation=4),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=8, dilation=8),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=16, dilation=16),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1),
            torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1)
        )

    def forward(self, flow, feat):
        return self.netMain(torch.cat([ flow, feat], dim=1))

class Decode_One_Level(nn.Module):
    def __init__(self):
        super(Decode_One_Level, self).__init__()
        chs = 81+32+2+16

        self.feat_ch = 16
        self.r = 4

        self.netfeat = nn.Conv2d(chs + 128 + 128 + 96 + 64 + 32, self.feat_ch, kernel_size=1, stride=1, padding=0)
        self.netOne = conv3x3_LeakyReLU(chs,                     128, kernel_size=3, stride=1)
        self.netTwo = conv3x3_LeakyReLU(chs + 128,                128, kernel_size=3, stride=1)
        self.netThr = conv3x3_LeakyReLU(chs + 128 + 128,           96,  kernel_size=3, stride=1)
        self.netFou = conv3x3_LeakyReLU(chs + 128 + 128 + 96,      64,  kernel_size=3, stride=1)
        self.netFiv = conv3x3_LeakyReLU(chs + 128 + 128 + 96 + 64, 32,  kernel_size=3, stride=1)
        self.netFLow = nn.Conv2d(chs + 128 + 128 + 96 + 64 + 32, 2, kernel_size=3, stride=1, padding=1)

    def forward(self, feat0, feat1, feat01x1, prev_feat, prev_flow):
        B, C, H, W = feat0.shape
        coords = coords_grid_v2(B, H, W).to(feat0.device)

        if prev_flow is None:
            prev_flow = torch.zeros_like(coords)
            # prev_volume = torch.zeros((B, 81, H, W)).to(feat0.device)
            feat = torch.zeros((B, self.feat_ch, H, W)).to(feat0.device)
        else:
            prev_flow = F.interpolate(prev_flow, size=(H, W), mode='bilinear', align_corners=True) * 2.0
            # prev_volume = F.interpolate(prev_volume, size=(H, W), mode='bilinear', align_corners=True)
            feat = self.netfeat(F.interpolate(prev_feat, size=(H, W), mode='bilinear', align_corners=True))

        tenVolume = F.leaky_relu(correlate(feat0, feat1, coords + prev_flow, r=self.r, mode='fast'), negative_slope=0.1, inplace=False)
        tenFeat = torch.cat([ tenVolume, feat01x1, prev_flow, feat ], 1)

        tenFeat = torch.cat([ self.netOne(tenFeat), tenFeat ], 1)
        tenFeat = torch.cat([ self.netTwo(tenFeat), tenFeat ], 1)
        tenFeat = torch.cat([ self.netThr(tenFeat), tenFeat ], 1)
        tenFeat = torch.cat([ self.netFou(tenFeat), tenFeat ], 1)
        tenFeat = torch.cat([ self.netFiv(tenFeat), tenFeat ], 1)
        
        flow = self.netFLow(tenFeat)

        flow = flow if prev_flow is None else flow + prev_flow
        return flow, tenFeat
    
class FlowDecoder(nn.Module):
    def __init__(self):
        super(FlowDecoder, self).__init__()

        self.decoder = Decode_One_Level()
        self.refiner = Refiner()
        self.conv1 = conv3x3_LeakyReLU(32, 32, 1, 1, padding=0, dilation=1)
        self.conv2 = conv3x3_LeakyReLU(64, 32, 1, 1, padding=0, dilation=1)
        self.conv3 = conv3x3_LeakyReLU(96, 32, 1, 1, padding=0, dilation=1)
        self.conv4 = conv3x3_LeakyReLU(128, 32, 1, 1, padding=0, dilation=1)
        self.conv5 = conv3x3_LeakyReLU(196, 32, 1, 1, padding=0, dilation=1)
    
    def forward(self, tenFirst, tenSecond, frame_id):
        t0_1x1_l2 = self.conv1(tenFirst[1])
        t0_1x1_l3 = self.conv2(tenFirst[2])
        t0_1x1_l4 = self.conv3(tenFirst[3])
        t0_1x1_l5 = self.conv4(tenFirst[4])
        t0_1x1_l6 = self.conv5(tenFirst[5])

        flow5, feat5 = self.decoder(tenFirst[-1], tenSecond[-1], t0_1x1_l6, None, None)
        flow4, feat4 = self.decoder(tenFirst[-2], tenSecond[-2], t0_1x1_l5, feat5, flow5)
        flow3, feat3 = self.decoder(tenFirst[-3], tenSecond[-3], t0_1x1_l4, feat4, flow4)
        flow2, feat2 = self.decoder(tenFirst[-4], tenSecond[-4], t0_1x1_l3, feat3, flow3)
        flow1, feat1 = self.decoder(tenFirst[-5], tenSecond[-5], t0_1x1_l2, feat2, flow2)

        # flow5 = flow5 + self.refiner(flow5, feat5)
        # flow4 = flow4 + self.refiner(flow4, feat4)
        # flow3 = flow3 + self.refiner(flow3, feat3)
        # flow2 = flow2 + self.refiner(flow2, feat2)
        flow1 = flow1 + self.refiner(flow1, feat1)

        outputs = {}
        outputs['flow', frame_id, 3] = flow4
        outputs['flow', frame_id, 2] = flow3
        outputs['flow', frame_id, 1] = flow2
        outputs['flow', frame_id, 0] = flow1

        outputs['flow_feat', frame_id, 3] = feat4
        outputs['flow_feat', frame_id, 2] = feat3
        outputs['flow_feat', frame_id, 1] = feat2
        outputs['flow_feat', frame_id, 0] = feat1
        return outputs

if __name__ == "__main__":
    device = "cuda:0"
    rgb0 = torch.randn([1, 3, 256, 256]).to(device)
    rgb1 = torch.randn([1, 3, 256, 256]).to(device)
    encoder = FlowEncoder().to(device)
    decoder = FlowDecoder().to(device)
    flow = decoder(encoder(rgb0), encoder(rgb1), 0)