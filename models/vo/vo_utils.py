import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def bilinear_interp(target, size, align=True):
    return F.interpolate(target, size, mode='bilinear', align_corners=align)
    
def disp_to_depth(disp, max_depth, min_depth):
    min_disp = 1. / max_depth  # 0.01
    max_disp = 1. / min_depth  # 10
    scaled_disp = min_disp + (max_disp - min_disp) * disp  # (10-0.01)*disp+0.01
    depth = 1. / scaled_disp
    return scaled_disp, depth

def grid_sampler(img, coords, mode='bilinear', padding='zeros'):
    coords = coords.permute(0, 2, 3, 1)
    img = F.grid_sample(img, coords, align_corners=True, mode=mode, padding_mode=padding)
    return img

def reproject(coords, depth, rot_matrix, K, K_inv):
    B, _, H, W = depth.shape

    src_xyz = torch.matmul(K_inv, coords.view(B, 3, -1)) * depth.view(B, 1, -1)
    # src_xyz[:, 2] *= -1

    Rot, trans = rot_matrix[:, :3, :3], rot_matrix[:, :3, 3:]
    tgt_xyz = torch.matmul(Rot, src_xyz) + trans # b 3 h*w

    tgt_xy = torch.matmul(K, tgt_xyz)
    tgt_xy = tgt_xy[:, 0:2, :] / tgt_xy[:, 2:, :].clamp(min=1e-6)

    return tgt_xy.view(B, 2, H, W)

def out_boundary_mask(coords):
    pos_x, pos_y = coords.split([1, 1], dim=1)

    with torch.no_grad():
        outgoing_mask = torch.ones_like(pos_x)
        outgoing_mask[pos_x > 1] = 0
        outgoing_mask[pos_x < -1] = 0
        outgoing_mask[pos_y > 1] = 0
        outgoing_mask[pos_y < -1] = 0
    return outgoing_mask.float()

def coords_grid(batch, h, w):
    coords = torch.meshgrid(torch.arange(h), torch.arange(w))
    coords = torch.stack(coords[::-1], dim=0).float()
    coords = coords.repeat(batch, 1, 1, 1) # b 2 h w

    xgrid, ygrid = coords.split([1,1], dim=1)
    xgrid = 2*xgrid/(w-1) - 1
    ygrid = 2*ygrid/(h-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=1)
    return grid
    
def convert_rt(rpy, trans, invert=False):
    B = trans.shape[0]
    trans = trans.view(B, 3, 1)

    rot = from_rpy(rpy)
    if invert:
        rot = rot.transpose(1, 2)
        trans = rot.matmul(-trans)
    return rot, trans

def to_matrix(rpy, trans, invert=False):
    rot, trans = convert_rt(rpy, trans, invert=invert)
    rot_matrix = torch.eye(4, requires_grad=True).view(1, 4, 4).expand([rpy.shape[0], 4, 4]).clone().to(rpy.device)
    rot_matrix[:, :3, :3] = rot
    rot_matrix[:, :3, 3:] = trans
    return rot_matrix

def from_rpy(rpy):
    B = rpy.shape[0]
    vec = rpy.view(B, 1, 3)
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((B, 3, 3), requires_grad=True).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    return rot

def robust_l1(pred, target):
    eps = 1e-3
    return torch.sqrt(torch.pow(target - pred, 2) + eps ** 2)

def depth_l1(pred, target, md=0.01):
    pred = -torch.log(pred.clamp(min=md))
    target = -torch.log(target.clamp(min=md))
    c_norm = -np.log(md)
    return robust_l1(pred, target) / c_norm

def ssim(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = nn.AvgPool2d(3, 1, padding=1)(x)
    mu_y = nn.AvgPool2d(3, 1, padding=1)(y)

    sigma_x = nn.AvgPool2d(3, 1, padding=1)(x**2) - mu_x**2
    sigma_y = nn.AvgPool2d(3, 1, padding=1)(y**2) - mu_y**2
    sigma_xy = nn.AvgPool2d(3, 1, padding=1)(x * y) - mu_x * mu_y

    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)

    SSIM = SSIM_n / SSIM_d
    return torch.clamp((1 - SSIM) / 2, 0, 1)

def census(img1, img2, q, max_distance=3):
    patch_size = 2 * max_distance + 1

    def _ternary_transform(image):
        R, G, B = torch.split(image, 1, 1)
        intensities_torch = (0.2989 * R + 0.5870 * G + 0.1140 * B)  # * 255  # convert to gray
        out_channels = patch_size * patch_size
        w = np.eye(out_channels).reshape((patch_size, patch_size, 1, out_channels))  # h,w,1,out_c
        w_ = np.transpose(w, (3, 2, 0, 1))  # 1,out_c,h,w
        weight = torch.from_numpy(w_).float()
        weight = weight.to(image.device)
        patches_torch = torch.conv2d(input=intensities_torch, weight=weight, bias=None, stride=[1, 1], padding=[max_distance, max_distance])
        transf_torch = patches_torch - intensities_torch
        transf_norm_torch = transf_torch / torch.sqrt(0.81 + transf_torch ** 2)
        return transf_norm_torch

    def _hamming_distance(t1, t2):
        dist = (t1 - t2) ** 2
        dist = torch.sum(dist / (0.1 + dist), 1, keepdim=True)
        return dist

    def create_mask(tensor, paddings):
        shape = tensor.shape  # N,c, H,W
        inner_width = shape[2] - (paddings[0][0] + paddings[0][1])
        inner_height = shape[3] - (paddings[1][0] + paddings[1][1])
        inner_torch = torch.ones([shape[0], shape[1], inner_width, inner_height]).float()
        inner_torch = inner_torch.to(tensor.device)
        mask2d = F.pad(inner_torch, [paddings[0][0], paddings[0][1], paddings[1][0], paddings[1][1]])
        return mask2d

    img1 = _ternary_transform(img1)
    img2 = _ternary_transform(img2)
    dist = _hamming_distance(img1, img2)
    transform_mask = create_mask(dist, [[max_distance, max_distance],
                                                [max_distance, max_distance]])
    census_loss = (torch.abs(dist) + 1e-2).pow(q)
    return census_loss, transform_mask
    
def rotation_error(rot_error):
    a = rot_error[:, 0, 0]
    b = rot_error[:, 1, 1]
    c = rot_error[:, 2, 2]
    d = 0.5*(a+b+c-1.0)
    rot_error = torch.arccos(torch.maximum(torch.minimum(d, torch.tensor([1.0])), torch.tensor([-1.0])))
    return rot_error

def translation_error(pose_error):
    dx = pose_error[:, 0]
    dy = pose_error[:, 1]
    dz = pose_error[:, 2]
    return torch.sqrt(dx**2+dy**2+dz**2)

def cartesian2polar(xyz):
    x, z = xyz[:, 0], xyz[:, 2]
    rho = torch.sqrt(x ** 2 + z ** 2)
    phi = torch.atan2(z, x)
    polar = torch.stack([rho, phi], 1)
    return polar

def correlate(feat1, feat2, coords, r, mode='fast'):
    b, c, h, w = feat1.shape
    feat2 = grid_sampler(feat2, coords)
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

def depth_discretization(depth, max_depth=1, num_planes=10):
    planes = torch.linspace(0, max_depth, num_planes + 1)

    depth_d = []
    for i in range(planes.shape[0] - 1):
        depth_d.append( (depth >= planes[i]).float() * (depth < planes[i+1]).float() )
    depth_d = torch.cat(depth_d, 1)

    re_arrange_idx = []
    for i in range(depth.shape[1]):
        re_arrange_idx.append(torch.arange(num_planes) * depth.shape[1] + i)
    re_arrange_idx = torch.cat(re_arrange_idx, 0)

    sorted_depth_d = depth_d[:, re_arrange_idx]
    return sorted_depth_d

def gabor_filter(inputs, sigma=1.0):
    r = int(sigma*3)
    yy, xx = torch.meshgrid(torch.linspace(-r,r,r*2+1), torch.linspace(-r,r,r*2+1))
    yy = -1 * yy
    # coeff = 1 / (np.sqrt(2*np.pi) * sigma**3)
    A = -xx * torch.exp(-(xx**2 + (0.25*yy**2)) / (2*sigma**2))
    kernel = (A).view(1, 1, *A.shape).repeat(inputs.shape[1], inputs.shape[1], 1, 1)
    response = F.conv2d(inputs, kernel.to(inputs.device), padding=r).abs()
    response = response / response.max()
    return response

if __name__ == "__main__":
    rot = from_rpy(torch.tensor([0., 0., 0.]).view(1, 3))
    print(rot)