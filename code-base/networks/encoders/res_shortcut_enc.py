import torch.nn as nn
from utils import CONFIG
from networks.encoders.resnet_enc import ResNet_D
from networks.ops import SpectralNorm
from ops.dcn import ModulatedDeformConvPack, modulated_deform_conv
import torch
import torchvision

class FeaFusion(nn.Module):
    def __init__(self):
        super(FeaFusion, self).__init__()

    def forward(self, ):
        pass


class DCNv2Pack(ModulatedDeformConvPack):
    """Modulated deformable conv for deformable alignment.

    Different from the official DCNv2Pack, which generates offsets and masks
    from the preceding features, this DCNv2Pack takes another different
    features to generate offsets and masks.

    Ref:
        Delving Deep into Deformable Alignment in Video Super-Resolution.
    """

    def forward(self, x, feat):
        out = self.conv_offset(feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        offset_absmean = torch.mean(torch.abs(offset))
        if offset_absmean > 50:
            # logger = get_root_logger()
            # logger.warning(f'Offset abs mean is {offset_absmean}, larger than 50.')
            print(f'Offset abs mean is {offset_absmean}, larger than 50.')

        # if LooseVersion(torchvision.__version__) >= LooseVersion('0.9.0'):
            return torchvision.ops.deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.padding,
                                                 self.dilation, mask)
        # else:
        #     return modulated_deform_conv(x, offset, mask, self.weight, self.bias, self.stride, self.padding,
        #                                  self.dilation, self.groups, self.deformable_groups)
class FeaAlign(nn.Module):

    def __init__(self, channel_num=64, deformable_groups=8):
        super(FeaAlign, self).__init__()

        self.conv1 = nn.Conv2d(channel_num * 2, channel_num, (1, 1))
        self.conv2 = nn.Sequential(
            nn.Conv2d(channel_num, channel_num, (3, 3), (1,1), 1),
            nn.Conv2d(channel_num, channel_num, (3, 3), (1,1), 1),
        )
        self.dcn = DCNv2Pack(channel_num * 2, channel_num, 3, padding=1, deformable_groups=deformable_groups)

    def forward(self, img_ref, img):
        img_cat = self.conv1(torch.cat([img_ref, img], dim=1))
        offset = self.conv2(img_cat)

        feat = self.dcn(img, offset)

        return feat

class AFModule(nn.Module):
    def __init__(self, channel_num=64, deformable_groups=8):
        super(AFModule, self).__init__()

        self.fea_align_1 = FeaAlign(channel_num, deformable_groups)
        self.fea_align_2 = FeaAlign(channel_num, deformable_groups)

    def forward(self, img1, img2, img3):
        # tfa
        feat1 = self.fea_align_1(img2, img1)
        feat2 = self.fea_align_2(img2, img3)
        # tff





class ResShortCut_D(ResNet_D):

    def __init__(self, block, layers, norm_layer=None, late_downsample=False):
        super(ResShortCut_D, self).__init__(block, layers, norm_layer, late_downsample=late_downsample)
        first_inplane = 3 + CONFIG.model.mask_channel
        self.shortcut_inplane = [first_inplane, self.midplanes, 64, 128, 256]
        self.shortcut_plane = [32, self.midplanes, 64, 128, 256]

        self.shortcut = nn.ModuleList()
        for stage, inplane in enumerate(self.shortcut_inplane):
            self.shortcut.append(self._make_shortcut(inplane, self.shortcut_plane[stage]))
        # init fusion module
        self.feafusion_inchannel = [32, self.midplanes, 64, 128, 256]
        self.feafusion_outchannel = [32, self.midplanes, 64, 128, 256]
        self.feafusion = nn.ModuleList()
        for stage, inchannel in enumerate(self.feafusion_inchannel):
            self.feafusion.append(self._make_feafusion(inchannel, self.feafusion_outchannel[stage]))

    def _make_feafusion(self, inchannel, outchannel):
        pass

    def _make_shortcut(self, inplane, planes):
        return nn.Sequential(
            SpectralNorm(nn.Conv2d(inplane, planes, kernel_size=3, padding=1, bias=False)),
            nn.ReLU(inplace=True),
            self._norm_layer(planes),
            SpectralNorm(nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)),
            nn.ReLU(inplace=True),
            self._norm_layer(planes)
        )

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)
        x1 = self.activation(out)  # N x 32 x 256 x 256
        out = self.conv3(x1)
        out = self.bn3(out)
        out = self.activation(out)

        x2 = self.layer1(out)  # N x 64 x 128 x 128
        x3 = self.layer2(x2)  # N x 128 x 64 x 64
        x4 = self.layer3(x3)  # N x 256 x 32 x 32
        out = self.layer_bottleneck(x4)  # N x 512 x 16 x 16

        fea1 = self.shortcut[0](x)  # input image and mask
        fea2 = self.shortcut[1](x1)
        fea3 = self.shortcut[2](x2)
        fea4 = self.shortcut[3](x3)
        fea5 = self.shortcut[4](x4)

        return out, {'shortcut': (fea1, fea2, fea3, fea4, fea5), 'image': x[:, :3, ...]}
