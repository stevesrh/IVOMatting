from distutils.version import LooseVersion

import torch.nn as nn
from utils import CONFIG
from networks.encoders.resnet_enc import ResNet_D
from networks.ops import SpectralNorm
from ops.dcn import ModulatedDeformConvPack, modulated_deform_conv
import torch
import torchvision
from networks.full_conv_network import _GlobalConvModule

class FeaFusion(nn.Module):
    def __init__(self,channel_num=64):
        super(FeaFusion, self).__init__()
        self.channel=channel_num
        # channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            # nn.Linear(channel_num, channel_num, bias=False),
            # nn.ReLU(inplace=True),
            nn.Linear(self.channel, self.channel, bias=False),
            nn.Sigmoid()
        )

        # spatial attention
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

        # 1*1conv
        self.conv2 = nn.Conv2d(channel_num,int(channel_num/2),(1,1))
        # global conv layer
        self.globalconv=_GlobalConvModule(channel_num // 2,channel_num // 2,kernel_size=(7, 7))


        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.fc = nn.Sequential(
        #     nn.Linear(channel, channel // reduction, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(channel // reduction, channel, bias=False),
        #     nn.Sigmoid()
        # )
    def forward(self, feat):
        # channel attention
        b, c, h, w = feat.shape
        feat_index=self.avg_pool(feat).view(-1, self.channel)
        feat_index=self.fc(feat_index).unsqueeze(2).unsqueeze(3).repeat(1, 1, h, w)
        feat_mid=feat * feat_index
        # spatial attention
        avgout = torch.mean(feat_mid, dim=1, keepdim=True)
        maxout, _ = torch.max(feat_mid, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv1(out))
        # out是空间权重，feat_mid是通道注意力的输出
        feat_out = feat_mid * out
        # reduce channel num
        out = self.conv2(feat_out)
        # globalconv
        out = self.globalconv(out)

        return out




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

        if LooseVersion(torchvision.__version__) >= LooseVersion('0.9.0'):
            return torchvision.ops.deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.padding,
                                                 self.dilation, mask)
        else:
            return modulated_deform_conv(x, offset, mask, self.weight, self.bias, self.stride, self.padding,
                                         self.dilation, self.groups, self.deformable_groups)
class FeaAlign(nn.Module):

    def __init__(self, channel_num=32, deformable_groups=8):
        super(FeaAlign, self).__init__()

        self.conv1 = nn.Conv2d(channel_num * 2, channel_num, (1, 1))
        self.conv2 = nn.Sequential(
            nn.Conv2d(channel_num, channel_num, (3, 3), (1,1), 1),
            nn.Conv2d(channel_num, channel_num, (3, 3), (1,1), 1),
        )
        self.dcn = DCNv2Pack(channel_num , channel_num, 3, padding=1, deformable_groups=deformable_groups)

    def forward(self, img_ref, img):
        img_cat = self.conv1(torch.cat([img_ref, img], dim=1))
        offset = self.conv2(img_cat)

        feat = self.dcn(img, offset)

        return feat

class AFModule(nn.Module):
    def __init__(self, channel_num=32, deformable_groups=8):
        super(AFModule, self).__init__()
        self.channel_in=channel_num
        self.fea_align_1 = FeaAlign(self.channel_in, deformable_groups)
        self.fea_align_2 = FeaAlign(self.channel_in, deformable_groups)
        self.fea_fusion = FeaFusion(self.channel_in*2)

    def forward(self, img1, img2, img3):
        # tfa   channel_in= channel_num,channel_out= 2*channel_num,
        feat1 = self.fea_align_1(img2, img1)
        feat2 = self.fea_align_2(img2, img3)
        # concat
        feat = torch.cat([feat1,feat2],dim=1)

        # tff   channel_in= 2*channel_num,channel_out= channel_num,
        feat_out = self.fea_fusion(feat)

        return feat_out



class ResAFModule(ResNet_D):

    def __init__(self, block, layers, norm_layer=None, late_downsample=False):
        super(ResAFModule, self).__init__(block, layers, norm_layer, late_downsample=late_downsample)
        first_inplane = 3 + CONFIG.model.mask_channel
        self.shortcut_inplane = [first_inplane, self.midplanes, 64, 128, 256]
        self.shortcut_plane = [32, self.midplanes, 64, 128, 256]

        self.shortcut = nn.ModuleList()
        for stage, inplane in enumerate(self.shortcut_inplane):
            self.shortcut.append(self._make_shortcut(inplane, self.shortcut_plane[stage]))
        # init fusion module
        self.afmodule_inchannel = [32, self.midplanes, 64, 128, 256]
        # self.afmodule_outchannel = [32, self.midplanes, 64, 128, 256]
        self.afmodule_list = nn.ModuleList()
        for stage, inchannel in enumerate(self.afmodule_inchannel):
            self.afmodule_list.append(self._make_afmodule(inchannel))
        # 输入输出一致

    def _make_afmodule(self, channel):
        return AFModule(channel)


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

        #
        # fea1 = self.afmodule_list[0](x)  # input image and mask
        # fea2 = self.afmodule_list[1](x1)
        # fea3 = self.afmodule_list[2](x2)
        # fea4 = self.afmodule_list[3](x3)
        # fea5 = self.afmodule_list[4](x4)


        fea1 = self.shortcut[0](x)  # input image and mask
        fea2 = self.shortcut[1](x1)
        fea3 = self.shortcut[2](x2)
        fea4 = self.shortcut[3](x3)
        fea5 = self.shortcut[4](x4)

        return out, {'shortcut': (fea1, fea2, fea3, fea4, fea5), 'image': x[:, :3, ...]}
