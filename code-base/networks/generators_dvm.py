import torch
import torch.nn as nn

from utils import CONFIG
from networks import encoders, decoders, ops
from .encoders.res_afmodule_enc import AFModule, ResAFModule
from .encoders import BasicBlock


# 原始的MGMatting生成器
class Generator(nn.Module):
    def __init__(self, encoder, decoder):

        super(Generator, self).__init__()

        if encoder not in encoders.__all__:
            raise NotImplementedError("Unknown Encoder {}".format(encoder))
        self.encoder = encoders.__dict__[encoder]()

        self.aspp = ops.ASPP(in_channel=512, out_channel=512)

        if decoder not in decoders.__all__:
            raise NotImplementedError("Unknown Decoder {}".format(decoder))
        self.decoder = decoders.__dict__[decoder]()

    def forward(self, image, guidance):
        inp = torch.cat((image, guidance), dim=1)
        embedding, mid_fea = self.encoder(inp)
        embedding = self.aspp(embedding)
        pred = self.decoder(embedding, mid_fea)

        return pred


def get_generator(encoder, decoder):
    generator = GeneratorForVideo(encoder=encoder, decoder=decoder)
    return generator


# 针对视频修改的MGMatting
class GeneratorForVideo(nn.Module):
    def __init__(self, encoder, decoder):

        super(GeneratorForVideo, self).__init__()

        if encoder not in encoders.__all__:
            raise NotImplementedError("Unknown Encoder {}".format(encoder))
        # encoder
        self.encoder = encoders.__dict__[encoder]()

        # aspp
        self.aspp = ops.ASPP(in_channel=512, out_channel=512)

        if decoder not in decoders.__all__:
            raise NotImplementedError("Unknown Decoder {}".format(decoder))
        self.decoder = decoders.__dict__[decoder]()

        self.resafmodule=ResAFModule(BasicBlock, [3, 4, 4, 2])

    # 修改后网络结构
    def forward(self, image, guidance):
        # image,guidance表
        img_mask, embedding, mid_fea_img, mid_fea, img = [None, None, None], [None, None, None], [None, None, None], \
                                                         [None, None, None], [None, None, None]
        for index in range(3):
            img_mask[index] = torch.cat([image[index:index + 1], guidance[index:index + 1]], dim=1)

            embedding[index], mid_fea_img[index] = self.encoder(img_mask[index])

            mid_fea[index] = mid_fea_img[index]['shortcut']
            img[index] = image[index:index + 1]


        af_fea = {'shortcut':[]}
        for index in range(len(mid_fea[0])):
            af_fea['shortcut'].append(self.resafmodule.afmodule_list[index](mid_fea[0][index], mid_fea[1][index], mid_fea[2][index]))


        # process mid_feature
        # aggregation module

        out = self.aspp(embedding[1])
        pred = self.decoder(out, af_fea)
        return pred


def get_generator_for_video(encoder, decoder):
    generator = GeneratorForVideo(encoder=encoder, decoder=decoder)
    return generator

# 针对图片修改的MGMatting
