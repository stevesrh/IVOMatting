import torch
import torch.nn as nn

from   utils import CONFIG
from   networks import encoders, decoders, ops

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
    generator = Generator(encoder=encoder, decoder=decoder)
    return generator


# 针对视频修改的MGMatting
class GeneratorForVideo(nn.Module):
    def __init__(self, encoder, decoder):

        super(GeneratorForVideo, self).__init__()

        if encoder not in encoders.__all__:
            raise NotImplementedError("Unknown Encoder {}".format(encoder))
        # encoder
        self.encoder= encoders.__dict__[encoder]()

        # aspp
        self.aspp = ops.ASPP(in_channel=512, out_channel=512)

        if decoder not in decoders.__all__:
            raise NotImplementedError("Unknown Decoder {}".format(decoder))
        self.decoder = decoders.__dict__[decoder]()
    # 修改后网络结构
    def forward(self, image, guidance):
        # image,guidance表
        input1 = torch.cat((image[0], guidance[0]), dim=1)
        input2 = torch.cat((image[1], guidance[1]), dim=1)
        input3 = torch.cat((image[2], guidance[2]), dim=1)
        embedding1, mid_fea1 = self.encoder(input1)
        embedding2, mid_fea2 = self.encoder(input2)
        embedding3, mid_fea3 = self.encoder(input3)
        # process mid_feature
        mid_fea_in = mid_fea1+mid_fea2+mid_fea3
        out = self.aspp(embedding2)
        pred = self.decoder(out, mid_fea_in)
        return pred


def get_generator_for_video(encoder, decoder):
    generator = GeneratorForVideo(encoder=encoder, decoder=decoder)
    return generator


# 针对图片修改的MGMatting
