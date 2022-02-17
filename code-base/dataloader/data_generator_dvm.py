import cv2
import os
import math
import numbers
import random
import logging
import numpy as np
import imgaug.augmenters as iaa

import torch
from torch.utils.data import Dataset
from torch.nn import functional as F
from torchvision import transforms

from utils import CONFIG

interp_list = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]


def maybe_random_interp(cv2_interp):
    if CONFIG.data.random_interp:
        return np.random.choice(interp_list)
    else:
        return cv2_interp


# gai
class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors with normalization.
    """

    def __init__(self, phase="test", real_world_aug=False):
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        self.phase = phase
        if real_world_aug:
            self.RWA = iaa.SomeOf((1, None), [
                iaa.LinearContrast((0.6, 1.4)),
                iaa.JpegCompression(compression=(0, 60)),
                iaa.GaussianBlur(sigma=(0.0, 3.0)),
                iaa.AdditiveGaussianNoise(scale=(0, 0.1 * 255))
            ], random_order=True)
        else:
            self.RWA = None

    def __call__(self, sample):
        # convert GBR images to RGB
        for _id in range(3):
            image, alpha, trimap, mask = sample[f'image{_id}'][:, :, ::-1], sample[f'alpha{_id}'], sample[
                f'trimap{_id}'], sample[f'mask{_id}']

            alpha[alpha < 0] = 0
            alpha[alpha > 1] = 1

            if self.phase == 'train' and self.RWA is not None and np.random.rand() < 0.5:
                image[image > 255] = 255
                image[image < 0] = 0
                image = np.round(image).astype(np.uint8)
                image = np.expand_dims(image, axis=0)
                image = self.RWA(images=image)
                image = image[0, ...]

            # swap color axis because
            # numpy image: H x W x C
            # torch image: C X H X W
            image = image.transpose((2, 0, 1)).astype(np.float32)
            alpha = np.expand_dims(alpha.astype(np.float32), axis=0)
            trimap[trimap < 85] = 0
            trimap[trimap >= 170] = 2
            trimap[trimap >= 85] = 1

            mask = np.expand_dims(mask.astype(np.float32), axis=0)

            # normalize image
            image /= 255.

            if self.phase == "train":
                # convert GBR images to RGB
                fg = sample[f'fg{_id}'][:, :, ::-1].transpose((2, 0, 1)).astype(np.float32) / 255.
                sample[f'fg{_id}'] = torch.from_numpy(fg).sub_(self.mean).div_(self.std)
                bg = sample[f'bg{_id}'][:, :, ::-1].transpose((2, 0, 1)).astype(np.float32) / 255.
                sample[f'bg{_id}'] = torch.from_numpy(bg).sub_(self.mean).div_(self.std)
                # del sample['image_name']

            sample[f'image{_id}'], sample[f'alpha{_id}'], sample[f'trimap{_id}'] = \
                torch.from_numpy(image), torch.from_numpy(alpha), torch.from_numpy(trimap).to(torch.long)
            sample[f'image{_id}'] = sample[f'image{_id}'].sub_(self.mean).div_(self.std)

            if CONFIG.model.trimap_channel == 3:
                sample[f'trimap{_id}'] = F.one_hot(sample[f'trimap{_id}'], num_classes=3).permute(2, 0, 1).float()
            elif CONFIG.model.trimap_channel == 1:
                sample[f'trimap{_id}'] = sample[f'trimap{_id}'][None, ...].float()
            else:
                raise NotImplementedError("CONFIG.model.trimap_channel can only be 3 or 1")

            sample[f'mask{_id}'] = torch.from_numpy(mask).float()

        return sample


# gai
class RandomAffine(object):
    """
    Random affine translation
    """

    def __init__(self, degrees, translate=None, scale=None, shear=None, flip=None, resample=False, fillcolor=0):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
                "degrees should be a list or tuple and it must be of length 2."
            self.degrees = degrees

        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            if isinstance(shear, numbers.Number):
                if shear < 0:
                    raise ValueError("If shear is a single number, it must be positive.")
                self.shear = (-shear, shear)
            else:
                assert isinstance(shear, (tuple, list)) and len(shear) == 2, \
                    "shear should be a list or tuple and it must be of length 2."
                self.shear = shear
        else:
            self.shear = shear

        self.resample = resample
        self.fillcolor = fillcolor
        self.flip = flip

    @staticmethod
    def get_params(degrees, translate, scale_ranges, shears, flip, img_size):
        """Get parameters for affine transformation

        Returns:
            sequence: params to be passed to the affine transformation
        """
        angle = random.uniform(degrees[0], degrees[1])
        if translate is not None:
            max_dx = translate[0] * img_size[0]
            max_dy = translate[1] * img_size[1]
            translations = (np.round(random.uniform(-max_dx, max_dx)),
                            np.round(random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = (random.uniform(scale_ranges[0], scale_ranges[1]),
                     random.uniform(scale_ranges[0], scale_ranges[1]))
        else:
            scale = (1.0, 1.0)

        if shears is not None:
            shear = random.uniform(shears[0], shears[1])
        else:
            shear = 0.0

        if flip is not None:
            flip = (np.random.rand(2) < flip).astype(np.int) * 2 - 1

        return angle, translations, scale, shear, flip

    def __call__(self, sample):
        fg, alpha = sample['fg1'], sample['alpha1']
        rows, cols, ch = fg.shape
        if np.maximum(rows, cols) < 1024:
            params = self.get_params((0, 0), self.translate, self.scale, self.shear, self.flip, fg.size)
        else:
            params = self.get_params(self.degrees, self.translate, self.scale, self.shear, self.flip, fg.size)

        center = (cols * 0.5 + 0.5, rows * 0.5 + 0.5)
        M = self._get_inverse_affine_matrix(center, *params)
        M = np.array(M).reshape((2, 3))

        for _id in range(3):
            sample[f'fg{_id}'] = cv2.warpAffine(sample[f'fg{_id}'], M, (cols, rows),
                                                flags=maybe_random_interp(cv2.INTER_NEAREST) + cv2.WARP_INVERSE_MAP)
            sample[f'alpha{_id}'] = cv2.warpAffine(sample[f'alpha{_id}'], M, (cols, rows),
                                                   flags=maybe_random_interp(cv2.INTER_NEAREST) + cv2.WARP_INVERSE_MAP)

        return sample

    @staticmethod
    def _get_inverse_affine_matrix(center, angle, translate, scale, shear, flip):
        # Helper method to compute inverse matrix for affine transformation

        # As it is explained in PIL.Image.rotate
        # We need compute INVERSE of affine transformation matrix: M = T * C * RSS * C^-1
        # where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
        # C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
        # RSS is rotation with scale and shear matrix
        # It is different from the original function in torchvision
        # The order are changed to flip -> scale -> rotation -> shear
        # x and y have different scale factors
        # RSS(shear, a, scale, f) = [ cos(a + shear)*scale_x*f -sin(a + shear)*scale_y     0]
        # [ sin(a)*scale_x*f          cos(a)*scale_y             0]
        # [     0                       0                      1]
        # Thus, the inverse is M^-1 = C * RSS^-1 * C^-1 * T^-1

        angle = math.radians(angle)
        shear = math.radians(shear)
        scale_x = 1.0 / scale[0] * flip[0]
        scale_y = 1.0 / scale[1] * flip[1]

        # Inverted rotation matrix with scale and shear
        d = math.cos(angle + shear) * math.cos(angle) + math.sin(angle + shear) * math.sin(angle)
        matrix = [
            math.cos(angle) * scale_x, math.sin(angle + shear) * scale_x, 0,
            -math.sin(angle) * scale_y, math.cos(angle + shear) * scale_y, 0
        ]
        matrix = [m / d for m in matrix]

        # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
        matrix[2] += matrix[0] * (-center[0] - translate[0]) + matrix[1] * (-center[1] - translate[1])
        matrix[5] += matrix[3] * (-center[0] - translate[0]) + matrix[4] * (-center[1] - translate[1])

        # Apply center translation: C * RSS^-1 * C^-1 * T^-1
        matrix[2] += center[0]
        matrix[5] += center[1]

        return matrix


# gai
class RandomJitter(object):
    """
    Random change the hue of the image
    """

    def __call__(self, sample):
        for _id in range(3):
            fg, alpha = sample[f'fg{_id}'], sample[f'alpha{_id}']
            # if alpha is all 0 skip
            if np.all(alpha == 0):
                return sample
            # convert to HSV space, convert to float32 image to keep precision during space conversion.
            fg = cv2.cvtColor(fg.astype(np.float32) / 255.0, cv2.COLOR_BGR2HSV)
            # Hue noise
            hue_jitter = np.random.randint(-40, 40)
            fg[:, :, 0] = np.remainder(fg[:, :, 0].astype(np.float32) + hue_jitter, 360)
            # Saturation noise
            sat_bar = fg[:, :, 1][alpha > 0].mean()
            sat_jitter = np.random.rand() * (1.1 - sat_bar) / 5 - (1.1 - sat_bar) / 10
            sat = fg[:, :, 1]
            sat = np.abs(sat + sat_jitter)
            sat[sat > 1] = 2 - sat[sat > 1]
            fg[:, :, 1] = sat
            # Value noise
            val_bar = fg[:, :, 2][alpha > 0].mean()
            val_jitter = np.random.rand() * (1.1 - val_bar) / 5 - (1.1 - val_bar) / 10
            val = fg[:, :, 2]
            val = np.abs(val + val_jitter)
            val[val > 1] = 2 - val[val > 1]
            fg[:, :, 2] = val
            # convert back to BGR space
            fg = cv2.cvtColor(fg, cv2.COLOR_HSV2BGR)
            sample[f'fg{_id}'] = fg * 255

        return sample


# gai
class RandomHorizontalFlip(object):
    """
    Random flip image and label horizontally
    """

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):

        # fg, alpha = sample['fg'], sample['alpha']
        # if np.random.uniform(0, 1) < self.prob:
        #     fg = cv2.flip(fg, 1)
        #     alpha = cv2.flip(alpha, 1)
        # sample['fg'], sample['alpha'] = fg, alpha
        if np.random.uniform(0, 1) < self.prob:
            for _id in range(3):
                fg, alpha = sample[f'fg{_id}'], sample[f'alpha{_id}']
                fg = cv2.flip(fg, 1)
                alpha = cv2.flip(alpha, 1)
                sample[f'fg{_id}'], sample[f'alpha{_id}'] = fg, alpha

        return sample


# gai
class RandomCrop(object):
    """
    Crop randomly the image in a sample, retain the center 1/4 images, and resize to 'output_size'

    :param output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size=(CONFIG.data.crop_size, CONFIG.data.crop_size)):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.margin = output_size[0] // 2
        self.logger = logging.getLogger("Logger")

    def __call__(self, sample):
        for _id in range(3):
            fg, alpha, trimap, mask, name = sample[f'fg{_id}'], sample[f'alpha{_id}'], sample[f'trimap{_id}'], sample[
                f'mask{_id}'], sample[
                                                f'image_name{_id}']
            bg = sample[f'bg{_id}']
            h, w = trimap.shape
            bg = cv2.resize(bg, (w, h), interpolation=maybe_random_interp(cv2.INTER_CUBIC))
            if w < self.output_size[0] + 1 or h < self.output_size[1] + 1:
                ratio = 1.1 * self.output_size[0] / h if h < w else 1.1 * self.output_size[1] / w
                # self.logger.warning("Size of {} is {}.".format(name, (h, w)))
                while h < self.output_size[0] + 1 or w < self.output_size[1] + 1:
                    fg = cv2.resize(fg, (int(w * ratio), int(h * ratio)),
                                    interpolation=maybe_random_interp(cv2.INTER_NEAREST))
                    alpha = cv2.resize(alpha, (int(w * ratio), int(h * ratio)),
                                       interpolation=maybe_random_interp(cv2.INTER_NEAREST))
                    trimap = cv2.resize(trimap, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_NEAREST)
                    bg = cv2.resize(bg, (int(w * ratio), int(h * ratio)),
                                    interpolation=maybe_random_interp(cv2.INTER_CUBIC))
                    mask = cv2.resize(mask, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_NEAREST)
                    h, w = trimap.shape
            small_trimap = cv2.resize(trimap, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST)
            unknown_list = list(zip(*np.where(small_trimap[self.margin // 4:(h - self.margin) // 4,
                                              self.margin // 4:(w - self.margin) // 4] == 128)))
            unknown_num = len(unknown_list)
            if len(unknown_list) < 10:
                left_top = (
                    np.random.randint(0, h - self.output_size[0] + 1),
                    np.random.randint(0, w - self.output_size[1] + 1))
            else:
                idx = np.random.randint(unknown_num)
                left_top = (unknown_list[idx][0] * 4, unknown_list[idx][1] * 4)

            fg_crop = fg[left_top[0]:left_top[0] + self.output_size[0], left_top[1]:left_top[1] + self.output_size[1],
                      :]
            alpha_crop = alpha[left_top[0]:left_top[0] + self.output_size[0],
                         left_top[1]:left_top[1] + self.output_size[1]]
            bg_crop = bg[left_top[0]:left_top[0] + self.output_size[0], left_top[1]:left_top[1] + self.output_size[1],
                      :]
            trimap_crop = trimap[left_top[0]:left_top[0] + self.output_size[0],
                          left_top[1]:left_top[1] + self.output_size[1]]
            mask_crop = mask[left_top[0]:left_top[0] + self.output_size[0],
                        left_top[1]:left_top[1] + self.output_size[1]]

            if len(np.where(trimap == 128)[0]) == 0:
                self.logger.error("{} does not have enough unknown area for crop. Resized to target size."
                                  "left_top: {}".format(name, left_top))
                fg_crop = cv2.resize(fg, self.output_size[::-1], interpolation=maybe_random_interp(cv2.INTER_NEAREST))
                alpha_crop = cv2.resize(alpha, self.output_size[::-1],
                                        interpolation=maybe_random_interp(cv2.INTER_NEAREST))
                trimap_crop = cv2.resize(trimap, self.output_size[::-1], interpolation=cv2.INTER_NEAREST)
                bg_crop = cv2.resize(bg, self.output_size[::-1], interpolation=maybe_random_interp(cv2.INTER_CUBIC))
                mask_crop = cv2.resize(mask, self.output_size[::-1], interpolation=cv2.INTER_NEAREST)

            sample.update(
                {f'fg{_id}': fg_crop, f'alpha{_id}': alpha_crop, f'trimap{_id}': trimap_crop, f'mask{_id}': mask_crop,
                 f'bg{_id}': bg_crop})
        return sample


class Resize(object):
    def __init__(self, max_wxh):
        self.max_wxh = max_wxh

    def __call__(self, sample):
        h, w = sample["alpha1"].shape[0:2]
        if h * w < self.max_wxh:
            return sample
        for key in ["image", "alpha", "trimap", "mask"]:
            for id in range(3):
                target_h = 32 * ((h // 2 - 1) // 32 + 1)
                target_w = 32 * ((w // 2 - 1) // 32 + 1)

                sample[f"{key}{id}"] = cv2.resize(sample[f"{key}{id}"], (target_h, target_w), cv2.INTER_NEAREST)

        return sample


# gai
class OriginScale(object):
    def __call__(self, sample):
        h, w = sample["alpha_shape"]

        if h % 32 == 0 and w % 32 == 0:
            return sample
        target_h = 32 * ((h - 1) // 32 + 1)
        target_w = 32 * ((w - 1) // 32 + 1)
        pad_h = target_h - h
        pad_w = target_w - w
        for _id in range(3):
            padded_image = np.pad(sample[f'image{_id}'], ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            padded_trimap = np.pad(sample[f'trimap{_id}'], ((0, pad_h), (0, pad_w)), mode="reflect")
            padded_mask = np.pad(sample[f'mask{_id}'], ((0, pad_h), (0, pad_w)), mode="reflect")
            sample[f'image{_id}'] = padded_image
            sample[f'trimap{_id}'] = padded_trimap
            sample[f'mask{_id}'] = padded_mask

        return sample


# gai
class GenMask(object):
    def __init__(self):
        self.erosion_kernels = [None] + [cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)) for size in
                                         range(1, 100)]

    def __call__(self, sample):
        for _id in range(3):
            alpha = sample[f'alpha{_id}']
            h, w = alpha.shape

            max_kernel_size = max(30, int((min(h, w) / 2048) * 30))

            ### generate trimap
            fg_mask = (alpha + 1e-5).astype(np.int).astype(np.uint8)
            bg_mask = (1 - alpha + 1e-5).astype(np.int).astype(np.uint8)
            fg_mask = cv2.erode(fg_mask, self.erosion_kernels[np.random.randint(1, max_kernel_size)])
            bg_mask = cv2.erode(bg_mask, self.erosion_kernels[np.random.randint(1, max_kernel_size)])

            fg_width = np.random.randint(1, 30)
            bg_width = np.random.randint(1, 30)
            fg_mask = (alpha + 1e-5).astype(np.int).astype(np.uint8)
            bg_mask = (1 - alpha + 1e-5).astype(np.int).astype(np.uint8)
            fg_mask = cv2.erode(fg_mask, self.erosion_kernels[fg_width])
            bg_mask = cv2.erode(bg_mask, self.erosion_kernels[bg_width])

            trimap = np.ones_like(alpha) * 128
            trimap[fg_mask == 1] = 255
            trimap[bg_mask == 1] = 0

            trimap = cv2.resize(trimap, (w, h), interpolation=cv2.INTER_NEAREST)
            sample[f'trimap{_id}'] = trimap

            ### generate mask
            low = 0.01
            high = 1.0
            thres = random.random() * (high - low) + low
            seg_mask = (alpha >= thres).astype(np.int).astype(np.uint8)
            random_num = random.randint(0, 3)
            if random_num == 0:
                seg_mask = cv2.erode(seg_mask, self.erosion_kernels[np.random.randint(1, max_kernel_size)])
            elif random_num == 1:
                seg_mask = cv2.dilate(seg_mask, self.erosion_kernels[np.random.randint(1, max_kernel_size)])
            elif random_num == 2:
                seg_mask = cv2.erode(seg_mask, self.erosion_kernels[np.random.randint(1, max_kernel_size)])
                seg_mask = cv2.dilate(seg_mask, self.erosion_kernels[np.random.randint(1, max_kernel_size)])
            elif random_num == 3:
                seg_mask = cv2.dilate(seg_mask, self.erosion_kernels[np.random.randint(1, max_kernel_size)])
                seg_mask = cv2.erode(seg_mask, self.erosion_kernels[np.random.randint(1, max_kernel_size)])

            sample[f'mask{_id}'] = seg_mask.astype(np.float32)

        return sample


# gai
class Composite(object):
    def __call__(self, sample):
        for _id in range(3):
            fg, bg, alpha = sample[f'fg{_id}'], sample[f'bg{_id}'], sample[f'alpha{_id}']
            alpha[alpha < 0] = 0
            alpha[alpha > 1] = 1
            fg[fg < 0] = 0
            fg[fg > 255] = 255
            bg[bg < 0] = 0
            bg[bg > 255] = 255

            image = fg * alpha[:, :, None] + bg * (1 - alpha[:, :, None])
            sample[f'image{_id}'] = image
        return sample


# gai
class CutMask(object):
    def __init__(self, perturb_prob=0):
        self.perturb_prob = perturb_prob

    def __call__(self, sample):
        if np.random.rand() < self.perturb_prob:
            mask = sample['mask1']  # H x W, trimap 0--255, segmask 0--1, alpha 0--1
            h, w = mask.shape
            perturb_size_h, perturb_size_w = random.randint(h // 4, h // 2), random.randint(w // 4, w // 2)
            x = random.randint(0, h - perturb_size_h)
            y = random.randint(0, w - perturb_size_w)
            x1 = random.randint(0, h - perturb_size_h)
            y1 = random.randint(0, w - perturb_size_w)

            for _id in range(3):
                sample[f'mask{_id}'][x:x + perturb_size_h, y:y + perturb_size_w] = sample[f'mask{_id}'][
                                                                                   x1:x1 + perturb_size_h,
                                                                                   y1:y1 + perturb_size_w].copy()

        return sample


class DataGenerator(Dataset):
    def __init__(self, data_root, phase="train"):
        self.phase = phase
        self.crop_size = CONFIG.data.crop_size

        def is_valid_path(_path):
            assert os.path.exists(_path), f'path <{_path}> is not exist'

        self.alpha_dir = os.path.join(data_root, 'a')
        is_valid_path(self.alpha_dir)

        if self.phase == "train":
            self.fg_dir = os.path.join(data_root, 'f')
            is_valid_path(self.fg_dir)
            self.list_fg_dir = os.listdir(self.fg_dir)

            self.bg_dir = os.path.join(data_root, 'b')
            is_valid_path(self.bg_dir)
            self.list_bg_dir = os.listdir(self.bg_dir)

            self.list_merged_dir = []
            self.list_trimap_dir = []

        else:
            self.list_fg_dir = []
            self.list_bg_dir = []

            self.merged_dir = os.path.join(data_root, 'v')
            is_valid_path(self.merged_dir)
            self.list_merged_dir = os.listdir(self.merged_dir)

            self.trimap_dir = os.path.join(data_root, 'trimap')
            is_valid_path(self.trimap_dir)
            self.list_trimap_dir = os.listdir(self.trimap_dir)

        train_trans = [
            RandomAffine(degrees=30, scale=[0.8, 1.25], shear=10, flip=0.5),
            GenMask(),
            CutMask(perturb_prob=CONFIG.data.cutmask_prob),
            RandomCrop((self.crop_size, self.crop_size)),
            RandomJitter(),
            Composite(),
            ToTensor(phase="train", real_world_aug=CONFIG.data.real_world_aug)]

        test_trans = [OriginScale(), ToTensor()]

        self.transform = {
            'train':
                transforms.Compose(train_trans),
            'val':
                transforms.Compose([
                    OriginScale(),
                    Resize(1440 * 1920),
                    ToTensor()
                ]),
            'test':
                transforms.Compose(test_trans)
        }[phase]

        self.data_num = len(self.list_fg_dir)

    def __getitem__(self, idx):

        if self.phase == "train":
            fg, alpha, bg, img_name, data_folder_name = self.get_three_data(idx)

            fg, alpha = self._composite_fg(fg, alpha, idx)

            sample = {'data_folder_name': data_folder_name}

            for _id in range(3):
                sample[f'fg{_id}'] = fg[_id]
                sample[f'alpha{_id}'] = alpha[_id]
                sample[f'bg{_id}'] = bg[_id]
                sample[f'image_name{_id}'] = img_name[_id]

        else:
            data_folder_name = self.list_merged_dir[idx]
            merged_folder = os.path.join(self.merged_dir, data_folder_name)
            list_merged = os.listdir(merged_folder)

            assert len(list_merged) > 3, f'data {data_folder_name} should have more than 3 images !!!'

            rd_id = np.random.randint(1, len(list_merged) - 1)

            img_name = [list_merged[rd_id - 1], list_merged[rd_id], list_merged[rd_id + 1]]

            image = [cv2.imread(os.path.join(self.merged_dir, data_folder_name, name)) for name in img_name]
            alpha = [cv2.imread(os.path.join(self.alpha_dir, data_folder_name, name), 0).astype(np.float32) / 255 for
                     name in img_name]
            trimap = [cv2.imread(os.path.join(self.trimap_dir, data_folder_name, name), 0) for name in img_name]
            mask = [(t >= 170).astype(np.float32) for t in trimap]

            sample = {'alpha_shape': alpha[0].shape, 'data_folder_name': data_folder_name}

            for _id in range(3):
                sample[f'image{_id}'] = image[_id]
                sample[f'alpha{_id}'] = alpha[_id]
                sample[f'trimap{_id}'] = trimap[_id]
                sample[f'mask{_id}'] = mask[_id]
                sample[f'image_name{_id}'] = img_name[_id]

        sample = self.transform(sample)

        return sample

    def get_three_data(self, idx):
        data_folder_name = self.list_fg_dir[idx % self.data_num]
        fg_folder = os.path.join(self.fg_dir, data_folder_name)
        list_fg = os.listdir(fg_folder)

        assert len(list_fg) > 3, f'data {data_folder_name} should have more than 3 images !!!'

        rd_id = np.random.randint(1, len(list_fg) - 1)

        img_name = [list_fg[rd_id - 1], list_fg[rd_id], list_fg[rd_id + 1]]

        fg = [cv2.imread(os.path.join(self.fg_dir, data_folder_name, name)) for name in img_name]
        alpha = [cv2.imread(os.path.join(self.alpha_dir, data_folder_name, name), 0).astype(np.float32) / 255 for name
                 in img_name]
        bg = [cv2.imread(os.path.join(self.bg_dir, data_folder_name, name), 1) for name in img_name]

        return fg, alpha, bg, img_name, data_folder_name

    def _composite_fg(self, fg, alpha, idx):

        if np.random.rand() < 0.5:
            idx2 = np.random.randint(self.data_num) + idx
            fg2, alpha2, bg2, img_name2, data_folder_name2 = self.get_three_data(idx2)

            h, w = alpha[0].shape
            alpha_tmp = [a for a in alpha]
            for _id in range(3):
                fg2[_id] = cv2.resize(fg2[_id], (w, h), interpolation=maybe_random_interp(cv2.INTER_NEAREST))
                alpha2[_id] = cv2.resize(alpha2[_id], (w, h), interpolation=maybe_random_interp(cv2.INTER_NEAREST))

                alpha_tmp[_id] = 1 - (1 - alpha[_id]) * (1 - alpha2[_id])
                if np.any(alpha_tmp[_id] < 1):
                    fg[_id] = fg[_id].astype(np.float32) * alpha[_id][:, :, None] + fg2[_id].astype(np.float32) * (
                            1 - alpha[_id][:, :, None])
                    # The overlap of two 50% transparency should be 25%
                    alpha[_id] = alpha_tmp[_id]
                    fg[_id] = fg[_id].astype(np.uint8)

        if np.random.rand() < 0.25:
            fg = [cv2.resize(f, (640, 640), interpolation=maybe_random_interp(cv2.INTER_NEAREST)) for f in fg]
            alpha = [cv2.resize(a, (640, 640), interpolation=maybe_random_interp(cv2.INTER_NEAREST)) for a in alpha]

        return fg, alpha

    def __len__(self):
        if self.phase == "train":
            return len(self.list_bg_dir)
        else:
            # return len(self.list_merged_dir)
            return 50