# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from . import transforms as T


def build_transforms(cfg, is_train=True):
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        # flip_prob = 0.5  # cfg.INPUT.FLIP_PROB_TRAIN
        # flip_prob = 0
        # rotate_prob = 0.5
        crop_prob = 0.5  # 0.5
        rotate_prob = 0.5  # 0.5
        angle_range = 10  # [-10,10]
        pixel_aug_prob = 0.2
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        # flip_prob = 0
        rotate_prob = 0
        pixel_aug_prob = 0
        random_crop_prob = 0

    to_bgr255 = cfg.INPUT.TO_BGR255
    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255
    )
    if cfg.DATASETS.AUG and is_train:  # True
        if cfg.DATASETS.FIX_CROP:  # False
            transform = T.Compose(
                [
                    T.RandomCrop(1.0, crop_min_size=512, crop_max_size=640, max_trys=50),
                    T.RandomBrightness(pixel_aug_prob),
                    T.RandomContrast(pixel_aug_prob),
                    T.RandomHue(pixel_aug_prob),
                    T.RandomSaturation(pixel_aug_prob),
                    T.RandomGamma(pixel_aug_prob),
                    T.RandomRotate(rotate_prob),
                    T.Resize(min_size, max_size, cfg.INPUT.STRICT_RESIZE),
                    T.ToTensor(),
                    normalize_transform,
                ]
            )
        else:
            transform = T.Compose(
                [
                    # T.RandomCrop(random_crop_prob),  # 0
                    T.RandomBrightness(pixel_aug_prob),
                    T.RandomContrast(pixel_aug_prob),
                    T.RandomHue(pixel_aug_prob),
                    T.RandomSaturation(pixel_aug_prob),
                    T.RandomGamma(pixel_aug_prob),
                    T.RandomRotate(rotate_prob, angle_range),
                    T.RandomCrop(crop_prob),
                    T.Resize(min_size, max_size, cfg.INPUT.STRICT_RESIZE),
                    T.ToTensor(),
                    normalize_transform,
                ]
            )
    else:
        transform = T.Compose(
            [
                T.Resize(min_size, max_size, cfg.INPUT.STRICT_RESIZE),
                T.ToTensor(),
                normalize_transform,
            ]
        )
    return transform