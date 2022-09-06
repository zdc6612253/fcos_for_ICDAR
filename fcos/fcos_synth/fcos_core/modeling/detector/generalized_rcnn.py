# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn
import numpy as np
import cv2

from fcos_core.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)  # 没有作用

    def score_maps_to_images_shape(self,images,score_maps):
        new_score_maps = []
        image_num = images.tensors.shape[0]
        images_h, images_w = images.tensors.shape[-2:]  # 一个batch的images大小肯定一样
        batched_imgs = torch.zeros([image_num,1,images_h,images_w])
        for img, pad_img in zip(score_maps, batched_imgs):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            new_score_maps.append(pad_img)
        return torch.stack(new_score_maps,0).to(images.tensors.device)

    def trans_tensor_shape(self,tensors,expected_h,expected_w):
        device = tensors.device
        array1=tensors.cpu().numpy()
        maxValue=array1.max()
        array1=array1*255/maxValue
        mat=np.uint8(array1)
        mat=mat.transpose(1,2,0)
        # print('mat_shape:',mat.shape)
        mat2 = torch.from_numpy(cv2.resize(mat,(expected_w,expected_h),interpolation=cv2.INTER_NEAREST)).unsqueeze(0)
        # 变成h行w列
        mat2 = (mat2*maxValue/255).round().int()
        # for i in range(int(maxValue)):
        #     print(i+1 in mat2)
        return mat2.to(device)

    def score_maps_to_features_shape(self,features,score_maps):
        score_maps_list = []
        for i in range(len(features)):  # fpn层  features: 5,2,256,92,160
            feature = features[i]
            batch_size,_,feature_h, feature_w = feature.shape  # feature_h, feature_w在不同fpn是不同的
            # print(feature_h, feature_w)
            image_tensor = []
            for j in range(batch_size):
                shaped_score_maps = self.trans_tensor_shape(score_maps[j],feature_h,feature_w)
                image_tensor.append(shaped_score_maps)
            score_maps_list.append(torch.stack(image_tensor,0))
        return score_maps_list  # shape: 5,2,1,92,160

    def forward(self, images, targets=None, score_maps=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        # print(images.tensors.shape)
        score_maps = self.score_maps_to_images_shape(images, score_maps)  # 将score_maps转化成与images同样的宽高尺寸
        # print(score_maps.shape)  # shape:2,763,1280
        images = to_image_list(images)  # 没啥作用
        features = self.backbone(images.tensors)  # [torch.size(2,256,92,160),(2,256,46,80),,,]
        # for feature in features:
        #     print(feature.shape)

        score_maps_list = self.score_maps_to_features_shape(features, score_maps)  # 为不同fpn层制作score map

        # for score_map in score_maps_list:
        #     print(score_map.shape)
        #     print(score_map.device)

        proposals, proposal_losses = self.rpn(images, features, targets, score_maps_list)
        # print(proposals)
        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses
        return result
