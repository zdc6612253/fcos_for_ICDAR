"""
This file contains specific functions for computing losses of FCOS
file
"""
import torch
from torch.nn import functional as F
from torch import ne, neg_, nn
import os
from ..utils import concat_box_prediction_layers
from fcos_core.layers import IOULoss
from fcos_core.layers import SigmoidFocalLoss, SigmoidQualityFocalLoss
from fcos_core.layers import EastGeoLoss
from fcos_core.modeling.matcher import Matcher
from fcos_core.modeling.utils import cat
from fcos_core.structures.boxlist_ops import boxlist_iou
from fcos_core.structures.boxlist_ops import cat_boxlist
from .get_pos_sample import isPoiWithinPoly

INF = 100000000


def get_num_gpus():
    return int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1


def reduce_sum(tensor):
    if get_num_gpus() <= 1:
        return tensor
    import torch.distributed as dist
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.reduce_op.SUM)
    return tensor


class FCOSLossComputation(object):
    """
    This class computes the FCOS losses.
    """
    def __init__(self, cfg):
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.center_sampling_radius = cfg.MODEL.FCOS.CENTER_SAMPLING_RADIUS  # 1.5
        self.iou_loss_type = cfg.MODEL.FCOS.IOU_LOSS_TYPE
        self.norm_reg_targets = cfg.MODEL.FCOS.NORM_REG_TARGETS
        # loss function
        self.use_gfloss = False
        if cfg.MODEL.FCOS.USE_GFLOSS:
            self.use_gfloss = True
            self.qfloss_weight = cfg.MODEL.FCOS.QFLOSS_WEIGHT
            self.cls_loss_func =  SigmoidQualityFocalLoss(cfg.MODEL.FCOS.GFLOSS_BETA)
        else:
            self.cls_loss_func = SigmoidFocalLoss(
                cfg.MODEL.FCOS.LOSS_GAMMA,
                cfg.MODEL.FCOS.LOSS_ALPHA
            )
        self.box_reg_loss_func = EastGeoLoss()
        self.centerness_loss_func = nn.BCEWithLogitsLoss(reduction="sum")

    def prepare_targets(self, points, targets, score_maps):
        object_sizes_of_interest = [
            [-1, 64],
            [64, 128],
            [128, 256],
            [256, 512],
            [512, INF],
        ]
        expanded_object_sizes_of_interest = []
        for l, points_per_level in enumerate(points):
            object_sizes_of_interest_per_level = \
                points_per_level.new_tensor(object_sizes_of_interest[l])
            expanded_object_sizes_of_interest.append(
                object_sizes_of_interest_per_level[None].expand(len(points_per_level), -1)
            )

        expanded_object_sizes_of_interest = torch.cat(expanded_object_sizes_of_interest, dim=0)
        num_points_per_level = [len(points_per_level) for points_per_level in points]
        self.num_points_per_level = num_points_per_level
        points_all_level = torch.cat(points, dim=0)
        labels, reg_targets, difficults, reg_anchors = self.compute_targets_for_locations(
            points_all_level, targets, expanded_object_sizes_of_interest, score_maps
        )
        for i in range(len(labels)):  # batch size
            labels[i] = torch.split(labels[i], num_points_per_level, dim=0)
            difficults[i] = torch.split(difficults[i], num_points_per_level, dim=0)
            reg_targets[i] = torch.split(reg_targets[i], num_points_per_level, dim=0)
            reg_anchors[i] = torch.split(reg_anchors[i], num_points_per_level, dim=0)

        labels_level_first = []
        reg_targets_level_first = []
        difficult_level_first = []
        reg_anchors_level_first = []
        for level in range(len(points)):
            labels_level_first.append(
                torch.cat([labels_per_im[level] for labels_per_im in labels], dim=0)
            )
            difficult_level_first.append(
                torch.cat([difficults_per_im[level] for difficults_per_im in difficults], dim=0)
            )
            # reg_targets
            reg_targets_per_level = torch.cat([
                reg_targets_per_im[level]
                for reg_targets_per_im in reg_targets
            ], dim=0)

            if self.norm_reg_targets:  # True
                reg_targets_per_level = reg_targets_per_level / self.fpn_strides[level]
            reg_targets_level_first.append(reg_targets_per_level)

            # anchors
            reg_anchors_per_level = torch.cat([
                reg_anchors_per_im[level]
                for reg_anchors_per_im in reg_anchors
            ], dim=0)
            if self.norm_reg_targets:  # True
                reg_anchors_per_level = reg_anchors_per_level / self.fpn_strides[level]
            reg_anchors_level_first.append(reg_anchors_per_level)

        return labels_level_first, reg_targets_level_first, difficult_level_first, reg_anchors_level_first

    def compute_targets_for_locations(self, locations, targets, object_sizes_of_interest, score_map_list):
        """
        locations: torch.Size([21486,2]) ,表示所有fpn层的每个每个位置的x,y坐标
        targets: shape:[1,2,,,batch_size]=batch 1.bbox:torch.Size([3, 8])表示一张图片3个GT
        score_maps : shape: [torch.Size([2, 1, 92, 160]),torch.Size([2, 1, 46, 80]),,,]
        """
        labels = []
        difficults = []
        reg_targets = []
        reg_anchors = []
        xs, ys = locations[:, 0], locations[:, 1]  # torch.Size([21486,]) 
        # print(score_maps) 
        # location是原图大小的位置，不同fpn层间隔不同
        for im_i in range(len(targets)):  # batch size 图片的数量
            score_maps_result_list = []
            for level, score_maps in enumerate(score_map_list):
                score_map = score_maps[im_i].reshape(-1)  # score_maps[im_i] shape: torch.Size([1, 92, 160])
                score_maps_result_list.append(score_map)
            score_maps_result_list = torch.cat(score_maps_result_list,0)  # 19620 

            targets_per_im = targets[im_i]
            if len(targets_per_im.bbox) == 0:
                print(targets_per_im)
                print(im_i)
                print('this pic dont have bbox')
            assert targets_per_im.mode == "xyn"
            bboxes = targets_per_im.bbox  # 3*8的GT坐标,torch.Size([3, 8])
            bboxes_result_list = list(range(1,len(bboxes)+1))
            bboxes_result_list = torch.tensor(bboxes_result_list).to(bboxes.device)
            labels_per_im = targets_per_im.get_field("labels")  # [1]也可能有其他数字，代表其他类别 ,shape:3 
            difficult_per_im = targets_per_im.difficult  # 表示GT是否是困难样本
            area = targets_per_im.area()  # 外接矩形面积,shape = GT Number 4

            x1 = xs[:, None] - bboxes[:, 0][None]  # x1:torch.Size([21486, 3]),表示每个location与每个gt坐标的差距
            y1 = ys[:, None] - bboxes[:, 1][None]
            x2 = xs[:, None] - bboxes[:, 2][None]
            y2 = ys[:, None] - bboxes[:, 3][None]
            x3 = xs[:, None] - bboxes[:, 4][None]
            y3 = ys[:, None] - bboxes[:, 5][None]
            x4 = xs[:, None] - bboxes[:, 6][None]
            y4 = ys[:, None] - bboxes[:, 7][None]
            reg_targets_per_im = torch.stack([x1,y1,x2,y2,x3,y3,x4,y4], dim=2)  # torch.Size([21486, 3, 8])
            anchors_per_im = torch.stack([xs, ys], dim=1)  # torch.Size([21486, 2])

            # select pos_neg samples
            is_in_boxes = score_maps_result_list[:,None] == bboxes_result_list
            max_reg_targets_per_im = abs(reg_targets_per_im).max(dim=2)[0]
            # limit the regression range for each location
            is_cared_in_the_level = \
                (max_reg_targets_per_im >= object_sizes_of_interest[:, [0]]) & \
                (max_reg_targets_per_im <= object_sizes_of_interest[:, [1]])  # 看最长边是不是在fpn这个范围内

            locations_to_gt_area = area[None].repeat(len(locations), 1)  # torch.Size([21486, 6]) 6表示GT数量
            locations_to_gt_area[is_in_boxes == 0] = INF  # 不是正样本的坐标点面积设置无穷
            locations_to_gt_area[is_cared_in_the_level == 0] = INF  # 不在fpn范围内的GT内的坐标点也设为无穷
            # if there are still more than one objects for a location,
            # we choose the one with minimal area
            locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(dim=1)  # locations_to_gt_inds是GT的索引
            reg_targets_per_im = reg_targets_per_im[range(len(locations)), locations_to_gt_inds]  # torch.Size([21486, 8])，每个pos对应的GT
            labels_per_im = labels_per_im[locations_to_gt_inds]  # shape: 21486 每个location对应的GT的label [1,1,1,1,,,]由于只有1个类别，我们这里是1
            difficult_per_im = difficult_per_im[locations_to_gt_inds]
            labels_per_im[locations_to_min_area == INF] = 0  # 让负样本点的label=0
            labels.append(labels_per_im)
            difficults.append(difficult_per_im)

            reg_targets.append(reg_targets_per_im)
            reg_anchors.append(anchors_per_im)
        return labels, reg_targets, difficults, reg_anchors

    def compute_centerness_targets(self, reg_targets):
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                      (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness)

    def __call__(self, locations, box_cls, box_regression, centerness, targets, score_maps):
        """
        Arguments:
            locations (list[BoxList]) 每个fpn的层在原图所对应的x,y坐标
            box_cls (list[Tensor])
            box_regression (list[Tensor])
            centerness (list[Tensor])
            targets (list[BoxList])

        Returns:
            cls_loss (Tensor)
            reg_loss (Tensor)
            centerness_loss (Tensor)
        """
        # box_cls: [torch.Size([4, 1, 96, 168]),torch.Size([4, 1, 48, 84]),,,torch.Size([4, 1, 6, 11])] 共5个
        # locations:[[16128, 2],,,[66, 2]] 共5个
        # targets:[BoxList(num_boxes=2, image_width=1333, image_height=750, mode=xyn),,
        # BoxList(num_boxes=2, image_width=1333, image_height=750, mode=xyn)] 共4个
        # box_regression : [torch.tensor(2,256,92,160),torch.tensor(2,256,46,80),,] 共5个fpn层
        N = box_cls[0].size(0)  # N指batch size, box_cls.size(0)是fpn层数
        num_classes = box_cls[0].size(1)  # 类别
        labels, reg_targets, difficults, reg_anchors = self.prepare_targets(locations, targets, score_maps)  
        # labels由0和其他数字组成，1，2，，各代表不同的类别

        box_cls_flatten = []
        box_regression_flatten = []
        centerness_flatten = []
        labels_flatten = []
        difficults_flatten = []
        reg_targets_flatten = []
        # score_map_flatten = []
        reg_anchors_flatten = []
        for l in range(len(labels)):  # fpn数量
            box_cls_flatten.append(box_cls[l].permute(0, 2, 3, 1).reshape(-1, num_classes))
            box_regression_flatten.append(box_regression[l].permute(0, 2, 3, 1).reshape(-1, 8))
            labels_flatten.append(labels[l].reshape(-1))  # labels[l] shape: torch.Size([64512])=torch.Size([4, 1, 96, 168])
            reg_targets_flatten.append(reg_targets[l].reshape(-1, 8))
            difficults_flatten.append(difficults[l].reshape(-1))
            reg_anchors_flatten.append(reg_anchors[l].reshape(-1, 2))
            # score_map_flatten.append(score_maps[l].permute(0, 2, 3, 1).reshape(-1))  # batchsize,h,w,channel
            # centerness_flatten.append(centerness[l].reshape(-1))

        box_cls_flatten = torch.cat(box_cls_flatten, dim=0)
        box_regression_flatten = torch.cat(box_regression_flatten, dim=0)  # torch.Size([85944，8])
        labels_flatten = torch.cat(labels_flatten, dim=0)  # torch.Size([85944])=torch.Size([64512])+...+
        reg_targets_flatten = torch.cat(reg_targets_flatten, dim=0)
        difficults_flatten = torch.cat(difficults_flatten, dim=0)
        reg_anchors_flatten = torch.cat(reg_anchors_flatten, dim=0)

        pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1)  # label中等于INF（负样本）的为0，返回非0的索引，这个.squeeze(1)好像没啥用
        not_ignore_inds = torch.where(difficults_flatten==0)[0]  # 找到非0的索引
        pos_inds_1 = pos_inds.repeat(len(not_ignore_inds),1).t()
        pos_inds_2 = pos_inds_1 == not_ignore_inds
        pos_inds_3 = torch.sum(pos_inds_2,0)
        pos_inds_inter_not_neg = not_ignore_inds[pos_inds_3.bool()]  # 正样本且不是ignore

        # do classification loss
        num_gpus = get_num_gpus()
        # sync num_pos from all gpus
        total_num_pos = reduce_sum(pos_inds.new_tensor([pos_inds.numel()])).item()
        num_pos_avg_per_gpu = max(total_num_pos / float(num_gpus), 1.0)
        if self.use_gfloss:
            cls_loss = self.qfloss_weight * self.cls_loss_func(
                box_cls_flatten[not_ignore_inds],
                labels_flatten[not_ignore_inds].int(),
                box_regression_flatten[not_ignore_inds].detach(),
                reg_targets_flatten[not_ignore_inds],
                reg_anchors_flatten[not_ignore_inds]
            ) / num_pos_avg_per_gpu
        else:
            cls_loss = self.cls_loss_func(
                box_cls_flatten[not_ignore_inds],
                labels_flatten[not_ignore_inds].int()
            ) / num_pos_avg_per_gpu


        # do regression loss
        if pos_inds_inter_not_neg.numel() > 0:
            reg_weights = box_cls_flatten.detach().sigmoid()
            reg_weights = reg_weights[0][pos_inds_inter_not_neg]
            sum_centerness_targets_avg_per_gpu = 1
            reg_loss = self.box_reg_loss_func(
                box_regression_flatten[pos_inds_inter_not_neg],
                reg_targets_flatten[pos_inds_inter_not_neg],
                reg_weights
            ) / sum_centerness_targets_avg_per_gpu
        else:
            reg_loss = torch.tensor(0.).to(labels_flatten.device)

        return cls_loss, reg_loss


def make_fcos_loss_evaluator(cfg):
    loss_evaluator = FCOSLossComputation(cfg)
    return loss_evaluator
