"""
This file contains specific functions for computing losses of FCOS
file
"""

import torch
from torch.nn import functional as F
from torch import nn
import os
from ..utils import concat_box_prediction_layers
from fcos_core.layers import IOULoss
from fcos_core.layers import SigmoidFocalLoss
from fcos_core.layers import smooth_l1_loss
from fcos_core.modeling.matcher import Matcher
from fcos_core.modeling.utils import cat
from fcos_core.structures.boxlist_ops import boxlist_iou
from fcos_core.structures.boxlist_ops import cat_boxlist


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
        self.cls_loss_func = SigmoidFocalLoss(
            cfg.MODEL.FCOS.LOSS_GAMMA,
            cfg.MODEL.FCOS.LOSS_ALPHA
        )
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.center_sampling_radius = cfg.MODEL.FCOS.CENTER_SAMPLING_RADIUS  # 1.5
        self.iou_loss_type = cfg.MODEL.FCOS.IOU_LOSS_TYPE
        self.norm_reg_targets = cfg.MODEL.FCOS.NORM_REG_TARGETS

        # we make use of IOU Loss for bounding boxes regression,
        # but we found that L1 in log scale can yield a similar performance
        # self.box_reg_loss_func = IOULoss(self.iou_loss_type)
        self.box_reg_loss_func = smooth_l1_loss
        self.centerness_loss_func = nn.BCEWithLogitsLoss(reduction="sum")

    def get_sample_region(self, gt, strides, num_points_per, gt_xs, gt_ys, radius=1.0):
        '''
        This code is from
        https://github.com/yqyao/FCOS_PLUS/blob/0d20ba34ccc316650d8c30febb2eb40cb6eaae37/
        maskrcnn_benchmark/modeling/rpn/fcos/loss.py#L42
        '''
        num_gts = gt.shape[0]
        K = len(gt_xs)  # 所有fpn的特征点的数目,21486=16128+4032+1008+252+66
        # assert False
        gt = gt[None].expand(K, num_gts, 8)  # 每个特征点预测k个类别8个数值,[21486, 3, 8]
        center_x = (gt[..., 0] + gt[..., 2] + gt[..., 4] + gt[..., 6]) / 4
        center_y = (gt[..., 1] + gt[..., 3] + gt[..., 5] + gt[..., 7]) / 4
        center_gt = gt.new_zeros(gt.shape)

        if center_x[..., 0].sum() == 0:
            return gt_xs.new_zeros(gt_xs.shape, dtype=torch.uint8)
        beg = 0
        for level, n_p in enumerate(num_points_per):  # 每个fpn层的网格数目,[16128, 4032, 1008, 252, 66]
            end = beg + n_p
            stride = strides[level] * radius
            xmin = center_x[beg:end] - stride
            ymin = center_y[beg:end] - stride
            xmax = center_x[beg:end] + stride
            ymax = center_y[beg:end] + stride
            # limit sample region in gt
            center_gt[beg:end, :, 0] = torch.where(
                xmin > gt[beg:end, :, 0], xmin, gt[beg:end, :, 0]  # 若满足条件，则选择xmin，否则选择gt[]
            )
            center_gt[beg:end, :, 1] = torch.where(
                ymin > gt[beg:end, :, 1], ymin, gt[beg:end, :, 1]
            )
            center_gt[beg:end, :, 2] = torch.where(
                xmax > gt[beg:end, :, 2],
                gt[beg:end, :, 2], xmax
            )
            center_gt[beg:end, :, 3] = torch.where(
                ymin > gt[beg:end, :, 3], ymin, gt[beg:end, :, 3]
            )
            center_gt[beg:end, :, 4] = torch.where(
                xmax > gt[beg:end, :, 4], 
                gt[beg:end, :, 4], xmax
            )
            center_gt[beg:end, :, 5] = torch.where(
                ymax > gt[beg:end, :, 5], 
                 gt[beg:end, :, 5], ymax
            )
            center_gt[beg:end, :, 6] = torch.where(
                xmin > gt[beg:end, :, 6], xmin, gt[beg:end, :, 6]
            )
            center_gt[beg:end, :, 7] = torch.where(
                ymax > gt[beg:end, :, 7],
                gt[beg:end, :, 7], ymax
            )
            beg = end
        x1 = gt_xs[:, None] - center_gt[..., 0]  # torch.Size([21486, 3]),3表示3个GT框
        y1 = gt_ys[:, None] - center_gt[..., 1]
        x2 = gt_xs[:, None] - center_gt[..., 2]
        y2 = gt_ys[:, None] - center_gt[..., 3]
        x3 = gt_xs[:, None] - center_gt[..., 4]
        y3 = gt_ys[:, None] - center_gt[..., 5]
        x4 = gt_xs[:, None] - center_gt[..., 6]
        y4 = gt_ys[:, None] - center_gt[..., 7]
        inside_gt_bbox_mask = (x1*x3<0) * (y1*y3<0)
        # left = gt_xs[:, None] - center_gt[..., 0]
        # right = center_gt[..., 2] - gt_xs[:, None]
        # top = gt_ys[:, None] - center_gt[..., 1]
        # bottom = center_gt[..., 3] - gt_ys[:, None]
        # center_bbox = torch.stack((left, top, right, bottom), -1)
        # inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        return inside_gt_bbox_mask

    def prepare_targets(self, points, targets):
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
        labels, reg_targets = self.compute_targets_for_locations(
            points_all_level, targets, expanded_object_sizes_of_interest
        )

        for i in range(len(labels)):
            labels[i] = torch.split(labels[i], num_points_per_level, dim=0)
            reg_targets[i] = torch.split(reg_targets[i], num_points_per_level, dim=0)

        labels_level_first = []
        reg_targets_level_first = []
        for level in range(len(points)):
            labels_level_first.append(
                torch.cat([labels_per_im[level] for labels_per_im in labels], dim=0)
            )

            reg_targets_per_level = torch.cat([
                reg_targets_per_im[level]
                for reg_targets_per_im in reg_targets
            ], dim=0)

            if self.norm_reg_targets:
                reg_targets_per_level = reg_targets_per_level / self.fpn_strides[level]
            reg_targets_level_first.append(reg_targets_per_level)

        return labels_level_first, reg_targets_level_first

    def compute_targets_for_locations(self, locations, targets, object_sizes_of_interest):
        labels = []
        reg_targets = []
        xs, ys = locations[:, 0], locations[:, 1]  # torch.Size([21486,])
        for im_i in range(len(targets)):  # 图片的数量
            targets_per_im = targets[im_i]
            assert targets_per_im.mode == "xyn"
            bboxes = targets_per_im.bbox  # 3*8的GT坐标,torch.Size([3, 8])
            labels_per_im = targets_per_im.get_field("labels")  # [1]
            if len(bboxes) == 0:
                continue
            area = targets_per_im.area()  # 外接矩形面积,shape = GT Number 4

            # l = xs[:, None] - bboxes[:, 0][None]
            # t = ys[:, None] - bboxes[:, 1][None]
            # r = bboxes[:, 2][None] - xs[:, None]
            # b = bboxes[:, 3][None] - ys[:, None]

            # xs：torch.Size([21486])  bboxes：torch.Size([3, 8])
            x1 = xs[:, None] - bboxes[:, 0][None]  # x1:torch.Size([21486, 3]),表示每个location与每个gt坐标的差距
            y1 = ys[:, None] - bboxes[:, 1][None]
            x2 = xs[:, None] - bboxes[:, 2][None]
            y2 = ys[:, None] - bboxes[:, 3][None]
            x3 = xs[:, None] - bboxes[:, 4][None]
            y3 = ys[:, None] - bboxes[:, 5][None]
            x4 = xs[:, None] - bboxes[:, 6][None]
            y4 = ys[:, None] - bboxes[:, 7][None]
            reg_targets_per_im = torch.stack([x1,y1,x2,y2,x3,y3,x4,y4], dim=2)  # torch.Size([21486, 3, 8])

            # if self.center_sampling_radius > 0:
            if False:
                is_in_boxes = self.get_sample_region(
                    bboxes,
                    self.fpn_strides,
                    self.num_points_per_level,
                    xs, ys,
                    radius=self.center_sampling_radius
                )
            else:
                # no center sampling, it will use all the locations within a ground-truth box
                # is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0  # 为True、False，torch.Size([21486, GT-num, 4])
                is_in_boxes = (x1*x3<0) + (y1*y3<0)  # torch.Size([21486, 3])   经过debug，是有True在里面的 
            # print(torch.sum((x1*x3<0) + (y1*y3<0)))  # 4052
            # print(torch.sum((x1*x3<0) * (y1*y3<0)))  # 51
            
            # if True in is_in_boxes:
            #     print(True)
            # else:
            #     print(False)

            # reg_targets_per_im = abs(reg_targets_per_im)
            max_reg_targets_per_im = abs(reg_targets_per_im).max(dim=2)[0]  
            # limit the regression range for each location
            is_cared_in_the_level = \
                (max_reg_targets_per_im >= object_sizes_of_interest[:, [0]]) & \
                (max_reg_targets_per_im <= object_sizes_of_interest[:, [1]])  # 看最长边是不是在fpn这个范围内
            # 经过debug，是有True在里面的

            # if True in is_cared_in_the_level:
            #     print(True)
            # else:
            #     print(False)
            
            locations_to_gt_area = area[None].repeat(len(locations), 1)  # torch.Size([21486, 6])
            # print(is_in_boxes.shape)
            # print(is_cared_in_the_level.shape)
            # print(locations_to_gt_area.shape)
            locations_to_gt_area[is_in_boxes == 0] = INF  # 不是正样本的坐标点面积设置无穷
            locations_to_gt_area[is_cared_in_the_level == 0] = INF  # 不在fpn范围内的GT内的坐标点也设为无穷
            # print(locations_to_gt_area)
            # print(locations_to_gt_area.shape)
            # if True in locations_to_gt_area<=INF:
            #     print(True)
            # else:
            #     print(False)
            

            # if there are still more than one objects for a location,
            # we choose the one with minimal area
            locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(dim=1)  
            # 如果一个location有多个GT
            # print(torch.sum(locations_to_min_area!=INF))

            reg_targets_per_im = reg_targets_per_im[range(len(locations)), locations_to_gt_inds]  # torch.Size([21486, 8])，每个pos对应的GT
            labels_per_im = labels_per_im[locations_to_gt_inds]
            labels_per_im[locations_to_min_area == INF] = 0  # 让负样本点的label=0
            labels.append(labels_per_im)
            # print(locations_to_gt_inds.shape)
            # print(reg_targets_per_im)
            # print(reg_targets_per_im.shape)

            reg_targets.append(reg_targets_per_im)

        return labels, reg_targets

    def compute_centerness_targets(self, reg_targets):
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                      (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness)

    def __call__(self, locations, box_cls, box_regression, centerness, targets):
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
        # targets:BoxList(num_boxes=2, image_width=1333, image_height=750, mode=xyn),,
        # BoxList(num_boxes=2, image_width=1333, image_height=750, mode=xyn)] 共4个
        N = box_cls[0].size(0)  # batch size, box_cls.size(0)是fpn层数
        num_classes = box_cls[0].size(1)
        labels, reg_targets = self.prepare_targets(locations, targets)

        box_cls_flatten = []
        box_regression_flatten = []
        centerness_flatten = []
        labels_flatten = []
        reg_targets_flatten = []
        for l in range(len(labels)):
            box_cls_flatten.append(box_cls[l].permute(0, 2, 3, 1).reshape(-1, num_classes))
            box_regression_flatten.append(box_regression[l].permute(0, 2, 3, 1).reshape(-1, 8))
            labels_flatten.append(labels[l].reshape(-1))
            reg_targets_flatten.append(reg_targets[l].reshape(-1, 8))
            # centerness_flatten.append(centerness[l].reshape(-1))

        box_cls_flatten = torch.cat(box_cls_flatten, dim=0)
        box_regression_flatten = torch.cat(box_regression_flatten, dim=0)
        #centerness_flatten = torch.cat(centerness_flatten, dim=0)
        labels_flatten = torch.cat(labels_flatten, dim=0)
        reg_targets_flatten = torch.cat(reg_targets_flatten, dim=0)

        pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1)  # label中等于INF（负样本）的为0，
        # print(pos_inds)
        # print(pos_inds.shape)

        box_regression_flatten = box_regression_flatten[pos_inds]
        reg_targets_flatten = reg_targets_flatten[pos_inds]
        #centerness_flatten = centerness_flatten[pos_inds]

        # print(box_regression_flatten)
        # print(box_regression_flatten.shape)
        # print(reg_targets_flatten)
        # print(reg_targets_flatten.shape)

        num_gpus = get_num_gpus()
        # sync num_pos from all gpus
        total_num_pos = reduce_sum(pos_inds.new_tensor([pos_inds.numel()])).item()
        num_pos_avg_per_gpu = max(total_num_pos / float(num_gpus), 1.0)

        cls_loss = self.cls_loss_func(
            box_cls_flatten,
            labels_flatten.int()
        ) / num_pos_avg_per_gpu

        if pos_inds.numel() > 0:
            #centerness_targets = self.compute_centerness_targets(reg_targets_flatten)

            # average sum_centerness_targets from all gpus,
            # which is used to normalize centerness-weighed reg loss
            #sum_centerness_targets_avg_per_gpu = \
                #reduce_sum(centerness_targets.sum()).item() / float(num_gpus)
            sum_centerness_targets_avg_per_gpu = 1
            reg_loss = self.box_reg_loss_func(
                box_regression_flatten,
                reg_targets_flatten
                # centerness_targets
            ) / sum_centerness_targets_avg_per_gpu
            # centerness_loss = self.centerness_loss_func(
            #     centerness_flatten,
            #     centerness_targets
            # ) / num_pos_avg_per_gpu
        else:
            reg_loss = box_regression_flatten.sum()
            #reduce_sum(centerness_flatten.new_tensor([0.0]))
            # centerness_loss = centerness_flatten.sum()

        # return cls_loss, reg_loss, centerness_loss
        # print(cls_loss,reg_loss)
        return cls_loss, reg_loss


def make_fcos_loss_evaluator(cfg):
    loss_evaluator = FCOSLossComputation(cfg)
    return loss_evaluator
