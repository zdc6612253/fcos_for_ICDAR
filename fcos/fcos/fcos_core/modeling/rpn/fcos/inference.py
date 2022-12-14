import torch

from ..inference import RPNPostProcessor
from ..utils import permute_and_flatten

from fcos_core.modeling.box_coder import BoxCoder
from fcos_core.modeling.utils import cat
from fcos_core.structures.bounding_box import BoxList
from fcos_core.structures.boxlist_ops import cat_boxlist
from fcos_core.structures.boxlist_ops import boxlist_ml_nms
from fcos_core.structures.boxlist_ops import boxlist_nms
from fcos_core.structures.boxlist_ops import remove_small_boxes
from .nms_new import py_cpu_nms,py_cpu_nms_polygon


class FCOSPostProcessor(torch.nn.Module):
    """
    Performs post-processing on the outputs of the RetinaNet boxes.
    This is only used in the testing.
    """
    def __init__(
        self,
        pre_nms_thresh,
        pre_nms_top_n,
        nms_thresh,
        fpn_post_nms_top_n,
        min_size,
        num_classes,
        bbox_aug_enabled=False
    ):
        """
        Arguments:
            pre_nms_thresh (float)
            pre_nms_top_n (int)
            nms_thresh (float)
            fpn_post_nms_top_n (int)
            min_size (int)
            num_classes (int)
            box_coder (BoxCoder)
        """
        super(FCOSPostProcessor, self).__init__()
        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.nms_thresh = nms_thresh
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.min_size = min_size
        self.num_classes = num_classes
        self.bbox_aug_enabled = bbox_aug_enabled  # False

    def forward_for_single_feature_map(
            self, locations, box_cls,
            box_regression,
            image_sizes):
        """
        Arguments:
            anchors: list[BoxList]
            box_cls: tensor of size N, A * C, H, W
            box_regression: tensor of size N, A * 4, H, W
        """
        N, C, H, W = box_cls.shape  # 8 1 96 168  

        # put in the same format as locations
        box_cls = box_cls.view(N, C, H, W).permute(0, 2, 3, 1)  # N,H,W,C
        box_cls = box_cls.reshape(N, -1, C).sigmoid()  # N,H*W,C
        box_regression = box_regression.view(N, 8, H, W).permute(0, 2, 3, 1)
        box_regression = box_regression.reshape(N, -1, 8)  #  N,H*W,8
        # centerness = centerness.view(N, 1, H, W).permute(0, 2, 3, 1)
        # centerness = centerness.reshape(N, -1).sigmoid()

        candidate_inds = box_cls > self.pre_nms_thresh  # N,H*W,C
        pre_nms_top_n = candidate_inds.view(N, -1).sum(1)  # 8,1 ????????????N?????????score?????????
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)  # N,H*W*C

        # multiply the classification scores with centerness scores
        # box_cls = box_cls * centerness[:, :, None]

        results = []
        for i in range(N):  # ???????????????
            per_box_cls = box_cls[i]  # w*h,c 66,1
            per_candidate_inds = candidate_inds[i]  # w*h*c  66,1 True,False
            per_box_cls = per_box_cls[per_candidate_inds]  # shape:66 ????????????????????????

            per_candidate_nonzeros = per_candidate_inds.nonzero()  # True?????????  num,2  [index,0],,,
            # print(per_candidate_nonzeros.shape)
            # print(per_candidate_nonzeros)
            per_box_loc = per_candidate_nonzeros[:, 0]  # ?????? Num
            per_class = per_candidate_nonzeros[:, 1] + 1  # ?????? Num

            per_box_regression = box_regression[i]
            per_box_regression = per_box_regression[per_box_loc]
            per_locations = locations[per_box_loc]

            per_pre_nms_top_n = pre_nms_top_n[i]

            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                per_box_cls, top_k_indices = \
                    per_box_cls.topk(per_pre_nms_top_n, sorted=False)
                per_class = per_class[top_k_indices]
                per_box_regression = per_box_regression[top_k_indices]
                per_locations = per_locations[top_k_indices]

            # detections = torch.stack([
            #     per_locations[:, 0] - per_box_regression[:, 0],
            #     per_locations[:, 1] - per_box_regression[:, 1],
            #     per_locations[:, 0] + per_box_regression[:, 2],
            #     per_locations[:, 1] - per_box_regression[:, 3],
            #     per_locations[:, 0] + per_box_regression[:, 4],
            #     per_locations[:, 1] + per_box_regression[:, 5],
            #     per_locations[:, 0] - per_box_regression[:, 6],
            #     per_locations[:, 1] + per_box_regression[:, 7],
            # ], dim=1)
            detections = torch.stack([
                per_locations[:, 0] - per_box_regression[:, 0],
                per_locations[:, 1] - per_box_regression[:, 1],
                per_locations[:, 0] - per_box_regression[:, 2],
                per_locations[:, 1] - per_box_regression[:, 3],
                per_locations[:, 0] - per_box_regression[:, 4],
                per_locations[:, 1] - per_box_regression[:, 5],
                per_locations[:, 0] - per_box_regression[:, 6],
                per_locations[:, 1] - per_box_regression[:, 7],
            ], dim=1)

            h, w = image_sizes[i]
            boxlist = BoxList(bbox=detections, difficult=torch.zeros(len(detections)), image_size=(int(w), int(h)), mode="xyn")
            # print(boxlist.difficult)
            boxlist.add_field("labels", per_class)
            boxlist.add_field("scores", torch.sqrt(per_box_cls))
            # print(torch.sqrt(per_box_cls))
            boxlist = boxlist.clip_to_image(remove_empty=False)
            if len(boxlist) != 0:
                boxlist = remove_small_boxes(boxlist, self.min_size)
            # print(len(boxlist.bbox))
            # print(len(boxlist.difficult))
            results.append(boxlist)

        return results

    def forward(self, locations, box_cls, box_regression, centerness, image_sizes):
        """
        Arguments:
            anchors: list[list[BoxList]]
            box_cls: list[tensor]
            box_regression: list[tensor]
            image_sizes: list[(h, w)]
        Returns:
            boxlists (list[BoxList]): the post-processed anchors, after
                applying box decoding and NMS
        """
        sampled_boxes = []
        # for _, (l, o, b, c) in enumerate(zip(locations, box_cls, box_regression, centerness)):
        for _, (l, o, b) in enumerate(zip(locations, box_cls, box_regression)):
            sampled_boxes.append(
                self.forward_for_single_feature_map(
                    l, o, b, image_sizes
                )
            )
        boxlists = list(zip(*sampled_boxes))
        boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]
        
        if not self.bbox_aug_enabled:  # not false: ???????????????
            boxlists = self.select_over_all_levels(boxlists)

        return boxlists

    # TODO very similar to filter_results from PostProcessor
    # but filter_results is per image
    # TODO Yang: solve this issue in the future. No good solution
    # right now.
    def select_over_all_levels(self, boxlists):
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            # multiclass nms
            # print(boxlists[i].difficult)
            # result = boxlist_ml_nms(boxlists[i], self.nms_thresh)
            # print(len(boxlists[i].bbox))
            # print(len(boxlists[i].difficult))
            # print('----')
            result = py_cpu_nms_polygon(boxlists[i], self.nms_thresh)
            # result = py_cpu_nms(boxlists[i], self.nms_thresh)
            # print(result)
            number_of_detections = len(result)

            # Limit to max_per_image detections **over all classes**
            if number_of_detections > self.fpn_post_nms_top_n > 0:
                cls_scores = result.get_field("scores")
                image_thresh, _ = torch.kthvalue(
                    cls_scores.cpu(),
                    number_of_detections - self.fpn_post_nms_top_n + 1
                )
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]
            results.append(result)
        return results


def make_fcos_postprocessor(config):
    pre_nms_thresh = config.MODEL.FCOS.INFERENCE_TH
    pre_nms_top_n = config.MODEL.FCOS.PRE_NMS_TOP_N
    nms_thresh = config.MODEL.FCOS.NMS_TH
    fpn_post_nms_top_n = config.TEST.DETECTIONS_PER_IMG
    bbox_aug_enabled = config.TEST.BBOX_AUG.ENABLED

    box_selector = FCOSPostProcessor(
        pre_nms_thresh=pre_nms_thresh,          # box_cls >  pre_nms_thresh
        pre_nms_top_n=pre_nms_top_n,
        nms_thresh=nms_thresh,                  # iou > nms_thresh
        fpn_post_nms_top_n=fpn_post_nms_top_n,
        min_size=0,
        num_classes=config.MODEL.FCOS.NUM_CLASSES,
        bbox_aug_enabled=bbox_aug_enabled
    )

    return box_selector
