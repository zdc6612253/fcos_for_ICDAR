# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Miscellaneous utility functions
"""

import torch
import numpy as np
from shapely.geometry import Polygon, MultiPoint

def cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def bbox_overlap(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-6, scale=64):
    """From MMdetection
        1) is_aligned is True
            bboxes1: N × 8
            bboxes2: N × 8
            ious: N x 1
    """
    assert (bboxes1.size(-1) == 8 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 8 or bboxes2.size(0) == 0)

    if is_aligned:
        score = []
        for bbox1, bbox2 in zip(bboxes1, bboxes2):
            bbox1 = (bbox1.cpu().numpy().reshape(4, 2) * scale).astype(np.int32)
            bbox2 = (bbox2.cpu().numpy().reshape(4, 2) * scale).astype(np.int32)
            poly1 = Polygon(bbox1).convex_hull
            poly2 = Polygon(bbox2).convex_hull
            if not poly1.intersects(poly2):
                iou = 0
            else:
                inter_area = poly1.intersection(poly2).area
                union_area = max(MultiPoint(np.concatenate((bbox1, bbox2))).convex_hull.area, eps)
                iou = float(inter_area) / union_area
            score.append(iou)
        score = torch.tensor(score).to(bboxes1.device).to(torch.float32)
    else:
        raise NotImplementedError
    return score