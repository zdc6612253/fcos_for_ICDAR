# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import numpy as np
import math
from shapely import affinity
from shapely.geometry import Polygon as ShapePolygon


# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1


class BoxList(object):
    """
    This class represents a set of bounding boxes.
    The bounding boxes are represented as a Nx4 Tensor.
    In order to uniquely determine the bounding boxes with respect
    to an image, we also store the corresponding image dimensions.
    They can contain extra information that is specific to each bounding box, such as
    labels.
    """

    def __init__(self, bbox, difficult, image_size, mode="xyn"):
        device = bbox.device if isinstance(bbox, torch.Tensor) else torch.device("cpu")
        bbox = torch.as_tensor(bbox, dtype=torch.float32, device=device)
        if bbox.ndimension() != 2:  # anchor num, anchor dim
            raise ValueError(
                "bbox should have 2 dimensions, got {}".format(bbox.ndimension())
            )
        # if bbox.size(-1) != 4:
        if bbox.size(-1) != 8:
            raise ValueError(
                "last dimension of bbox should have a "
                "size of 4, got {}".format(bbox.size(-1))
            )
        if mode not in ("xyxy", "xywh", 'xyn'):
            raise ValueError("mode should be 'xyxy' or 'xywh'")

        self.bbox = bbox
        self.difficult = difficult
        self.size = image_size  # (image_width, image_height)
        self.mode = mode
        self.extra_fields = {}

    def add_field(self, field, field_data):
        self.extra_fields[field] = field_data

    def get_field(self, field):
        return self.extra_fields[field]

    def has_field(self, field):
        return field in self.extra_fields

    def fields(self):
        return list(self.extra_fields.keys())

    def _copy_extra_fields(self, bbox):
        for k, v in bbox.extra_fields.items():
            self.extra_fields[k] = v

    def convert(self, mode):
        if mode not in ("xyxy", "xywh", 'xyn'):
            raise ValueError("mode should be 'xyxy' or 'xywh'")
        if mode == self.mode:
            return self
        # we only have two modes, so don't need to check
        # self.mode
        xmin, ymin, xmax, ymax = self._split_into_xyxy()
        if mode == "xyxy":
            bbox = torch.cat((xmin, ymin, xmax, ymax), dim=-1)
            bbox = BoxList(bbox, self.difficult, self.size, mode=mode)
        elif mode == "xywh":
            TO_REMOVE = 1
            bbox = torch.cat(
                (xmin, ymin, xmax - xmin + TO_REMOVE, ymax - ymin + TO_REMOVE), dim=-1
            )
            bbox = BoxList(bbox, self.difficult, self.size, mode=mode)
        else:
            raise RuntimeError("Should not be here, should be xyn")
        bbox._copy_extra_fields(self)
        return bbox

    def _split_into_xyn(self):
        if self.mode != "xyn":
            raise RuntimeError("Should not be here, should be xyn")
        else:
            x1, y1, x2, y2, x3, y3, x4, y4 = self.bbox.split(1, dim=-1)
            return x1, y1, x2, y2, x3, y3, x4, y4

    def _split_into_xyxy(self):
        if self.mode == "xyxy":
            xmin, ymin, xmax, ymax = self.bbox.split(1, dim=-1)
            return xmin, ymin, xmax, ymax
        elif self.mode == "xywh":
            TO_REMOVE = 1
            xmin, ymin, w, h = self.bbox.split(1, dim=-1)
            return (
                xmin,
                ymin,
                xmin + (w - TO_REMOVE).clamp(min=0),
                ymin + (h - TO_REMOVE).clamp(min=0),
            )
        else:
            raise RuntimeError("Should not be here")

    def resize(self, size, *args, **kwargs):
        """
        Returns a resized copy of this bounding box

        :param size: The requested size in pixels, as a 2-tuple:
            (width, height).
        """

        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(size, self.size))
        if ratios[0] == ratios[1]:
            ratio = ratios[0]
            scaled_box = self.bbox * ratio
            bbox = BoxList(scaled_box, self.difficult, size, mode=self.mode)
            # bbox._copy_extra_fields(self)
            for k, v in self.extra_fields.items():
                if not isinstance(v, torch.Tensor):
                    v = v.resize(size, *args, **kwargs)
                bbox.add_field(k, v)
            return bbox

        ratio_width, ratio_height = ratios
        x1, y1, x2, y2, x3, y3, x4, y4 = self._split_into_xyn()
        scaled_x1 = x1 * ratio_width
        scaled_x2 = x2 * ratio_width
        scaled_x3 = x3 * ratio_width
        scaled_x4 = x4 * ratio_width
        scaled_y1 = y1 * ratio_height
        scaled_y2 = y2 * ratio_height
        scaled_y3 = y3 * ratio_height
        scaled_y4 = y4 * ratio_height
        scaled_box = torch.cat(
            (scaled_x1, scaled_y1, scaled_x2, scaled_y2,scaled_x3, scaled_y3, scaled_x4, scaled_y4), dim=-1
        )
        bbox = BoxList(scaled_box, self.difficult, size, mode="xyn")
        # bbox._copy_extra_fields(self)
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.resize(size, *args, **kwargs)
            bbox.add_field(k, v)

        return bbox.convert(self.mode)

    def transpose(self, method):
        """
        Transpose bounding box (flip or rotate in 90 degree steps)
        :param method: One of :py:attr:`PIL.Image.FLIP_LEFT_RIGHT`,
          :py:attr:`PIL.Image.FLIP_TOP_BOTTOM`, :py:attr:`PIL.Image.ROTATE_90`,
          :py:attr:`PIL.Image.ROTATE_180`, :py:attr:`PIL.Image.ROTATE_270`,
          :py:attr:`PIL.Image.TRANSPOSE` or :py:attr:`PIL.Image.TRANSVERSE`.
        """
        if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM):
            raise NotImplementedError(
                "Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented"
            )

        image_width, image_height = self.size
        # xmin, ymin, xmax, ymax = self._split_into_xyxy()
        x1, y1, x2, y2, x3, y3, x4, y4 = self._split_into_xyn()
        if method == FLIP_LEFT_RIGHT:
            TO_REMOVE = 1
            transposed_x1 = image_width - x2 - TO_REMOVE
            transposed_x2 = image_width - x1 - TO_REMOVE
            transposed_x3 = image_width - x4 - TO_REMOVE
            transposed_x4 = image_width - x3 - TO_REMOVE
            transposed_y1 = y1
            transposed_y2 = y2
            transposed_y3 = y3
            transposed_y4 = y4
        elif method == FLIP_TOP_BOTTOM:
            transposed_x1 = x1
            transposed_x2 = x2
            transposed_x3 = x3
            transposed_x4 = x4
            transposed_y1 = image_height - y2
            transposed_y2 = image_height - y1
            transposed_y3 = image_height - y4
            transposed_y4 = image_height - y3

        transposed_boxes = torch.cat(
            (transposed_x1, transposed_y1, transposed_x2, transposed_y2, transposed_x3, transposed_y3, transposed_x4, transposed_y4), dim=-1
        )
        bbox = BoxList(transposed_boxes, self.difficult, self.size, mode="xyn")
        # bbox._copy_extra_fields(self)
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.transpose(method)
            bbox.add_field(k, v)
        return bbox.convert(self.mode)
    

    def poly_rotate(self, poly, angle, r_c, start_h, start_w):
        poly = poly.numpy().reshape(-1, 2)
        poly[:, 0] += start_w
        poly[:, 1] += start_h
        polys = ShapePolygon(poly)
        r_polys = list(affinity.rotate(polys, angle, r_c).boundary.coords[:-1])
        rp = []
        for r in r_polys:
            rp += list(r)
        if np.array(rp).reshape(-1).shape[0] == 6:
            rp += rp[:2]
        #     print(poly.shape)
        #     print(np.array(rp).reshape(-1).shape)
        #     import ipdb; ipdb.set_trace()
        return np.array(rp).reshape(-1)

    def rotate(self, angle, r_c, start_h, start_w):
        boxes = []
        for poly in self.bbox:
            box = self.poly_rotate(poly, angle, r_c, start_h, start_w)
            boxes.append(box)
        boxes = torch.as_tensor(boxes).reshape(-1, 8)
        self.size = (r_c[0] * 2, r_c[1] * 2)
        bbox = BoxList(boxes, self.difficult, self.size, mode="xyn")
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                # v = v.rotate(angle, r_c, start_h, start_w)
                print(k)
                assert False
            bbox.add_field(k, v)
        return bbox.convert(self.mode)


    def crop(self, box):
        """
        Cropss a rectangular region from this bounding box. The box is a
        4-tuple defining the left, upper, right, and lower pixel
        coordinate.
        """
        x1, y1, x2, y2, x3, y3, x4, y4  = self._split_into_xyn()
        # xmin, ymin, xmax, ymax = self._split_into_xyxy()
        w, h = box[2] - box[0], box[3] - box[1]
        cropped_x1 = (x1 - box[0]).clamp(min=0, max=w)  # n * 1
        cropped_y1 = (y1 - box[1]).clamp(min=0, max=h)
        cropped_x2 = (x2 - box[0]).clamp(min=0, max=w)
        cropped_y2 = (y2 - box[1]).clamp(min=0, max=h)
        cropped_x3 = (x3 - box[0]).clamp(min=0, max=w)
        cropped_y3 = (y3 - box[1]).clamp(min=0, max=h)
        cropped_x4 = (x4 - box[0]).clamp(min=0, max=w)
        cropped_y4 = (y4 - box[1]).clamp(min=0, max=h)

        # TODO should I filter empty boxes here?
        if False:
            is_empty = (cropped_xmin == cropped_xmax) | (cropped_ymin == cropped_ymax)

        cropped_box = torch.cat(
            (cropped_x1, cropped_y1, cropped_x2, cropped_y2, cropped_x3, cropped_y3, cropped_x4, cropped_y4), dim=-1
        )
        bbox = BoxList(cropped_box, self.difficult, (w, h), mode="xyn")
        # bbox._copy_extra_fields(self)
        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                print(k)
                v = v.crop(box)
            bbox.add_field(k, v)
        return bbox.convert(self.mode)

    # def crop(self, box):
    #     """
    #     Cropss a rectangular region from this bounding box. The box is a
    #     4-tuple defining the left, upper, right, and lower pixel
    #     coordinate.
    #     """
    #     xmin, ymin, xmax, ymax = self._split_into_xyxy()
    #     w, h = box[2] - box[0], box[3] - box[1]
    #     cropped_xmin = (xmin - box[0]).clamp(min=0, max=w)
    #     cropped_ymin = (ymin - box[1]).clamp(min=0, max=h)
    #     cropped_xmax = (xmax - box[0]).clamp(min=0, max=w)
    #     cropped_ymax = (ymax - box[1]).clamp(min=0, max=h)

    #     # TODO should I filter empty boxes here?
    #     if False:
    #         is_empty = (cropped_xmin == cropped_xmax) | (cropped_ymin == cropped_ymax)

    #     cropped_box = torch.cat(
    #         (cropped_xmin, cropped_ymin, cropped_xmax, cropped_ymax), dim=-1
    #     )
    #     bbox = BoxList(cropped_box, self.difficult, (w, h), mode="xyxy")
    #     # bbox._copy_extra_fields(self)
    #     for k, v in self.extra_fields.items():
    #         if not isinstance(v, torch.Tensor):
    #             v = v.crop(box)
    #         bbox.add_field(k, v)
    #     return bbox.convert(self.mode)

    def to(self, device):
        bbox = BoxList(self.bbox.to(device), self.difficult.to(device), self.size, self.mode)
        for k, v in self.extra_fields.items():
            if hasattr(v, "to"):
                v = v.to(device)
            bbox.add_field(k, v)
        return bbox

    def __getitem__(self, item):
        bbox = BoxList(self.bbox[item], self.difficult[item], self.size, self.mode)
        for k, v in self.extra_fields.items():
            bbox.add_field(k, v[item])
        return bbox

    def __len__(self):
        return self.bbox.shape[0]

    def clip_to_image(self, remove_empty=True):
        TO_REMOVE = 1
        self.bbox[:, 0].clamp_(min=0, max=self.size[0] - TO_REMOVE)
        self.bbox[:, 1].clamp_(min=0, max=self.size[1] - TO_REMOVE)
        self.bbox[:, 2].clamp_(min=0, max=self.size[0] - TO_REMOVE)
        self.bbox[:, 3].clamp_(min=0, max=self.size[1] - TO_REMOVE)
        self.bbox[:, 4].clamp_(min=0, max=self.size[0] - TO_REMOVE)
        self.bbox[:, 5].clamp_(min=0, max=self.size[1] - TO_REMOVE)
        self.bbox[:, 6].clamp_(min=0, max=self.size[0] - TO_REMOVE)
        self.bbox[:, 7].clamp_(min=0, max=self.size[1] - TO_REMOVE)
        # if remove_empty:
        #     box = self.bbox
        #     keep = (box[:, 5] > box[:, 1]) & (box[:, 4] > box[:, 0])  # x2>x1, y4>y1
        #     return self[keep]
        return self

    def area(self):
        box = self.bbox
        if self.mode == "xyxy":
            TO_REMOVE = 1
            area = (box[:, 2] - box[:, 0] + TO_REMOVE) * (box[:, 3] - box[:, 1] + TO_REMOVE)
        elif self.mode == "xywh":
            area = box[:, 2] * box[:, 3]
        elif self.mode == "xyn":
            x_min, _ = torch.min(box[:,::2], dim=1)
            x_max, _ = torch.max(box[:,::2], dim=1)
            y_min, _ = torch.min(box[:,1::2], dim=1)
            y_max, _ = torch.max(box[:,1::2], dim=1)
            area = (x_max-x_min)*(y_max-y_min)
        else:
            raise RuntimeError("Should not be here")

        return area

    def copy_with_fields(self, fields, skip_missing=False):
        bbox = BoxList(self.bbox, self.difficult, self.size, self.mode)
        if not isinstance(fields, (list, tuple)):
            fields = [fields]
        for field in fields:
            if self.has_field(field):
                bbox.add_field(field, self.get_field(field))
            elif not skip_missing:
                raise KeyError("Field '{}' not found in {}".format(field, self))
        return bbox

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_boxes={}, ".format(len(self))
        s += "image_width={}, ".format(self.size[0])
        s += "image_height={}, ".format(self.size[1])
        s += "mode={})".format(self.mode)
        return s


if __name__ == "__main__":
    bbox = BoxList([[0, 0, 10, 10], [0, 0, 5, 5]], (10, 10))
    s_bbox = bbox.resize((5, 5))
    print(s_bbox)
    print(s_bbox.bbox)

    t_bbox = bbox.transpose(0)
    print(t_bbox)
    print(t_bbox.bbox)
