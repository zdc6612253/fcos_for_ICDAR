# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torchvision
import numpy as np
import cv2
from PIL import Image
import os

from fcos_core.structures.bounding_box import BoxList
from fcos_core.structures.segmentation_mask import SegmentationMask
from fcos_core.structures.keypoint import PersonKeypoints


min_keypoints_per_image = 10


def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= min_keypoints_per_image:
        return True
    return False


class COCODataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
        self, ann_file, root, remove_images_without_annotations, transforms=None
    ):
        super(COCODataset, self).__init__(root, ann_file)
        # sort indices for reproducible results
        self.ids = sorted(self.ids)

        # filter images without detection annotations
        if remove_images_without_annotations:
            ids = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.coco.loadAnns(ann_ids)
                if has_valid_annotation(anno):
                    ids.append(img_id)
            self.ids = ids

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self._transforms = transforms

    def get_score_map(self, img_h, img_w, polys):
        score_map  = np.zeros((int(img_h), int(img_w), 1), np.float32)
        # polys_list = []
        for i in range(len(polys)):
            poly = polys[i]
            poly = np.around(poly.reshape(4, 2)).astype(np.int32)
            # polys_list.append(poly)
            cv2.fillPoly(score_map,poly.reshape(1,4,2),i+1)
        # print(len(polys) in score_map)
        return score_map
    
    def test_coor(self, img, target):
        image = img.transpose(0,1).transpose(2,1)
        image = image.numpy()
        # TODO qiuyang:
        folder_path = '/data/projects/FCOS/vis'
        id_name_file = '/data/projects/FCOS/datas/ICDAR_2015/test_train_json_id.txt'
        # folder_path = '/data/ocr/model_for_ICDAR/fcos/fcos_rotate/fcos/save_pic'
        # id_name_file = '/data/ocr/dataset/ICDAR_2015/test_train_json_id.txt'
        id_name_dic = {}
        for line in open(id_name_file):
            id, name = line.strip('\n').split('\t')
            id_name_dic[id] = name
        # image = Image.fromarray(np.uint8(image))
        # image.save(os.path.join(folder_path, name))
        image_id = target.get_field("idx")
        name = id_name_dic[str(image_id.item())]
        txt_name = name.replace('.jpg','.txt')
        cv2.imwrite(os.path.join(folder_path, name), image)
        txt_file = open(os.path.join(folder_path,txt_name),'w')
        bboxs = target.bbox.numpy().tolist()
        for bbox in bboxs:
            bbox = list(map(float,bbox))
            bbox = list(map(str,bbox))
            txt_file.write(','.join(bbox)+'\n')
        txt_file.close()


    def __getitem__(self, idx):
        img, anno = super(COCODataset, self).__getitem__(idx)

        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        # boxes = [obj["bbox"] for obj in anno]
        boxes = [obj["segmentation"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 8)  # 由4改成8

        difficult = [obj["difficult"] for obj in anno]
        difficult = torch.as_tensor(difficult)
        # print(difficult)
        target = BoxList(boxes, difficult, img.size, mode="xyn").convert("xyn")  # 转化为x1.y1.x2.y2.x3.y3.x4.y4

        classes = [obj["category_id"] for obj in anno]  # [1] 
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)
        idx = torch.tensor(idx)
        target.add_field("idx", idx)

        #masks = [obj["segmentation"] for obj in anno]
        #masks = SegmentationMask(masks, img.size, mode='poly')
        #target.add_field("masks", masks)

        if anno and "keypoints" in anno[0]:  # False
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = PersonKeypoints(keypoints, img.size)
            target.add_field("keypoints", keypoints)

        target = target.clip_to_image(remove_empty=True)  # 多卡训练失败问题
        # print(img.size)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        new_img_h, new_img_w = img.shape[-2:]
        score_map = self.get_score_map(new_img_h, new_img_w, np.array(target.bbox))
        score_map = torch.Tensor(score_map).permute(2,0,1)  # shape: 1,height,weight
        # self.test_coor(img, target)  # 验证坐标是否正确

        return img, target, score_map

    def get_img_info(self, index):
        if isinstance(index,torch.Tensor):
            index = int(index.item())
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data
