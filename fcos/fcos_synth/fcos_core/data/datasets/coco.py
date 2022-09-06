# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import imp
import torch
import torchvision
import numpy as np
import cv2
from PIL import Image
import os
import torch.utils.data as data
import tqdm
import time
import json
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import numpy as np
# import copy
# import itertools
# # from . import mask as maskUtils
# from collections import defaultdict
# import sys
# PYTHON_VERSION = sys.version_info[0]
# if PYTHON_VERSION == 2:
#     from urllib import urlretrieve
# elif PYTHON_VERSION == 3:
#     from urllib.request import urlretrieve

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


class COCODataset(data.Dataset):
    def __init__(
        self, ann_file, root, remove_images_without_annotations, transforms=None
    ):
        # super(COCODataset, self).__init__(root, ann_file)
        from pycocotools.coco import COCO
        self.root = root
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())
        self._transforms = transforms
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self.dic = self.get_image_folder_map(self.root)

        self.json_category_id_to_contiguous_id = {  # 将类别转化为1,2,3,,,其中背景为0
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }

    def get_image_folder_map(self, root):
        dic = {}
        image_folder_name_list = os.listdir(root)
        for image_folder_name in tqdm.tqdm(image_folder_name_list):
            image_folder_path = os.path.join(root,image_folder_name)
            image_path_list = os.listdir(image_folder_path)
            for image_name in image_path_list:
                dic[image_name] = image_folder_name
        return dic

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
        # img, anno = super(COCODataset, self).__getitem__(idx)
        # print(self.root)
        # print(self.ids)
        coco = self.coco
        # print(len(self.ids))
        img_id = self.ids[idx]
        # print(img_id)
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target_ = coco.loadAnns(ann_ids)
        # print(target_)
        image_name = coco.loadImgs(img_id)[0]['file_name']
        # print(image_name)
        image_folder = self.dic[image_name]
        image_path = os.path.join(self.root,image_folder,image_name)
        # print(image_path)
        # image_path = os.path.join(self.root,image_name)
        img = Image.open(os.path.join(self.root, image_path)).convert('RGB')
        # print(img.size)
        
        anno = [obj for obj in target_ if obj["iscrowd"] == 0]
        # print(anno)

        # boxes = [obj["bbox"] for obj in anno]
        boxes = [obj["segmentation"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 8)  # 由4改成8
        # print(boxes)

        difficult = [obj["difficult"] for obj in anno]
        difficult = torch.as_tensor(difficult)
        # print(difficult)
        # print(difficult)
        target = BoxList(boxes, difficult, img.size, mode="xyn").convert("xyn")  # 转化为x1.y1.x2.y2.x3.y3.x4.y4

        classes = [obj["category_id"] for obj in anno]  # [1] 
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        # print(classes)
        target.add_field("labels", classes)
        idx = torch.tensor(idx)
        # print(idx)
        target.add_field("idx", idx)

        #masks = [obj["segmentation"] for obj in anno]
        #masks = SegmentationMask(masks, img.size, mode='poly')
        #target.add_field("masks", masks)

        if anno and "keypoints" in anno[0]:  # False
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = PersonKeypoints(keypoints, img.size)
            target.add_field("keypoints", keypoints)

        target = target.clip_to_image(remove_empty=True)  # 多卡训练失败问题
        # print(target.bbox)
        # print(img.size)
        if self._transforms is not None:
            try:
                img, target = self._transforms(img, target)
            except:
                img, target = img, target
        new_img_h, new_img_w = img.shape[-2:]
        # print(new_img_h, new_img_w)
        score_map = self.get_score_map(new_img_h, new_img_w, np.array(target.bbox))
        score_map = torch.Tensor(score_map).permute(2,0,1)  # shape: 1,height,weight
        # self.test_coor(img, target)  # 验证坐标是否正确

        return img, target, score_map
    
    def __len__(self):
        return len(self.ids)
    
    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self._transforms.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        # tmp = '    Target Transforms (if any): '
        # fmt_str += '{0}{1}'.format(tmp, self._transforms.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def get_img_info(self, index):
        if isinstance(index,torch.Tensor):
            index = int(index.item())
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data

def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')
    
# class COCO:
#     def __init__(self, annotation_file=None):
#         """
#         Constructor of Microsoft COCO helper class for reading and visualizing annotations.
#         :param annotation_file (str): location of annotation file
#         :param image_folder (str): location to the folder that hosts images.
#         :return:
#         """
#         # load dataset
#         self.dataset,self.anns,self.cats,self.imgs = dict(),dict(),dict(),dict()
#         self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
#         if not annotation_file == None:
#             print('loading annotations into memory...')
#             tic = time.time()
#             # print('AAAA')
#             with open(annotation_file, 'r') as annotation_file_:
#                 dataset = json.load(annotation_file_)
#             # dataset = json.load(open(annotation_file, 'r'))
#             # print('BBBB')
#             # print(dataset is not None)
#             assert type(dataset)==dict, 'annotation file format {} not supported'.format(type(dataset))
#             # print('Done (t={:0.2f}s)'.format(time.time()- tic))
#             self.dataset = dataset
#             self.createIndex()

#     def createIndex(self):
#         # create index
#         print('creating index...')
#         anns, cats, imgs = {}, {}, {}
#         imgToAnns,catToImgs = defaultdict(list),defaultdict(list)
#         if 'annotations' in self.dataset:
#             for ann in self.dataset['annotations']:
#                 imgToAnns[ann['image_id']].append(ann)
#                 anns[ann['id']] = ann

#         if 'images' in self.dataset:
#             for img in self.dataset['images']:
#                 imgs[img['id']] = img

#         if 'categories' in self.dataset:
#             for cat in self.dataset['categories']:
#                 cats[cat['id']] = cat

#         if 'annotations' in self.dataset and 'categories' in self.dataset:
#             for ann in self.dataset['annotations']:
#                 catToImgs[ann['category_id']].append(ann['image_id'])

#         print('index created!')

#         # create class members
#         self.anns = anns
#         self.imgToAnns = imgToAnns
#         self.catToImgs = catToImgs
#         self.imgs = imgs
#         self.cats = cats

#     def info(self):
#         """
#         Print information about the annotation file.
#         :return:
#         """
#         for key, value in self.dataset['info'].items():
#             print('{}: {}'.format(key, value))

#     def getAnnIds(self, imgIds=[], catIds=[], areaRng=[], iscrowd=None):
#         """
#         Get ann ids that satisfy given filter conditions. default skips that filter
#         :param imgIds  (int array)     : get anns for given imgs
#                catIds  (int array)     : get anns for given cats
#                areaRng (float array)   : get anns for given area range (e.g. [0 inf])
#                iscrowd (boolean)       : get anns for given crowd label (False or True)
#         :return: ids (int array)       : integer array of ann ids
#         """
#         imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
#         catIds = catIds if _isArrayLike(catIds) else [catIds]

#         if len(imgIds) == len(catIds) == len(areaRng) == 0:
#             anns = self.dataset['annotations']
#         else:
#             if not len(imgIds) == 0:
#                 lists = [self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns]
#                 anns = list(itertools.chain.from_iterable(lists))
#             else:
#                 anns = self.dataset['annotations']
#             anns = anns if len(catIds)  == 0 else [ann for ann in anns if ann['category_id'] in catIds]
#             anns = anns if len(areaRng) == 0 else [ann for ann in anns if ann['area'] > areaRng[0] and ann['area'] < areaRng[1]]
#         if not iscrowd == None:
#             ids = [ann['id'] for ann in anns if ann['iscrowd'] == iscrowd]
#         else:
#             ids = [ann['id'] for ann in anns]
#         return ids

#     def getCatIds(self, catNms=[], supNms=[], catIds=[]):
#         """
#         filtering parameters. default skips that filter.
#         :param catNms (str array)  : get cats for given cat names
#         :param supNms (str array)  : get cats for given supercategory names
#         :param catIds (int array)  : get cats for given cat ids
#         :return: ids (int array)   : integer array of cat ids
#         """
#         catNms = catNms if _isArrayLike(catNms) else [catNms]
#         supNms = supNms if _isArrayLike(supNms) else [supNms]
#         catIds = catIds if _isArrayLike(catIds) else [catIds]

#         if len(catNms) == len(supNms) == len(catIds) == 0:
#             cats = self.dataset['categories']
#         else:
#             cats = self.dataset['categories']
#             cats = cats if len(catNms) == 0 else [cat for cat in cats if cat['name']          in catNms]
#             cats = cats if len(supNms) == 0 else [cat for cat in cats if cat['supercategory'] in supNms]
#             cats = cats if len(catIds) == 0 else [cat for cat in cats if cat['id']            in catIds]
#         ids = [cat['id'] for cat in cats]
#         return ids

#     def getImgIds(self, imgIds=[], catIds=[]):
#         '''
#         Get img ids that satisfy given filter conditions.
#         :param imgIds (int array) : get imgs for given ids
#         :param catIds (int array) : get imgs with all given cats
#         :return: ids (int array)  : integer array of img ids
#         '''
#         imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
#         catIds = catIds if _isArrayLike(catIds) else [catIds]

#         if len(imgIds) == len(catIds) == 0:
#             ids = self.imgs.keys()
#         else:
#             ids = set(imgIds)
#             for i, catId in enumerate(catIds):
#                 if i == 0 and len(ids) == 0:
#                     ids = set(self.catToImgs[catId])
#                 else:
#                     ids &= set(self.catToImgs[catId])
#         return list(ids)

#     def loadAnns(self, ids=[]):
#         """
#         Load anns with the specified ids.
#         :param ids (int array)       : integer ids specifying anns
#         :return: anns (object array) : loaded ann objects
#         """
#         if _isArrayLike(ids):
#             return [self.anns[id] for id in ids]
#         elif type(ids) == int:
#             return [self.anns[ids]]

#     def loadCats(self, ids=[]):
#         """
#         Load cats with the specified ids.
#         :param ids (int array)       : integer ids specifying cats
#         :return: cats (object array) : loaded cat objects
#         """
#         if _isArrayLike(ids):
#             return [self.cats[id] for id in ids]
#         elif type(ids) == int:
#             return [self.cats[ids]]

#     def loadImgs(self, ids=[]):
#         """
#         Load anns with the specified ids.
#         :param ids (int array)       : integer ids specifying img
#         :return: imgs (object array) : loaded img objects
#         """
#         if _isArrayLike(ids):
#             return [self.imgs[id] for id in ids]
#         elif type(ids) == int:
#             return [self.imgs[ids]]

#     # def showAnns(self, anns, draw_bbox=False):
#     #     """
#     #     Display the specified annotations.
#     #     :param anns (array of object): annotations to display
#     #     :return: None
#     #     """
#     #     if len(anns) == 0:
#     #         return 0
#     #     if 'segmentation' in anns[0] or 'keypoints' in anns[0]:
#     #         datasetType = 'instances'
#     #     elif 'caption' in anns[0]:
#     #         datasetType = 'captions'
#     #     else:
#     #         raise Exception('datasetType not supported')
#     #     if datasetType == 'instances':
#     #         ax = plt.gca()
#     #         ax.set_autoscale_on(False)
#     #         polygons = []
#     #         color = []
#     #         for ann in anns:
#     #             c = (np.random.random((1, 3))*0.6+0.4).tolist()[0]
#     #             if 'segmentation' in ann:
#     #                 if type(ann['segmentation']) == list:
#     #                     # polygon
#     #                     for seg in ann['segmentation']:
#     #                         poly = np.array(seg).reshape((int(len(seg)/2), 2))
#     #                         polygons.append(Polygon(poly))
#     #                         color.append(c)
#     #                 else:
#     #                     # mask
#     #                     t = self.imgs[ann['image_id']]
#     #                     if type(ann['segmentation']['counts']) == list:
#     #                         rle = maskUtils.frPyObjects([ann['segmentation']], t['height'], t['width'])
#     #                     else:
#     #                         rle = [ann['segmentation']]
#     #                     m = maskUtils.decode(rle)
#     #                     img = np.ones( (m.shape[0], m.shape[1], 3) )
#     #                     if ann['iscrowd'] == 1:
#     #                         color_mask = np.array([2.0,166.0,101.0])/255
#     #                     if ann['iscrowd'] == 0:
#     #                         color_mask = np.random.random((1, 3)).tolist()[0]
#     #                     for i in range(3):
#     #                         img[:,:,i] = color_mask[i]
#     #                     ax.imshow(np.dstack( (img, m*0.5) ))
#     #             if 'keypoints' in ann and type(ann['keypoints']) == list:
#     #                 # turn skeleton into zero-based index
#     #                 sks = np.array(self.loadCats(ann['category_id'])[0]['skeleton'])-1
#     #                 kp = np.array(ann['keypoints'])
#     #                 x = kp[0::3]
#     #                 y = kp[1::3]
#     #                 v = kp[2::3]
#     #                 for sk in sks:
#     #                     if np.all(v[sk]>0):
#     #                         plt.plot(x[sk],y[sk], linewidth=3, color=c)
#     #                 plt.plot(x[v>0], y[v>0],'o',markersize=8, markerfacecolor=c, markeredgecolor='k',markeredgewidth=2)
#     #                 plt.plot(x[v>1], y[v>1],'o',markersize=8, markerfacecolor=c, markeredgecolor=c, markeredgewidth=2)

#     #             if draw_bbox:
#     #                 [bbox_x, bbox_y, bbox_w, bbox_h] = ann['bbox']
#     #                 poly = [[bbox_x, bbox_y], [bbox_x, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y]]
#     #                 np_poly = np.array(poly).reshape((4,2))
#     #                 polygons.append(Polygon(np_poly))
#     #                 color.append(c)

#     #         p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.4)
#     #         ax.add_collection(p)
#     #         p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=2)
#     #         ax.add_collection(p)
#     #     elif datasetType == 'captions':
#     #         for ann in anns:
#     #             print(ann['caption'])

#     # def loadRes(self, resFile):
#     #     """
#     #     Load result file and return a result api object.
#     #     :param   resFile (str)     : file name of result file
#     #     :return: res (obj)         : result api object
#     #     """
#     #     res = COCO()
#     #     res.dataset['images'] = [img for img in self.dataset['images']]

#     #     print('Loading and preparing results...')
#     #     tic = time.time()
#     #     if type(resFile) == str or (PYTHON_VERSION == 2 and type(resFile) == unicode):
#     #         anns = json.load(open(resFile))
#     #     elif type(resFile) == np.ndarray:
#     #         anns = self.loadNumpyAnnotations(resFile)
#     #     else:
#     #         anns = resFile
#     #     assert type(anns) == list, 'results in not an array of objects'
#     #     annsImgIds = [ann['image_id'] for ann in anns]
#     #     assert set(annsImgIds) == (set(annsImgIds) & set(self.getImgIds())), \
#     #            'Results do not correspond to current coco set'
#     #     if 'caption' in anns[0]:
#     #         imgIds = set([img['id'] for img in res.dataset['images']]) & set([ann['image_id'] for ann in anns])
#     #         res.dataset['images'] = [img for img in res.dataset['images'] if img['id'] in imgIds]
#     #         for id, ann in enumerate(anns):
#     #             ann['id'] = id+1
#     #     elif 'bbox' in anns[0] and not anns[0]['bbox'] == []:
#     #         res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
#     #         for id, ann in enumerate(anns):
#     #             bb = ann['bbox']
#     #             x1, x2, y1, y2 = [bb[0], bb[0]+bb[2], bb[1], bb[1]+bb[3]]
#     #             if not 'segmentation' in ann:
#     #                 ann['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
#     #             ann['area'] = bb[2]*bb[3]
#     #             ann['id'] = id+1
#     #             ann['iscrowd'] = 0
#     #     elif 'segmentation' in anns[0]:
#     #         res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
#     #         for id, ann in enumerate(anns):
#     #             # now only support compressed RLE format as segmentation results
#     #             ann['area'] = maskUtils.area(ann['segmentation'])
#     #             if not 'bbox' in ann:
#     #                 ann['bbox'] = maskUtils.toBbox(ann['segmentation'])
#     #             ann['id'] = id+1
#     #             ann['iscrowd'] = 0
#     #     elif 'keypoints' in anns[0]:
#     #         res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
#     #         for id, ann in enumerate(anns):
#     #             s = ann['keypoints']
#     #             x = s[0::3]
#     #             y = s[1::3]
#     #             x0,x1,y0,y1 = np.min(x), np.max(x), np.min(y), np.max(y)
#     #             ann['area'] = (x1-x0)*(y1-y0)
#     #             ann['id'] = id + 1
#     #             ann['bbox'] = [x0,y0,x1-x0,y1-y0]
#     #     print('DONE (t={:0.2f}s)'.format(time.time()- tic))

#     #     res.dataset['annotations'] = anns
#     #     res.createIndex()
#     #     return res

#     # def download(self, tarDir = None, imgIds = [] ):
#     #     '''
#     #     Download COCO images from mscoco.org server.
#     #     :param tarDir (str): COCO results directory name
#     #            imgIds (list): images to be downloaded
#     #     :return:
#     #     '''
#     #     if tarDir is None:
#     #         print('Please specify target directory')
#     #         return -1
#     #     if len(imgIds) == 0:
#     #         imgs = self.imgs.values()
#     #     else:
#     #         imgs = self.loadImgs(imgIds)
#     #     N = len(imgs)
#     #     if not os.path.exists(tarDir):
#     #         os.makedirs(tarDir)
#     #     for i, img in enumerate(imgs):
#     #         tic = time.time()
#     #         fname = os.path.join(tarDir, img['file_name'])
#     #         if not os.path.exists(fname):
#     #             urlretrieve(img['coco_url'], fname)
#     #         print('downloaded {}/{} images (t={:0.1f}s)'.format(i, N, time.time()- tic))

#     # def loadNumpyAnnotations(self, data):
#     #     """
#     #     Convert result data from a numpy array [Nx7] where each row contains {imageID,x1,y1,w,h,score,class}
#     #     :param  data (numpy.ndarray)
#     #     :return: annotations (python nested list)
#     #     """
#     #     print('Converting ndarray to lists...')
#     #     assert(type(data) == np.ndarray)
#     #     print(data.shape)
#     #     assert(data.shape[1] == 7)
#     #     N = data.shape[0]
#     #     ann = []
#     #     for i in range(N):
#     #         if i % 1000000 == 0:
#     #             print('{}/{}'.format(i,N))
#     #         ann += [{
#     #             'image_id'  : int(data[i, 0]),
#     #             'bbox'  : [ data[i, 1], data[i, 2], data[i, 3], data[i, 4] ],
#     #             'score' : data[i, 5],
#     #             'category_id': int(data[i, 6]),
#     #             }]
#     #     return ann

#     # def annToRLE(self, ann):
#     #     """
#     #     Convert annotation which can be polygons, uncompressed RLE to RLE.
#     #     :return: binary mask (numpy 2D array)
#     #     """
#     #     t = self.imgs[ann['image_id']]
#     #     h, w = t['height'], t['width']
#     #     segm = ann['segmentation']
#     #     if type(segm) == list:
#     #         # polygon -- a single object might consist of multiple parts
#     #         # we merge all parts into one mask rle code
#     #         rles = maskUtils.frPyObjects(segm, h, w)
#     #         rle = maskUtils.merge(rles)
#     #     elif type(segm['counts']) == list:
#     #         # uncompressed RLE
#     #         rle = maskUtils.frPyObjects(segm, h, w)
#     #     else:
#     #         # rle
#     #         rle = ann['segmentation']
#     #     return rle

#     # def annToMask(self, ann):
#     #     """
#     #     Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
#     #     :return: binary mask (numpy 2D array)
#     #     """
#     #     rle = self.annToRLE(ann)
#     #     m = maskUtils.decode(rle)
#     #     return m
