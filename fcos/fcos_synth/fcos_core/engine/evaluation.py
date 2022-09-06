import json
import numpy as np
from shapely.geometry import Polygon
import os
import time
import datetime

import torch

def save_txt_func(img_name,box_list,score_list,txt_floder,ignore_pre_idx):
    output_txt = open(os.path.join(txt_floder,'res_'+img_name.replace('.jpg','.txt')), 'w')
    box_list = np.array(box_list)
    for i in range(len(box_list)):
        if i in ignore_pre_idx:
            continue
        box = box_list[i]
        score = score_list[i]
        box = box.tolist()
        box = list(map(str, box))
        score = str(round(score.item(), 3))
        output_txt.write(','.join(box)+','+score+'\n')
    output_txt.close()

def json_deal(json_path):
    python_json_file = json.load(open(json_path))
    images_list = python_json_file['images']
    annotations_list = python_json_file['annotations']
    image_id_segmengt = {}
    image_id_difficult = {}
    # id_img_name = {}
    # for dic in images_list:
    #     id = dic['id']
    #     img_name = dic['file_name']
    #     id_img_name[id] = img_name
    for dic in annotations_list:
        image_id = dic['image_id']
        # image_name = id_img_name[image_id]
        segmentation = dic['segmentation']
        difficult = dic['difficult']
        if str(image_id) not in image_id_segmengt:
            image_id_segmengt[str(image_id)] = [segmentation[0]]
        else:
            image_id_segmengt[str(image_id)].append(segmentation[0])
        if str(image_id) not in image_id_difficult:
            image_id_difficult[str(image_id)] = [difficult]
        else:
            image_id_difficult[str(image_id)].append(difficult)
    return image_id_segmengt, image_id_difficult

def cal_evaluate(pre_box_list,gt_box_list,iou_thresh,difficult_list,ignore_iou):  # 一张图片的不同GT
    detMatched=0

    pre_box_list = np.array(pre_box_list)
    gt_box_list = np.array(gt_box_list)

    def get_union(pD, pG):
        return Polygon(pD).buffer(0.01).union(Polygon(pG).buffer(0.01)).area

    def get_intersection_over_union(pD, pG):
        pD = np.array(pD).reshape(4,2)
        pG = np.array(pG).reshape(4,2)
        return get_intersection(pD, pG) / get_union(pD, pG)

    def get_intersection(pD, pG):
        return Polygon(pD).buffer(0.01).intersection(Polygon(pG).buffer(0.01)).area
    useful_gt_idx = []
    ignore_gt_idx = []
    for i in range(len(difficult_list)):
        if difficult_list[i] == 0:
            useful_gt_idx.append(i)
        elif difficult_list[i] == 1:
            ignore_gt_idx.append(i)
        else:
            assert False, "gt difficult must in (0,1)"
    ignore_pre_idx = []
    for i in range(len(pre_box_list)):
        if len(ignore_gt_idx) > 0:
            for dontCare_idx in ignore_gt_idx:
                dontCare_gt = gt_box_list[dontCare_idx]
                # print(dontCare_gt)
                # print(pre_box_list[i])
                intersected_area_with_ignore = get_intersection(dontCare_gt.reshape(4,2), pre_box_list[i].reshape(4,2))
                pre_box_area = Polygon(pre_box_list[i].reshape(4,2)).area
                precision = 0 if pre_box_area == 0 else intersected_area_with_ignore / pre_box_area
                if (precision > ignore_iou):
                    ignore_pre_idx.append(i)
                    break
    # print(len(pre_box_list))
    # print(len(ignore_pre_idx))
    
    if len(gt_box_list) > 0 and len(pre_box_list) > 0:
        outputShape = [len(gt_box_list), len(pre_box_list)]
        iouMat = np.empty(outputShape)
        gtRectMat = np.zeros(len(gt_box_list), np.int8)
        detRectMat = np.zeros(len(pre_box_list), np.int8)
        for gtNum in range(len(gt_box_list)):
            for detNum in range(len(pre_box_list)):
                pG = np.float32(gt_box_list[gtNum])
                pD = np.float32(pre_box_list[detNum])
                iouMat[gtNum, detNum] = get_intersection_over_union(pD, pG)
        for gtNum in range(len(gt_box_list)):
            for detNum in range(len(pre_box_list)):
                if gtRectMat[gtNum] == 0 and detRectMat[detNum] == 0 and \
                    gtNum not in ignore_gt_idx and detNum not in ignore_pre_idx: 
                # 已经匹配好的GT和匹配好的Pre都不计数了
                    if iouMat[gtNum, detNum] > iou_thresh:
                        gtRectMat[gtNum] = 1
                        detRectMat[detNum] = 1
                        detMatched += 1
        return detMatched, len(gt_box_list)-len(ignore_gt_idx), len(pre_box_list)-len(ignore_pre_idx), ignore_pre_idx
    else:
        return 0, len(gt_box_list)-len(ignore_gt_idx), len(pre_box_list)-len(ignore_pre_idx), ignore_pre_idx

def evaluate(json_path, predictions, iou_thresh, dataset, ignore_iou, txt_floder, print_txt):
    detMatched_all=0
    gt_length_all = 0
    pre_length_all = 0
    image_id_segmengt, image_id_difficult = json_deal(json_path)
    start_time = time.time()
    for img_id in predictions:
        prediction = predictions[img_id]
        img_info = dataset.get_img_info(img_id)
        # print(dataset.__getitem__(img_id))
        # print(dataset.__getitem__(img_id)[1].bbox)
        image_width = img_info["width"]
        image_height = img_info["height"]
        prediction = prediction.resize((image_width, image_height))
        pre_box_list = prediction.bbox  # n,8
        pre_box_score = prediction.get_field("scores")  # n
        # print(prediction.bbox)
        # print(prediction.get_field("scores"))
        if isinstance(img_id,torch.Tensor):
            img_id = int(img_id.item())
        if str(img_id) not in image_id_segmengt:
            gt_box_list = []
            difficult_list = []
        else:
            gt_box_list = image_id_segmengt[str(img_id)]
            difficult_list = image_id_difficult[str(img_id)]
        detMatched, gt_length, pre_length, ignore_pre_idx = cal_evaluate(pre_box_list,\
            gt_box_list,iou_thresh,difficult_list,ignore_iou)
        assert (detMatched>=0 and gt_length>=0 and pre_length>=0)
        detMatched_all += detMatched
        gt_length_all += gt_length
        pre_length_all += pre_length
        print(img_info["file_name"])
        print("detMatched:"+str(detMatched),"gt_length:"+str(gt_length),"pre_length:"+str(pre_length))
        if print_txt:  # 打印txt坐标
            save_txt_func(img_info["file_name"],pre_box_list,pre_box_score,txt_floder,ignore_pre_idx)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    if gt_length_all == 0:
        recall = float(1)
        precision = float(0) if pre_length_all > 0 else float(1)
    else:
        recall = float(detMatched_all) / gt_length_all
        precision = float(0) if pre_length_all == 0 else float(detMatched_all) / pre_length_all
    # print(detMatched_all,pre_length_all,gt_length_all)
    f1 = 0 if (precision + recall) == 0 else (2.0 *  precision * recall) / (precision + recall)

    print("Total inference time: {} (fps: {} img num / s)".format(
                total_time_str,  len(predictions)/total_time
            )
        )
    return f1, recall, precision