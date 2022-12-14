# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import time
import os

import torch
from tqdm import tqdm

from fcos_core.config import cfg
# from fcos_core.data.datasets.evaluation import evaluate
from ..utils.comm import is_main_process, get_world_size
from ..utils.comm import all_gather
from ..utils.comm import synchronize
from ..utils.timer import Timer, get_time_str
from .bbox_aug import im_detect_bbox_aug
from .save_txt import save_txt_func
from .evaluation import evaluate


def compute_on_dataset(model, data_loader, device, timer=None):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    for _, batch in enumerate(tqdm(data_loader)):
        images, targets, score_maps = batch
        with torch.no_grad():
            if timer:
                timer.tic()
            if cfg.TEST.BBOX_AUG.ENABLED:  # False
                output = im_detect_bbox_aug(model, images, device)
            else:
                output = model(images.to(device),targets=targets,score_maps=score_maps)
            if timer:
                torch.cuda.synchronize()
                timer.toc()
            output = [o.to(cpu_device) for o in output]
            # print(output)
            # print(output[0].bbox)
            image_ids = [target.get_field("idx") for target in targets]
        results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, output)}
        )
        
    return results_dict


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("fcos_core.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def inference(
        model,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
        txt_path=None,
        json_path=None,
        iou_thresh=0.5,
        ignore_iou=0.5,
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("fcos_core.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    predictions = compute_on_dataset(model, data_loader, device, inference_timer)  # {img_id:BBox[...],,,}
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )

    if output_folder:  # inference/coco_ICDAR2015_test
        txt_floder = os.path.join(output_folder, 'txt_result')
        if not os.path.exists(txt_floder):
            os.mkdir(txt_floder)
        # save_txt_func(txt_path,txt_floder,predictions,dataset)  # txt_path:img_id???img name?????????txt_floder?????????????????????predictions???????????????
    f1, recall, precision = evaluate(json_path,predictions,iou_thresh,dataset,ignore_iou,txt_floder,print_txt=True)
    print("f1: " + str(f1))
    print("recall: " + str(recall))
    print("precision: " + str(precision))
    
    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    if not is_main_process():
        return

    # print(predictions)
    # print(len(predictions))

    # if output_folder:
    #     torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    # extra_args = dict(
    #     box_only=box_only,
    #     iou_types=iou_types,
    #     expected_results=expected_results,
    #     expected_results_sigma_tol=expected_results_sigma_tol,
    # )

    # return evaluate(dataset=dataset,
    #                 predictions=predictions,
    #                 output_folder=output_folder,
    #                 **extra_args)
    print('finish test!')
