import os
import numpy as np

def save_txt_func(txt_path,txt_floder,predictions,dataset):
    # print(predictions)  {img_id:BoxList(num_boxes=100, image_width=1333, image_height=750, mode=xyn)}
    imgid_imgname = {}
    txt_path_file = open(txt_path)
    for line in txt_path_file:
        img_id,img_name = line.strip('\n').split('\t')
        imgid_imgname[img_id] = img_name
    for img_id in predictions:
    # for img_id, prediction in enumerate(predictions):
        prediction = predictions[img_id]
        img_info = dataset.get_img_info(img_id)
        image_width = img_info["width"]
        image_height = img_info["height"]
        prediction = prediction.resize((image_width, image_height))

        img_name = imgid_imgname[str(img_id)]
        # print(predictions)
        output_txt = open(os.path.join(txt_floder,'res_'+img_name.replace('.jpg','.txt')), 'w')
        # box_list = predictions[img_id].bbox
        box_list = prediction.bbox
        box_list = np.array(box_list)
        for box in box_list:
            box = box.tolist()
            box = list(map(str, box))
            output_txt.write(','.join(box)+'\n')
        output_txt.close()
        
    