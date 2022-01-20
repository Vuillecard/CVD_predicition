from matplotlib.pyplot import ylabel
import numpy as np
import cv2
import os
import torch

from maskrcnn_benchmark.modeling.rrpn.anchor_generator import draw_anchors
from maskrcnn_benchmark.utils import cv2_util

from predictor import Predictor
from iou import intersection_over_union_aligned

def get_random_color():
    return (np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255))

def return_box(file_annot , img) :
    f = np.load(file_annot, allow_pickle=True)
    X, Y = f['polys'], f['label']

    
    # need to swap the axis because of the images x, y axis is different
    # only to read from the annotated file .npz 
    if 0 in Y: 
        return []
    polygons_ = X.copy()
    polygons_[:,:,0] = X[:,:,1]
    polygons_[:,:,1] = X[:,:,0]
    polygons_ = polygons_.astype(np.float32)
    boxes = []
    for poly in polygons_:

        # firt plot the box and then we plot the rectangle inside 
        seg_data_arr = poly if type(poly[0][0]) in [list, np.ndarray] else [poly]
        concat_arr = np.concatenate(seg_data_arr)
        bbox = np.array([np.amin(concat_arr, axis=0), np.amax(concat_arr, axis=0)]).reshape(4)
        #bbox[2:] -= bbox[:2] # to convert into xywh
        bbox = np.int0(bbox).tolist()
        cv2.rectangle(img, tuple(bbox[:2]), tuple(bbox[2:]), (255, 0, 0), 2)
        boxes.append(bbox)
        #area = sum([cv2.contourArea(arr) for arr in seg_data_arr])
        

        rect = cv2.minAreaRect(poly.astype(int))
        box = cv2.boxPoints(rect) # cv2.boxPoints(rect) for OpenCV 3.x
        box = np.int0(box)
        cv2.drawContours(img,[box],0,(0,255,0),2)
    return boxes

if __name__ == '__main__':
    import glob

    confidence_threshold = 0.01

    # CLASSES = ["__background__", "MI"]

    #config_file = "./model_save/with_FPN/config.yml"
    #model_file = "./model_save/with_FPN/model_final.pth"

    model_type = 'baseline'
    model_file = "./model_save/%s/model_final.pth"%(model_type)
    config_file= "./model_save/%s/config.yml"%(model_type)
    save_dir = './image_prediction/%s'%(model_type)
    image_dir = "/data/cardio/SPUM/CVD_detection_code/Data_CVD/data_RCNN_MI/Images/Val"
    annotated_dir = "/data/cardio/SPUM/CVD_detection_code/Data_CVD/data_RCNN_MI/Annotations/Val"
    image_files = glob.glob("%s/*.jpg"%(image_dir))

    prediction_model = Predictor(config_file, min_score=confidence_threshold, device="cuda")
    prediction_model.load_weights(model_file)
    image_cnt = -1
    
    for image_file in image_files: #glob.glob("%s/*%s"%(image_dir, image_ext)):
        img = cv2.imread(image_file)

        #print(image_file)
        get_id = image_file.split('_')[-1][:-4]
        #print(get_id)
        file_annot = os.path.join(annotated_dir,'gt_img_%s.npz'%(get_id))

        image_cnt +=1
        if img is None:
            print("Could not find %s"%(image_file))
            continue

        img_copy = img.copy()

        # Plot the true box in green
        bbox_true = return_box(file_annot , img_copy)

        data = prediction_model.run_on_opencv_image(img)
        if not data:
            print("No predictions for image")
            continue

        scores = data["scores"]
        bboxes = data["bboxes"]
        has_labels = "labels" in data
        has_rrects = "rrects" in data
        has_masks = "masks" in data

        bboxes = np.round(bboxes).astype(np.int32)
        if image_cnt == 1 : 
            print(bbox_true[0])
            print(bboxes[0])
            print(intersection_over_union_aligned(torch.tensor([bboxes[0]], dtype=torch.float32),torch.tensor([bbox_true[0]], dtype=torch.float32),box_format="corners"))

        for ix, (bbox, score) in enumerate(zip(bboxes[:5], scores[:5])):

            if has_labels:
                label = data["labels"][ix]

            if has_rrects:
                rr = data["rrects"][ix]
                img_copy = draw_anchors(img_copy, [rr], [[0,0,255]])
                img_copy = cv2.rectangle(img_copy, tuple(bbox[:2]), tuple(bbox[2:]), (255, 0, 0), 2)
                cv2.putText(img_copy, str(round(score,2)),(int(bbox[0]), int(bbox[1]-10)),0, 2, (0,0,255), 2)
            else:
                img_copy = cv2.rectangle(img_copy, tuple(bbox[:2]), tuple(bbox[2:]), (0, 0, 255), 2)

            if has_masks:
                mask = data["masks"][ix]

                contours, hierarchy = cv2_util.findContours(
                    mask.astype(np.uint8) * 255, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
                )
                color = get_random_color()
                img_copy = cv2.drawContours(img_copy, contours, -1, color, 3)

        # from maskrcnn_benchmark.modeling.rotate_ops import merge_rrects_by_iou
        # if has_masks and has_rrects:
        #     img_copy2 = img.copy()
        #
        #     match_inds = merge_rrects_by_iou(data["rrects"], iou_thresh=0.5)
        #
        #     masks = data["masks"]
        #     for idx, inds in match_inds.items():
        #         if len(inds) == 0:
        #             continue
        #         mask = masks[inds[0]]
        #         for ix in inds[1:]:
        #             mask = np.logical_or(mask, masks[ix])
        #
        #         _, contours, hierarchy = cv2.findContours(
        #             mask.astype(np.uint8) * 255, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        #         )
        #         color = get_random_color()
        #         img_copy2 = cv2.drawContours(img_copy2, contours, -1, color, 3)
        #
        #     cv2.imshow("pred_merged", img_copy2)

        #cv2.imshow("img", img)
        #cv2.imshow("pred", img_copy)
        #cv2.waitKey(0)

        cv2.imwrite( save_dir+ '/true'+str(image_cnt)+'.png', img)
        cv2.imwrite( save_dir+ '/pred '+str(image_cnt)+'.png', img_copy)
        

