import numpy as np
import cv2
import os
import torch
import json 

from collections import Counter
from apex import amp
from maskrcnn_benchmark import _Custom as _C
from maskrcnn_benchmark.modeling.rrpn.anchor_generator import draw_anchors
from maskrcnn_benchmark.utils import cv2_util

from predictor import Predictor
from matplotlib import pyplot as plt
from iou import intersection_over_union_aligned , intersection_over_union_rotated


def intersection_over_union(box1, box2, method = None):
    """ Define the intersection over union to work with 
    rotated rectangle """
    if method == 'rotated':
        rotate_iou_matrix = amp.float_function(_C.rotate_iou_matrix)
        #print(box1)
        #box1_t = torch.tensor([list(box1])
        #box2_t = torch.tensor([box2])
        return rotate_iou_matrix(torch.tensor([box1], dtype=torch.float32),
                                    torch.tensor([box2], dtype=torch.float32))

    if method == 'rotated_my':
        return intersection_over_union_rotated(box1,box2,over_truth = True)
    else : 
        return intersection_over_union_aligned(torch.tensor([box1], dtype=torch.float32), 
                                                torch.tensor([box2], dtype=torch.float32),
                                                box_format="corners")

def return_true_rrect(file_annot , img) :
    f = np.load(file_annot, allow_pickle=True)
    X, Y = f['polys'], f['label']

    if 0 in Y :
        return [] , []
    # need to swap the axis because of the images x, y axis is different
    # only to read from the annotated file .npz 
    polygons_ = X.copy()
    polygons_[:,:,0] = X[:,:,1]
    polygons_[:,:,1] = X[:,:,0]
    
    rects = []
    boxes = []

    for poly in polygons_:

        seg_data_arr = poly if type(poly[0][0]) in [list, np.ndarray] else [poly]
        concat_arr = np.concatenate(seg_data_arr)
        bbox = np.array([np.amin(concat_arr, axis=0), np.amax(concat_arr, axis=0)]).reshape(4)
        #bbox[2:] -= bbox[:2] # to convert into xywh
        bbox = np.int0(bbox).tolist()
        boxes.append(bbox)

        rect = cv2.minAreaRect(poly.astype(int))
        rects.append( [rect[0][0], rect[0][1], rect[1][0], rect[1][1], rect[2]] )

    return rects , boxes

def convert_rect_to_pts(anchor):
    x_c, y_c, w, h, theta = anchor
    rect = ((x_c, y_c), (w, h), theta)
    rect = cv2.boxPoints(rect)
    # rect = np.int0(np.round(rect))
    return rect

def stand_rect(rect_pred):
    poly = convert_rect_to_pts(rect_pred)
    rect = cv2.minAreaRect(poly.astype(int))
    return [rect[0][0], rect[0][1], rect[1][0], rect[1][1], rect[2]]

def mean_average_precision(
    pred_boxes, true_boxes, iou_threshold=0.5, num_classes=1 , method_iou = None
):
    """
    Calculates mean average precision 
    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x_c, y_c, w, h,theta]
        true_boxes (list): Similar as pred_boxes except all the correct ones 
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes
    Returns:
        float: mAP value across all classes given a specific IoU threshold 
    """
    # used for numerical stability later on
    epsilon = 1e-6
    average_precision = None
    precisions = None
    recalls = None
    for c in range(1,num_classes+1):
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)
        
        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    detection[3:],
                    gt[3:],method_iou)
                #print('rotated', intersection_over_union(detection[3:],gt[3:],'rotated') )
                #print('rotated_new' ,intersection_over_union(detection[3:],gt[3:],'rotated_my') )
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1
        
        print('total_true_bboxes',total_true_bboxes)
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        precisions = torch.cat((torch.tensor([1.0]), precisions)).numpy()
        recalls = torch.cat((torch.tensor([0.0]), recalls)).numpy()
        # torch.trapz for numerical integration
        average_precision = np.trapz(precisions, recalls)
       
    return average_precision , precisions , recalls

if __name__ == '__main__':
    import glob

    area = [32 ** 2, 96 ** 2]

    # CLASSES = ["__background__", "MI"]

    model_type = 'baseline'
    model_file = "./model_save/%s/model_final.pth"%(model_type)
    config_file= "./model_save/%s/config.yml"%(model_type)
    
    image_dir = "/data/cardio/SPUM/CVD_detection_code/Data_CVD/data_RCNN_MI/Images/Val"
    annotated_dir = "/data/cardio/SPUM/CVD_detection_code/Data_CVD/data_RCNN_MI/Annotations/Val"
    output_dir = "./model_save/%s/cvd_pred_other.json"%(model_type)
    image_files = glob.glob("%s/*.jpg"%(image_dir))
    prediction_model = Predictor(config_file, min_score=0.001, device="cuda")
    prediction_model.load_weights(model_file)

    results = {}
    results['model_file'] = model_file

    for method_iou in ['rotated_my','rotated','bbox']:
        results_m = {}
        for max_pred in [5,10,-1] :
            
            predictions = []
            ground_truths = []
            
            for image_file in image_files: #glob.glob("%s/*%s"%(image_dir, image_ext)):
                img = cv2.imread(image_file)

                #print(image_file)
                get_id = image_file.split('_')[-1][:-4]
                #print(get_id)
                file_annot = os.path.join(annotated_dir,'gt_img_%s.npz'%(get_id))

                
                if img is None:
                    print("Could not find %s"%(image_file))
                    continue

                img_copy = img.copy()

                # Plot the true box in green
                rr_true, bb_true = return_true_rrect(file_annot , img_copy)

                for idx_true, rect in enumerate(rr_true):
                    l = [get_id,1,1]
                    if 'rotated' in method_iou :
                        l += list(rect)
                        area_true = l[-2]*l[-3]
                        if area_true <= area[1] and area_true >= area[0] :
                            print('m',area_true)
                    else : 
                        l += bb_true[idx_true]    
                    ground_truths.append(l)


                data = prediction_model.run_on_opencv_image(img)
                if not data:
                    print("No predictions for image")
                    continue

                scores = data["scores"]
                label = data["labels"]
                rr = data["rrects"]
                bboxes = data["bboxes"]
                bboxes = np.round(bboxes).astype(int)
                for idx_pred, score in enumerate(scores[:max_pred]):
                    
                    l = [get_id,label[idx_pred],score]
                    if 'rotated' in method_iou :
                        l += stand_rect(rr[idx_pred])
                        area_true = l[-2]*l[-3]
                        if area_true <= area[1] and area_true >= area[0] :
                            print('m',area_true)
                    else :
                        l.append(bboxes[idx_pred])
                    
                    predictions.append(l)
            
            print(method_iou)
            for thre in [0.25,0.5,0.75]: 
                average_precision , precisions , recalls = mean_average_precision( predictions,
                                                                            ground_truths,
                                                                            iou_threshold=thre,
                                                                            method_iou = method_iou)
                
                results_m['AP@'+str(thre)+'_maxdet_'+str(max_pred)] = float(average_precision)
                results_m['AR@'+str(thre)+'_maxdet_'+str(max_pred)] = float(np.mean(recalls))
            mAP = []
            mAR = []
            for thre in np.linspace(0.25,0.75,10) :
                
                average_precision , precisions , recalls = mean_average_precision( predictions,
                                                                            ground_truths,
                                                                            iou_threshold=thre,
                                                                            method_iou = method_iou)
                mAP.append(average_precision)
                mAR.append(np.mean(recalls))
            
            results_m['mAP@0.25:0.75_maxdet_'+str(max_pred)] = float(np.mean(mAP))
            results_m['mAR@0.25:0.75_maxdet_'+str(max_pred)] = float(np.mean(mAR))
                
        results[method_iou] = results_m

    
    with open(output_dir, 'w') as fp:
        json.dump(results, fp,indent=4)

                                                           