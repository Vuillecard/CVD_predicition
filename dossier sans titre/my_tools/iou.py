import torch
import shapely.geometry
import shapely.affinity

class RotatedRect:
    def __init__(self, cx, cy, w, h, angle):
        self.cx = cx
        self.cy = cy
        self.w = w
        self.h = h
        self.angle = angle

    def get_contour(self):
        w = self.w
        h = self.h
        c = shapely.geometry.box(-w/2.0, -h/2.0, w/2.0, h/2.0)
        rc = shapely.affinity.rotate(c, self.angle)
        return shapely.affinity.translate(rc, self.cx, self.cy)

    def intersection(self, other):
        return self.get_contour().intersection(other.get_contour())

def intersection_over_union_rotated(boxes_preds, boxes_labels, over_truth = False):
    """
    Calculate intersection over union for rotated box of the form [xc,yc,w,h,angle]
    """
    
    r1_pred = RotatedRect(boxes_preds[0] , boxes_preds[1] , boxes_preds[2], boxes_preds[3], boxes_preds[4])
    r2_true = RotatedRect(boxes_labels[0] , boxes_labels[1] , boxes_labels[2], boxes_labels[3], boxes_labels[4])

    intersection = r1_pred.intersection(r2_true).area

    if intersection == 0.0 :
        return 0.0

    if over_truth : 
        iou = intersection / r2_true.get_contour().area
    else : 
        iou = intersection / (r1_pred.get_contour().area +r2_true.get_contour().area - intersection + 1e-6)
    
    return iou


def intersection_over_union_aligned(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Calculates intersection over union
    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct Labels of Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
    Returns:
        tensor: Intersection over union for all examples
    """
    #print(boxes_preds.size(),boxes_labels.size())
    # Slicing idx:idx+1 in order to keep tensor dimensionality
    # Doing ... in indexing if there would be additional dimensions
    # Like for Yolo algorithm which would have (N, S, S, 4) in shape
    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    elif box_format == "corners":
        box1_x1 = boxes_preds[..., 0]
        box1_y1 = boxes_preds[..., 1]
        box1_x2 = boxes_preds[..., 2]
        box1_y2 = boxes_preds[..., 3]
        box2_x1 = boxes_labels[..., 0]
        box2_y1 = boxes_labels[..., 1]
        box2_x2 = boxes_labels[..., 2]
        box2_y2 = boxes_labels[..., 3]
        
    assert box1_x1 < box1_x2
    assert box1_y1 < box1_y2
    assert box2_x1 < box2_x2
    assert box2_y1 < box2_y2
    
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)
    #print('x1',x1)
    #print('x2',x2)
    #print('y1',y1)
    #print('y2',y2)
    if x2 < x1 or y2 < y1:
        return 0.0

    # Need clamp(0) in case they do not intersect, then we want intersection to be 0
    intersection = (x2 - x1) * (y2 - y1)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
    #print(intersection)
    #print(box1_area)
    #print(box2_area)
    return intersection / (box1_area + box2_area - intersection + 1e-6)