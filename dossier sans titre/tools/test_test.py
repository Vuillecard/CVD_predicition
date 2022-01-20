
x = False if False else False

print(x)

import torch
from maskrcnn_benchmark import _Custom as _C
from apex import amp

rotate_iou_matrix = amp.float_function(_C.rotate_iou_matrix)

boxes1 = torch.tensor([[672.4067, 290.7776, 791.0275,  38.9333,  34.1454]])
boxes2 = torch.tensor([[672.4067, 290.7776, 38.9333, 791.0275 ,  -34.1454]])
print(rotate_iou_matrix(boxes1, boxes2))

#If I shift boxes1 a little bit.
boxes2[0][0] = 672.3
print(rotate_iou_matrix(boxes1, boxes2))
