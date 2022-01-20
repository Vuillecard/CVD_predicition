import cv2
import numpy as np
import pandas as pd
from skimage import io
from os import path as osp
import os
from matplotlib import pyplot as plt
from gaussian_attention import create_gaussian_attention

# import one images to test 
data_dir = 'Z:/cardio/SPUM/CVD_detection_code/Data_CVD/data_RCNN'

# load patient images information 
patient_info = pd.read_pickle('Z:/cardio/SPUM/CVD_detection_code/Data_CVD/patient_image_info_clean')

img_dir = osp.join(data_dir, 'Images/Train')
annot_dir = osp.join(data_dir, 'Annotations/Train')

img_id = 4531

path_images = patient_info.iloc[img_id].path_box
print(patient_info.iloc[img_id].color_box)
coordinate_poly = np.asarray(patient_info.iloc[img_id].coordinate_box)

path_images = path_images.replace('/data', 'Z:')
if os.path.isfile(path_images):
    print('it exists !')
print(path_images)
print(coordinate_poly.shape)
# load the images and the annotation annotation

img_original = cv2.imread(path_images)
if img_original.shape[-1] == 4:
    img_original = img_original[:,:,:-1]

plt.imshow(img_original)
plt.show()

img = img_original.copy()
Z = create_gaussian_attention(img,coordinate_poly)
Z = Z/np.max(Z)*255
img[:,:,1] = Z
img[:,:,2] = 0



#img = Z*img
#img = np.concatenate((img,np.expand_dims(Z,axis=2)),axis=2)
print('img', img.shape)
#for i in range(img.shape[0]):
#        img[i,:] = img[i,:]*Z[i,:]

cv2.drawContours(img,[coordinate_poly.astype(int)],0,(255,0,0),2)

dim = (1500,1500)
img = cv2.resize(img, dim, interpolation = cv2.INTER_LINEAR)
plt.imshow(img)
plt.show()
#dst = cv2.addWeighted(img, 0.7, , 0.6, 0)
#
#cv2.imshow('', img)
#cv2.waitKey(0)

print(img.shape)
cv2.imwrite('test.jpg', img)

img_test = cv2.imread('test.jpg')
print(img_test.shape)
plt.imshow(img_test)
plt.show()