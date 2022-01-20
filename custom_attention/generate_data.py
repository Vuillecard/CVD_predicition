import cv2
import numpy as np
import os
import json
import pandas as pd 
from os import path as osp
from pathlib import Path
from skimage import io 

from gaussian_attention import create_gaussian_attention_aug

""" 
Convert original images into a dataset for CNN with custom gaussian attention

data_CNN_ATT
    annotated_box #contain the information about the images
    img_0.npz
    img_1.npz
    .
    .
"""
cluster = '/data'
local = 'Z:/'
main = cluster
# import one images to test 
patient_per_folder = {}
list_dir = ['Farhang MI with or without revasc','Farhang MI without revasc','Farhang NCL','Farhang NCL 2',
            'Farhang NCL 3','Farhang NCL 4', 'Farhang NCL 5_low_quality']

data_dir = os.path.join(main,'cardio/SPUM/Labelling_EPFL_CHUV')

for folder in list_dir:
    patient_per_folder[folder] = os.listdir(os.path.join(data_dir,folder))

dir_img = os.path.join(main,'cardio/SPUM/CVD_detection_code/Data_CVD/data_CNN_ATT')

# load patient images information 
patient_info = pd.read_pickle(os.path.join(main,'cardio/SPUM/CVD_detection_code/Data_CVD/patient_image_info_clean'))

patient_info['path_box'] = patient_info.path_box.apply(lambda x: x.replace('\\','/'))
if main == cluster :
    print('change path')
    patient_info['path_box'] = patient_info.path_box.apply(lambda x: x.replace('Z:','/data'))


annotated_line = []
col_names = ['img_id' ,'path_img', 'target', 'patient', 'segment', 'color','view', 'aug']
img_id = -1
for idx, row in patient_info.iterrows():

    split_path = row['path_box'].split('/')
    img_which = split_path[-1]
    img_patient = [ name for name in patient_per_folder[split_path[-3]] if name.startswith( split_path[-2].split('.')[0] ) ][0]
    img_path = osp.join(data_dir,split_path[-3],img_patient,img_which)

    if not os.path.isfile(img_path[:-4]+' copie.tif'):
        continue
    img_id = create_gaussian_attention_aug(img_id , img_path, row,annotated_line,dir_img)
    print(idx,img_id)

df_annotated_box = pd.DataFrame(annotated_line,columns = col_names)
df_annotated_box.to_pickle(os.path.join(dir_img,'annotated_box' ),protocol = 4)


        