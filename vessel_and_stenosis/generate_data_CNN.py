import os 
import pandas as pd 
from utiles import image_preprocessing_test, crop_images2, crop_images4
import pickle
import glob
import numpy as np 
import cv2
from pathlib import Path
from utiles import image_preprocessing_dot_extract, aggregat_info

"""
This file contain the function used to preprocess the annotated images.
It extracts the boxes coordinate with the associated class MI or non MI. 
"""
def detect_dot(root= '/Volumes/cardio-project/'):
    """ This function create a file with for each patient and each view colect the coordinate of 
    the stenosis and the class associated to it and then save it."""
    data_dir = root + 'cardio/SPUM/Labelling_EPFL_CHUV'
    save_info_images = root + 'cardio/SPUM/CVD_detection_code/Data_CVD'
    list_dir = ['Farhang MI with revasc 2nd annotation']
  
    data_frame_list = []
    col_names = ['path_box', 'patient_box', 'segment_box', 'view_box', 'color_box', 'coordinate_box', 'categories_box', 'coord_stenosis']

    prev_dataf = False
    if os.path.isfile(os.path.join(save_info_images,'patient_image_info_annot')) :
        print('find file')
        df_ = pd.read_pickle(os.path.join(save_info_images,'patient_image_info_annot'))
        patient_check = df_.patient_box.values
        prev_dataf = True
    else :
        patient_check = []

    for name in list_dir:
        print(name)
        path_group = os.path.join(data_dir,name)
        patients = os.listdir(path_group)
        for patient in patients :

            if not patient in patient_check:
                path_patient = os.path.join(path_group,patient)
                for path, subdirs, files in os.walk(path_patient):
                    images = [ x for x in files if (x.split(' ')[-1] == '2.tif') and (x.split('.')[-1] == 'tif') and (x[0] != '.') ]
                    for image in images:
                        path_image = os.path.join(path_patient, image)
                        image_info = image_preprocessing_dot_extract(path_image,False,False)
                        aggregat_info(data_frame_list,image_info,col_names)

                df = pd.DataFrame(data_frame_list, columns =col_names)
                if prev_dataf:
                    df = pd.concat([df_,df])
                df.to_pickle(os.path.join(save_info_images,'patient_image_info_annot'),protocol=4)

def detect_boxes():
    """ This function create a file with for each patient and each view colect the boxes coordinate 
    and the class associated to it and then save it."""
    data_dir = '/data/cardio/SPUM/Labelling_EPFL_CHUV'
    save_info_images = '/data/cardio/SPUM/CVD_detection_code/Data_CVD'
    list_dir = ['Farhang MI with or without revasc',
                'Farhang MI without revasc',
                'Farhang NCL',
                'Farhang NCL 2',
                'Farhang NCL 3',
                'Farhang NCL 4', 
                'Farhang NCL 5_low_quality']
    print(__file__)
    data_frame_list = []
    col_names = ['path_box', 'patient_box', 'segment_box', 'view_box', 'color_box', 'coordinate_box', 'categories_box']

    prev_dataf = False
    if os.path.isfile(os.path.join(save_info_images,'patient_image_info')) :
        print('find file')
        df_ = pd.read_pickle(os.path.join(save_info_images,'patient_image_info'))
        patient_check = df_.patient_box.values
        prev_dataf = True
    else :
        patient_check = []

    for name in list_dir:
        path_group = os.path.join(data_dir,name)
        patients = os.listdir(path_group)
        for patient in patients :

            if not patient in patient_check:
                path_patient = os.path.join(path_group,patient)
                for path, subdirs, files in os.walk(path_patient):
                    images = [ x for x in files if (x.split(' ')[-1] != 'copie.tif') and (x.split('.')[-1] == 'tif') and (x[0] != '.') ]
                    for image in images:
                        path_image = os.path.join(path_patient, image)
                        image_info = image_preprocessing_test(path_image,False,False)
                        aggregat_info(data_frame_list,image_info,col_names)

                df = pd.DataFrame(data_frame_list, columns =col_names)
                if prev_dataf:
                    df = pd.concat([df_,df])
                df.to_pickle(os.path.join(save_info_images,'patient_image_info'),protocol=4)

def prepare_annotation():
    """
    Crop the images in funciton of the coordinate from the annotated file 
    """
    image_dir = '/data/cardio/SPUM/CVD_detection_code/Data_CVD/data_CNN'
    df_image_info = '/data/cardio/SPUM/CVD_detection_code/Data_CVD/patient_2_view_info_cluster'

    crop_images2(image_dir, df_image_info, time_aug_MI = 10)
    

def preprare_annotation_stenosis():
    image_dir = '/data/cardio/SPUM/CVD_detection_code/Data_CVD/data_CNN_stenosis'
    df_image_info = '/data/cardio/SPUM/CVD_detection_code/Data_CVD/patient_image_info_annot_clean'

    crop_images4(image_dir, df_image_info)
    

def load_data(file_path): 
    
    """Load training data from file in ``.npz`` format."""
    f = np.load(file_path, allow_pickle=True)
    X, Y = f['image'], f['label']
    Y=np.squeeze(Y)
    return X,Y

def save_image_file(dir_image, name, img, label):
    save_path = os.path.join(dir_image,name )
    file_ = Path(save_path).with_suffix('.npz')
    file_.parent.mkdir(parents=True,exist_ok=True)
    np.savez(str(save_path),image= np.asarray(img),label =np.asarray(label))

def data_augmentation(dir_image):
    """
    It augments the less representative class in order to balance the dataset 
    """
    annotated_path = os.path.join(dir_image,'annotated_box')
    image_files = glob.glob("%s/*.npz"%(dir_image))
   
    annotated = pd.read_pickle(annotated_path)
    tot_image = annotated.shape[0]
    annotated_culprit = annotated.loc[annotated['target'] ==1]
    img_idx = tot_image -1 

    file_to_add = []
    print_idx =0
    for i,row in annotated_culprit.iterrows():
        print_idx += 1
        print(print_idx)
        line = row.values

        img_path = os.path.join(dir_image, row['img_id'])
        image,label = load_data(img_path)
        assert label == 1
        img_90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        img_idx += 1
        line[0] = 'img_%d.npz'%(img_idx)
        line[-1] = 'rot_0'
        file_to_add.append(line)
        save_image_file(dir_image, line[0], img_90, label)
        
        img_270 = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img_idx += 1
        line[0] = 'img_%d.npz'%(img_idx)
        line[-1] = 'rot_1'
        file_to_add.append(line)
        save_image_file(dir_image, line[0], img_270, label)

        img_180 = cv2.rotate(image, cv2.ROTATE_180)
        img_idx += 1
        line[0] = 'img_%d.npz'%(img_idx)
        line[-1] = 'rot_2'
        file_to_add.append(line)
        save_image_file(dir_image, line[0], img_180, label)

        img_f = cv2.flip(image, 1)
        img_idx += 1
        line[0] = 'img_%d.npz'%(img_idx)
        line[-1] = 'rot_3'
        file_to_add.append(line)
        save_image_file(dir_image, line[0], img_f, label)

        img_f_90 = cv2.rotate(img_f, cv2.ROTATE_90_CLOCKWISE)
        img_idx += 1
        line[0] = 'img_%d.npz'%(img_idx)
        line[-1] = 'rot_4'
        file_to_add.append(line)
        save_image_file(dir_image, line[0], img_f_90, label)

        img_f_180 = cv2.rotate(img_f, cv2.ROTATE_180)
        img_idx += 1
        line[0] = 'img_%d.npz'%(img_idx)
        line[-1] = 'rot_5'
        file_to_add.append(line)
        save_image_file(dir_image, line[0], img_f_180, label)

    col_names = ['img_id', 'path_img1', 'path_img2', 'target', 'patient', 'segment', 'color', 'aug']
    annotated_aug_culpritpd = pd.DataFrame(file_to_add, columns = col_names )
    annotated_aug_culpritpd = pd.concat([annotated, annotated_aug_culpritpd])
    annotated_aug_culpritpd.reset_index(drop=True,inplace=True)
    annotated_aug_culpritpd.to_pickle(os.path.join(dir_image,'annotated_box_aug'),protocol=4)


if __name__ == '__main__':

    # For the stenosis detection:
    # first detect the dot 
    print('start detect dot')
    detect_dot()
    # then prepare the dataset by cropping around the images
    print('prepate annotation stenosis')
    preprare_annotation_stenosis()

    # For the patches vessel classification :
    # first detect the dot
    detect_boxes()
    print('prepare annotation')
    prepare_annotation()
    print('Augment_dataset')
    data_augmentation('/data/cardio/SPUM/CVD_detection_code/Data_CVD/data_CNN')
