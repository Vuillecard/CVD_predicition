import os
import pandas as pd
from torchvision.io import read_image
from matplotlib.pyplot import imread
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2
from pathlib import Path
import random
import json

"""
This file contain all the function to load and spli the data for the training time.
"""

def load_data(file_path): 
    """Load training data from file in ``.npz`` format."""
    f = np.load(file_path, allow_pickle=True)
    X, Y = f['image'], f['label']
    Y=np.squeeze(Y)
    return X,Y

def train_test_split(annotation_file , ratio_train = 0.8,seeds = 42) :
    """
    Split the data into a train and test set. It makes sur no images from train and 
    test belong to the same patient.
    """
    np.random.seed(seeds)
    # load the files 
    files = pd.read_pickle(annotation_file)
    # load the different patient and split it in different fold
    patient = np.unique(files.patient)
    nb_patient = patient.shape[0]
    patient = patient[np.random.permutation(nb_patient)]
    nb_MI_patients = files.groupby(['patient']).target.sum()
    patient_MI = np.array(list((nb_MI_patients>1).index[np.where((nb_MI_patients>1)==True)[0]]))
    patient_non_MI = np.array([pat for pat in patient if not pat in patient_MI])
    # shuffle the patient 
    patient_MI = patient_MI[np.random.permutation(len(patient_MI))].tolist()
    patient_non_MI = patient_non_MI[np.random.permutation(len(patient_non_MI))].tolist()
    # split the patient into train and test set 
    patient_train = patient_MI[:int(len(patient_MI)*ratio_train)] + patient_non_MI[:int(len(patient_non_MI)*ratio_train)]
    patient_test = patient_MI[int(len(patient_MI)*ratio_train):] + patient_non_MI[int(len(patient_non_MI)*ratio_train):]
    split = {}
    split['train'] = [ row['img_id'] for index, row in files.iterrows() if row['patient'] in patient_train ]
    split['val']  = [ row['img_id'] for index, row in files.iterrows() if row['patient'] in patient_test ]
    #Check 
    l_train = [ row['target'] for index, row in files.iterrows() if row['patient'] in patient_train ]
    l_test = [ row['target'] for index, row in files.iterrows() if row['patient'] in patient_test ]

    print('train ',np.sum(l_train)/len(l_train)*100)
    print('train ',np.sum(l_test)/len(l_test)*100)

    class_sample_count =np.unique(l_train, return_counts=True)[1]
    weight = 1. / class_sample_count
    samples_weight = weight[l_train]
    samples_weight = torch.from_numpy(samples_weight)
    weight = torch.from_numpy(weight)
    #samples_weigth = samples_weight.double()
    sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight, len(samples_weight))
    samplers = {'weight': weight ,
                'sampler': sampler}
    return split, samplers

def k_fold(k,annotation_file,results_save, seeds = 42) :
    """
    Split the data in k fold. First split the patient that contain MI. Then split MI patient 
    into k folds and add the other patient to complet the folds. 
    Consider also test fold where there are no augmentation in it.
    """
    np.random.seed(seeds)
    # load the files
    files = pd.read_pickle(annotation_file)
    if 'aug' in files.columns:
        files_no_aug = files.loc[files['aug']=='no']
    else : 
        files_no_aug = files
    # load the different patient and split it in different fold
    patient = np.unique(files.patient)
    nb_patient = patient.shape[0]
    patient = patient[np.random.permutation(nb_patient)]
    nb_MI_patients = files.groupby(['patient']).target.sum()
    patient_MI = np.array(list((nb_MI_patients>1).index[np.where((nb_MI_patients>1)==True)[0]]))
    patient_non_MI = np.array([pat for pat in patient if not pat in patient_MI])

    # if number of non MI patient is not sufficient just slipt the patient
    if len(patient_non_MI) > k :
        # shuffle the patient 
        patient_MI = patient_MI[np.random.permutation(len(patient_MI))].tolist()
        patient_non_MI = patient_non_MI[np.random.permutation(len(patient_non_MI))].tolist()
        k_MI = int(len(patient_MI)/k)
        k_non_MI = int(len(patient_non_MI)/k)
        np.random.seed()
        patient_fold = []
        for i in range(k-1):
            patient_fold.append( patient_MI[i*k_MI:(i+1)*k_MI] + patient_non_MI[i*k_non_MI:(i+1)*k_non_MI] ) 
        patient_fold.append(patient_MI[(k-1)*k_MI:] + patient_non_MI[(k-1)*k_non_MI:] ) 
    else : 
        k_ = int(len(patient)/k)

        patient_fold = []
        for i in range(k-1):
            patient_fold.append( patient[i*k_:(i+1)*k_] ) 
        patient_fold.append(patient[(k-1)*k_]  )

    np.random.seed()
    folds_train = {}
    folds_test = {}
    folds_test_label = {}
    folds_train_label = {}
    for i in range(k):
        folds_train['fold_'+str(i+1)] = [ row['img_id'] for index, row in files.iterrows() if row['patient'] in patient_fold[i]]
        folds_train_label['fold_'+str(i+1)] = [ row['target'] for index, row in files.iterrows() if row['patient'] in patient_fold[i]]
        folds_test['fold_'+str(i+1)] = [ row['img_id'] for index, row in files_no_aug.iterrows() if row['patient'] in patient_fold[i]]
        folds_test_label['fold_'+str(i+1)] = [ row['target'] for index, row in files_no_aug.iterrows() if row['patient'] in patient_fold[i]]
    repartition_save = {
        'training' : {} ,
        'testing' : {}
    }
    print('fold train repartition')
    for k,v in folds_train_label.items():
        print(k,np.sum(v)/len(v)*100)
        repartition_save['training'][k] = np.sum(v)/len(v)*100

    print('fold test repartition')
    for k,v in folds_test_label.items():
        print(k,np.sum(v)/len(v)*100)
        repartition_save['testing'][k] = np.sum(v)/len(v)*100

    with open(os.path.join(results_save,'folds_split.json'), 'w') as outfile:
        json.dump(repartition_save, outfile)

    return folds_train, folds_test, folds_test_label, folds_train_label

def cv_iterate(i,folds_train, folds_test, folds_train_label = None, shuffle = False):
    """ Function that iterate through the different fold and create a train and validation set"""
    fold_names = list(folds_train.keys())
    fold_test_name = fold_names[i]
    fold_train_names = [ f for f in fold_names if f!=fold_test_name]
    fold_test = folds_test[fold_test_name]
    fold_train = []
    fold_train_label = []
    for fold_train_name in fold_train_names:
        fold_train += folds_train[fold_train_name]
        if folds_train_label:
            fold_train_label += folds_train_label[fold_train_name]

    #shuffle
    if shuffle :
        #seed = np.random.randint(1000)
        seed = i
        random.seed(seed)
        random.shuffle(fold_train)
        if folds_train_label:
            random.seed(seed)
            random.shuffle(fold_train_label)

    if folds_train_label:
        class_sample_count =np.unique(fold_train_label, return_counts=True)[1]
        weight = 1. / class_sample_count
        samples_weight = weight[fold_train_label]
        samples_weight = torch.from_numpy(samples_weight)
        weight = torch.from_numpy(weight)
        #samples_weigth = samples_weight.double()
        sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight, len(samples_weight))
        samplers = {'weight': weight ,
                    'sampler': sampler}
    else:
        samplers = None

    split = {  'train' : fold_train,
               'val' : fold_test}

    return split , samplers

class CardioDataset(Dataset):

    """ Class Dataset, load and transform the data. It load the data from an input directory and 
    annotated file with image_file, features, labels """

    def __init__(self, img_dir,annotations_file_path=None, annotations_file =None, transform=None, target_transform=None ,
                     to_tensor = True ,norm = True , norm_imgnet = False, transform_stenosis = None ):
        assert(annotations_file_path!=annotations_file)
        if annotations_file :
            self.annotations_file = annotations_file 
        else: 
            self.annotations_file = pd.read_pickle(annotations_file_path)['img_id'].values
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.to_tensor = to_tensor 
        self.norm = norm 
        self.norm_imgnet = norm_imgnet
        self.transform_stenosis = transform_stenosis

    def __len__(self):
        return len(self.annotations_file)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.annotations_file[idx])
        image,label = load_data(img_path)
        

        if self.to_tensor==True : 
            trsf = transforms.ToTensor()
            image = trsf(image)

        if self.transform_stenosis :
            image = self.transform_stenosis(image)

        if (label ==1) and (self.transform):
            img_1 = self.transform(image[:1])
            img_2 = self.transform(image[1:])
            image = torch.cat((img_1,img_2),dim = 0 )

        if self.target_transform:
            label = self.target_transform(label)
        
        # normalize settings
        if self.norm==True :
            if self.norm_imgnet :
                trsf2 = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            else :
                trsf2 = transforms.Normalize(mean=[0.449, 0.449], std=[0.226, 0.226])
            image = trsf2(image)

        return image, label