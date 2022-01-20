import os
import pandas as pd
from torchvision.io import read_image
from matplotlib.pyplot import imread
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import cv2
from pathlib import Path
import random
import json 

"""
dataset.py contain all the function related to the data.
"""
def load_data(img_path): 
    
    """Load training data from file in ``.npz`` format."""
    img = cv2.imread(img_path)
    img = img[:,:,:2]
    return img

def train_test_split(annotation_file , ratio_train = 0.8,seeds = 42) :
    np.random.seed(seeds)
    # load the files 
    files = pd.read_pickle(annotation_file)
    # load the different patient and split it in different fold
    patient = np.unique([row['file'].split('_')[0] for index, row in files.iterrows()])
    nb_patient = patient.shape[0]
    patient = patient[np.random.permutation(nb_patient)]
    split = {}
    split['train'] = [ row['file'] for index, row in files.iterrows() if row['file'].split('_')[0] in patient[:int(nb_patient*ratio_train)] ]
    split['val']  = [ row['file'] for index, row in files.iterrows() if row['file'].split('_')[0] in patient[int(nb_patient*ratio_train):] ]
    #Check 
    l_train = [ row['label'] for index, row in files.iterrows() if row['file'].split('_')[0] in patient[:int(nb_patient*ratio_train)] ]
    l_test = [ row['label'] for index, row in files.iterrows() if row['file'].split('_')[0] in patient[int(nb_patient*ratio_train):] ]

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
    files_no_aug = files.loc[files['aug']=='no']
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
        for i in range(k-1) :
            patient_fold.append( patient_MI[i*k_MI:(i+1)*k_MI] + patient_non_MI[i*k_non_MI:(i+1)*k_non_MI] ) 
        patient_fold.append(patient_MI[(k-1)*k_MI:] + patient_non_MI[(k-1)*k_non_MI:] ) 
    else : 
        k_ = int(len(patient)/k)

        patient_fold = []
        for i in range(k-1) :
            patient_fold.append( patient[i*k_:(i+1)*k_] ) 
        patient_fold.append( patient[(k-1)*k_] )
    
    np.random.seed()
    patient_fold = []
    for i in range(k-1):
        patient_fold.append( patient_MI[i*k_MI:(i+1)*k_MI] + patient_non_MI[i*k_non_MI:(i+1)*k_non_MI] ) 
    patient_fold.append(patient_MI[(k-1)*k_MI:] + patient_non_MI[(k-1)*k_non_MI:] ) 

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

def cv_iterate(i,folds_train, folds_test, folds_train_label = None, shuffle = True):
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

    def __init__(self, img_dir,annotations_img_id = None, annotations_file_path = None, transform=False, target_transform=None ,
                     to_tensor = True ,norm = True , norm_imgnet = False, apply_attention = False ):

        self.annotations_img_id = annotations_img_id
        self.annotations_file_pd = pd.read_pickle(annotations_file_path)
        self.annotations_file_pd.set_index('img_id', inplace = True)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.to_tensor = to_tensor 
        self.norm = norm 
        self.norm_imgnet = norm_imgnet
        self.apply_attention = apply_attention

    def __len__(self):
        return len(self.annotations_img_id)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.annotations_img_id[idx])
        image = load_data(img_path)
        label = self.annotations_file_pd.loc[self.annotations_img_id[idx]].target
        if self.apply_attention : 
            image = np.expand_dims(image[:,:,0]*(image[:,:,1]/255.0),axis= 2)

        #print(label)
        #assert len(label) == 1
        if self.to_tensor == True :
            trsf = T.ToTensor()
            image = trsf(image)
            if self.apply_attention : 
                image = image/255
            
        if (label == 1) and (self.transform):
            transform = T.Compose([T.RandomHorizontalFlip(0.5) , T.RandomVerticalFlip(0.5)])
            transform_sharp = T.Compose([T.RandomAdjustSharpness(sharpness_factor=2,p=0.5)])
            image = transform(image)

            if not self.apply_attention : 
                img_1 = image[:1]
                img_2 = image[1:]
                image = torch.cat((transform_sharp(img_1),img_2),dim = 0)
            else : 
                image = transform_sharp(image)
                
        if self.target_transform:
            label = self.target_transform(label)
        
        # normalize settings
        if self.norm  :
            if self.norm_imgnet :
                trsf2 = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            elif self.apply_attention : 
                trsf2 = T.Normalize(mean=[0.449], std=[0.229]) # std=[0.08233915, 0.08233915]
            else :
                trsf2 = T.Normalize(mean=[0.449, 0.449], std=[0.226, 0.226]) # std=[0.08233915, 0.08233915]

            image = trsf2(image)

        return image.float(), label

def data_augmentation(dir_image,dir_image_save):
    """
    It augments the less representative class in order to balance the dataset 
    """
    
    annotated_path = os.path.join(dir_image,'annotated_box')
    annotated = pd.read_pickle(annotated_path)
    annotated_culprit = annotated.loc[annotated['label'] ==1]
    annotated_non_culprit = annotated.loc[annotated['label'] ==0]
    file_to_add = []
    for i,row in annotated_culprit.iterrows():
        file_to_add_tmp = []
        image_tmp = []
        img_path = os.path.join(dir_image, row['file'])
        image,label = load_data(img_path)
        file_to_add_tmp.append(row['file'])
        image_tmp.append(image)
        file_to_save = row['file'].split('.')[0] + '_aug_90.npz'
        img_90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        file_to_add_tmp.append(file_to_save)
        image_tmp.append(img_90)

        file_to_save = row['file'].split('.')[0] + '_aug_270.npz'
        img_270 = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        file_to_add_tmp.append(file_to_save)
        image_tmp.append(img_270)

        file_to_save = row['file'].split('.')[0] + '_aug_180.npz'
        img_180 = cv2.rotate(image, cv2.ROTATE_180)
        file_to_add_tmp.append(file_to_save)
        image_tmp.append(img_180)

        file_to_save = row['file'].split('.')[0] + '_aug_f.npz'
        img_f = cv2.flip(image, 1)
        file_to_add_tmp.append(file_to_save)
        image_tmp.append(img_f)

        file_to_save = row['file'].split('.')[0] + '_aug_f90.npz'
        img_f_90 = cv2.rotate(img_f, cv2.ROTATE_90_CLOCKWISE)
        file_to_add_tmp.append(file_to_save)
        image_tmp.append(img_f_90)

        file_to_save = row['file'].split('.')[0] + '_aug_f180.npz'
        img_f_180 = cv2.rotate(img_f, cv2.ROTATE_180)
        file_to_add_tmp.append(file_to_save)
        image_tmp.append(img_f_180)

        for name,img in zip(file_to_add_tmp,image_tmp):
            save_path = os.path.join(dir_image_save,name )
            file_ = Path(save_path).with_suffix('.npz')
            file_.parent.mkdir(parents=True,exist_ok=True)
            np.savez(str(save_path),image= np.asarray(img),label =np.asarray(label))
        file_to_add += file_to_add_tmp
    annotated_aug_culpritpd = pd.DataFrame(list(zip(file_to_add,[1 for i in file_to_add])),columns = ['file','label'])

    for ind, row in annotated_non_culprit.iterrows():
        
        img_path = os.path.join(dir_image, row['file'])
        image,label = load_data(img_path)

        save_path = os.path.join(dir_image_save,row['file'] )
        file_ = Path(save_path).with_suffix('.npz')
        file_.parent.mkdir(parents=True,exist_ok=True)
        np.savez(str(save_path),image= np.asarray(image),label =np.asarray(label))
    
    #concat les both culprot and non culprit dataframe 
    annotated_aug = pd.concat([annotated_aug_culpritpd,annotated_non_culprit])
    annotated_aug.to_pickle(os.path.join(dir_image_save,'annotated_box' ),protocol=4)



