import argparse
from implementation import train_model
from models_init import initialize_model,Siamese
from dataset import *
import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader
import pickle
from pathlib import Path
import os
import json

def run(data_dir, split_data, model_parameter, mode, samplers , which): 
    
    mode_traing = {'scratch': {'feature_extract': False,'pretrained_model': False ,'load_best_model':False},
                   'ft_imgnet':{'feature_extract': False,'pretrained_model': True ,'load_best_model':False},
                   'ft_best':{'feature_extract': False,'pretrained_model': True ,'load_best_model':True}}

    setting = mode_traing[mode]
    
    pretrained_model = setting['pretrained_model']
    load_best_model = setting['load_best_model']
    
    print("Initializing Models...")
    if model_parameter['name'] == 'siamese' :
        model_ft = Siamese(pretrained_model=pretrained_model,load_best_model=load_best_model , dropout = model_parameter['dropout'])
    else : 
        model_ft = initialize_model(model_parameter['name'], model_parameter['num_classes'], feature_extract= False, 
                                pretrained_model=pretrained_model,load_best_model=load_best_model , dropout = model_parameter['dropout'])

    print("Initializing Datasets and Dataloaders...")

    # create some transformations to augment the culprit class
    transform = T.Compose([ T.RandomHorizontalFlip(0.5) , T.RandomVerticalFlip(0.5), T.RandomAdjustSharpness(sharpness_factor=2,p=0.5),
                            T.RandomApply([T.RandomRotation(degrees=(89,91)) ],p=0.5), T.RandomApply([T.RandomRotation(degrees=(269,271)) ],p=0.5) ])

    # Create training and validation datasets
    
    # Create training and validation dataloaders
    if which == 'augmented_culprit' : 
        image_datasets_train = {'train': CardioDataset(data_dir,annotations_file = split_data['train'], transform =transform, to_tensor=True, norm=True ) ,
                                'val': CardioDataset(data_dir,annotations_file = split_data['val'], transform =None, to_tensor=True, norm=True ) }
        dataloaders_dict = {'train': DataLoader(image_datasets_train['train'], batch_size=model_parameter['batch_size'], sampler =samplers['sampler']) ,
                            'val': DataLoader(image_datasets_train['val'], batch_size=model_parameter['batch_size'])}
    elif which == 'weighted_dataloader' :
        image_datasets_train = {x: CardioDataset(data_dir,annotations_file = split_data[x],
                                   transform =None,to_tensor=True,norm=True ) for x in ['train', 'val']}
        dataloaders_dict = {'train': DataLoader(image_datasets_train['train'], batch_size=model_parameter['batch_size'], sampler =samplers['sampler']) ,
                            'val': DataLoader(image_datasets_train['val'], batch_size=model_parameter['batch_size'])}
    else :
        image_datasets_train = {x: CardioDataset(data_dir,annotations_file = split_data[x],
                                    transform =None,to_tensor=True,norm=True ) for x in ['train', 'val']}
        dataloaders_dict = {x: DataLoader(image_datasets_train[x], batch_size=model_parameter['batch_size'] ) for x in ['train', 'val']}

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device', device)
    # Send the model to GPU
    model_ft = model_ft.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters but slower for the base 
    #  paramter. However, if we are doing feature extract method, we will 
    #  only update the parameters that we have just initialized, 
    #  i.e. the parameters with requires_grad is True.
    print("Initializing optimizer...")

    if mode[:2] == 'ft' :
        print('mode ---> ',mode)
        if model_parameter['name'] == 'siamese' :
            # finetunning learn all parameter but slower rate for base parmeter
            to_learn = [model_ft.classifier]
            ignored_params = []
            for module in to_learn : 
                ignored_params += list(map(id, module.parameters()))
            base_params = filter(lambda p: id(p) not in ignored_params, model_ft.parameters())
            optimizer_ft = optim.SGD([
                    {'params': base_params},
                    {'params': model_ft.classifier.parameters(), 'lr': model_parameter['lr']}], lr=model_parameter['lr']*0.1, momentum=0.9,weight_decay=model_parameter['wd'])
        else : 
            # finetunning learn all parameter but slower rate for base parmeter
            to_learn = [model_ft.conv1,model_ft.bn1,model_ft.fc]
            ignored_params = []
            for module in to_learn : 
                ignored_params += list(map(id, module.parameters()))
            base_params = filter(lambda p: id(p) not in ignored_params, model_ft.parameters())
            optimizer_ft = optim.SGD([
                    {'params': base_params},
                    {'params': model_ft.conv1.parameters(), 'lr': model_parameter['lr']},
                    {'params': model_ft.bn1.parameters(), 'lr': model_parameter['lr']},
                    {'params': model_ft.fc.parameters(), 'lr': model_parameter['lr']}], lr=model_parameter['lr']*0.1, momentum=0.9,weight_decay=model_parameter['wd'])
            
            # optimizer_ft = optim.Adam(model_ft.parameters(), lr=model_parameter['lr'],weight_decay=model_parameter['wd'])
    if mode == 'scratch' : 
        print('mode ---> scratch')
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=model_parameter['lr'], momentum=0.9,weight_decay=model_parameter['wd'])

    else:
        print('No valide mode')
    
    # Setup the loss founction
    if which == 'weighted_loss':
        criterion = nn.CrossEntropyLoss(weight = samplers['weight'].float().to(device))
    if 'debug_test' in which :
        print('debug loss')
        criterion = nn.CrossEntropyLoss(weight = torch.tensor(model_parameter['weightloss']).float().to(device))
    else : 
        criterion = nn.CrossEntropyLoss() 
    print('weight is ', samplers['weight'])
    # Train and evaluate
    perf = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft,
                                    device,path_save_model= model_parameter['path_save_model'] ,
                                    num_epochs=model_parameter['num_epochs'])

    return perf

def main_CV(data_dir,results_save, model_parameter,num_fold,which,seed):
    annotated_file = os.path.join(data_dir , model_parameter['annotated_file'])
    modes = ['ft_imgnet']
    
    folds_train, folds_test, folds_test_label, folds_train_label = k_fold(num_fold,annotated_file,results_save,seed)
    
    for mode in modes:
        print(mode,which)
        for k in range(num_fold):

            split_data, samplers = cv_iterate(k,folds_train, folds_test, folds_train_label)
            perf = run(data_dir,split_data,model_parameter,mode, samplers , which)
            name_result = which+'_perf_' + mode + '_cv_'+str(k+1) + '.pkl'

            with open(os.path.join(results_save, name_result ), 'wb') as fp:
                pickle.dump(perf, fp)

def main_CV_tunning(data_dir,results_save, model_parameter,num_fold,which,seed,experiment):
    annotated_file = os.path.join(data_dir , model_parameter['annotated_file'])
    modes = ['ft_imgnet']
    
    folds_train, folds_test, folds_test_label, folds_train_label = k_fold(num_fold,annotated_file,results_save,seed)
    
    for mode in modes:
        print(mode,which)
        for k in range(num_fold):

            split_data, samplers = cv_iterate(k,folds_train, folds_test, folds_train_label)
            perf = run(data_dir,split_data,model_parameter,mode, samplers , which)
            name_result = experiment+'_'+ which+'_perf_' + mode + '_cv_'+str(k+1) + '.pkl'

            with open(os.path.join(results_save, name_result ), 'wb') as fp:
                pickle.dump(perf, fp)



def main_CV_tune_weight(data_dir,results_save, model_parameter,num_fold,which,seed):
    annotated_file = os.path.join(data_dir , 'annotated_box_aug')
    #modes = ['scratch','ft_imgnet','ft_best']
    modes = ['scratch']
    folds_train, folds_test, folds_test_label, folds_train_label = k_fold(num_fold,annotated_file,results_save,seed)
    weight_settings = [[0.25,1],[0.5,1],[0.75,1]]
    for p, weight_setting in enumerate(weight_settings):
        print(weight_setting)
        model_parameter['weightloss'] = weight_setting
        for mode in modes:

            for k in range(num_fold):

                split_data, samplers = cv_iterate(k,folds_train, folds_test, folds_train_label) 
                perf = run(data_dir,split_data,model_parameter,mode, samplers , which)
                name_result = which+'_perf_' + mode + '_wl_'+str(p+1)+'_cv_'+str(k+1) + '.pkl'

                with open(os.path.join(results_save, name_result ), 'wb') as fp:
                    pickle.dump(perf, fp)
 
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run baseline for CVD detection')
    parser.add_argument('--typerun', type=str, default='None', metavar='V',
                        help='version of the code',required=True)
    parser.add_argument('--model', type=str, default='None', metavar='M',
                        help='version of the model',required=True)
    parser.add_argument('--path_annot', type=str, default='None', metavar='P',
                        help='annotated version',required=True)
    args = parser.parse_args()

    data_path = '/data/cardio/SPUM/CVD_detection_code/Data_CVD'
    results_dir = '/data/cardio/SPUM/CVD_detection_code/cnn/results'
    # Cross validation parameters
    num_fold = 5
    seed = 42
    # Model parameters
    model_parameter = {'name': None ,
                       'annotated_file': None ,
                       'num_classes' : 2 ,
                       'batch_size' : 64 ,
                       'num_epochs' : 25 ,
                       'dropout' : False,
                       'lr' : 0.002 ,
                       'wd' : 0 ,
                       'path_save_model': None} 

    model_parameter['name'] = args.model
    model_parameter['annotated_file'] = args.path_annot
    
    if args.typerun == 'tuning_augment':
        dict_hyper_tuning = {}
        lrs= [ 0.0005, 0.001 ,0.003, 0.005]
        wds = [ 0, 0.01, 0.1, 0.2 ]
        drps = [ True, False] 
        #lrs= [ 0.001 ]
        #wds = [ 0 ]
        #drps = [ True, False] 

        exp_id = 0
        for lr in lrs:
            for wd in wds:
                for drp in drps : 
                    exp_id +=1
                    exp = 'exp_%d'%(exp_id)
                    param = {}
                    param['lr'] = lr
                    param['wd'] = wd
                    param['drp'] = drp
                    save_names = []
                    for k in range(num_fold):
                        save_names.append(exp+'_augmented_culprit_perf_ft_imgnet_cv_'+str(k+1) + '.pkl')
                    param['paths'] = save_names
                    dict_hyper_tuning[exp] = param

                    model_parameter['lr'] = lr
                    model_parameter['wd'] = wd
                    model_parameter['dropout'] = drp

                    main_CV_tunning(data_dir=os.path.join(data_path, 'data_CNN' ),results_save = os.path.join(results_dir, args.typerun),
                    model_parameter=model_parameter ,num_fold = num_fold ,which = 'augmented_culprit' ,seed = seed,experiment = exp)

        with open(os.path.join(results_dir, args.typerun,'config_prameter_tuning.json'), 'w') as outfile:
            json.dump(dict_hyper_tuning, outfile,indent=4)

    
    if args.typerun == 'baseline' :
        main_CV(os.path.join(data_path, 'data_CNN' ),os.path.join(results_dir, args.model ),
                     model_parameter,num_fold,args.typerun,seed) 

    if args.typerun == 'weighted_dataloader' :
        main_CV(os.path.join(data_path, 'data_CNN' ),os.path.join(results_dir, args.model),
                     model_parameter,num_fold,args.typerun,seed)

    if args.typerun == 'weighted_loss' :
        main_CV(os.path.join(data_path, 'data_CNN' ),os.path.join(results_dir, args.model ),
                     model_parameter,num_fold,args.typerun,seed)

    if args.typerun == 'augmented_culprit' :
        model_parameter['annotated_file'] = args.path_annot + '_aug'
        main_CV(os.path.join(data_path, 'data_CNN' ),os.path.join(results_dir, args.model ),
                     model_parameter,num_fold,args.typerun,seed)

    if args.typerun == 'test_augmented_culprit' :
        model_parameter['annotated_file'] = args.path_annot + '_aug'
        main_CV(os.path.join(data_path, 'data_CNN' ),os.path.join(results_dir, args.typerun ),
                     model_parameter,num_fold,'augmented_culprit',seed)
    

    if args.typerun == 'debug_test01' :
        # Test to increase a lot the weighted_loss to see if it predict only the culprit class
        model_parameter = {'name': "resnet18" ,
                       'num_classes' : 2 ,
                       'batch_size' : 64 ,
                       'num_epochs' : 10 ,
                       'dropout' : False,
                       'lr' : 0.001 ,
                       'wd' : 0.0,
                       'path_save_model': None}
        main_CV(os.path.join(data_path, 'data_CNN' ),os.path.join(results_dir, args.typerun ),
                     model_parameter,num_fold,args.typerun,seed)   

    if args.typerun == 'debug_test02' :
        num_fold = 3
        # Test to increase a lot the weighted_loss to see if it predict only the culprit class
        model_parameter = {'name': "resnet18" ,
                       'num_classes' : 2 ,
                       'batch_size' : 64 ,
                       'num_epochs' : 10 ,
                       'dropout' : False,
                       'lr' : 0.001 ,
                       'wd' : 0.0,
                       'path_save_model': None}
        main_CV_tune_weight(os.path.join(data_path, 'data_CNN' ),os.path.join(results_dir, args.typerun ),
                     model_parameter,num_fold,args.typerun,seed)