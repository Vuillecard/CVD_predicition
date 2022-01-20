
import argparse
from implementation import train_model
from model import Large_input_ResNet
from dataset import *
import torch 
import torch.nn as nn
import torch.optim as optim  
import torchvision.transforms as T
from torch.utils.data import DataLoader
import pickle
from pathlib import Path
import os

def run(data_dir,annotated_file, split_data, model_parameter, mode, samplers , which): 
    
    mode_traing = { 'scratch': {'feature_extract': False,'pretrained_model': False ,'load_best_model':False},
                    'ft_imgnet':{'feature_extract': False,'pretrained_model': True ,'load_best_model':False},
                    'ft_best':{'feature_extract': False,'pretrained_model': True ,'load_best_model':True}}

    setting = mode_traing[mode]
    pretrained_model = setting['pretrained_model']
    load_best_model = setting['load_best_model']
    
    print("Initializing Models...")
    model_ft = Large_input_ResNet( in_channel = model_parameter['input_channel'],num_classes = model_parameter['num_classes'], pretrained=pretrained_model,
                                    load_best_model=load_best_model , dropout = model_parameter['dropout'])

    print("Initializing Datasets and Dataloaders...")
    
    # Create training and validation dataloaders
    if which == 'augmented_culprit' : 
        # create some transformations to augment the culprit class
        image_datasets_train = {'train': CardioDataset(data_dir,annotations_img_id = split_data['train'],annotations_file_path = annotated_file ,transform =True, to_tensor=True, norm=True,apply_attention = model_parameter['apply_attention'] ) ,
                                'val': CardioDataset(data_dir,annotations_img_id = split_data['val'], annotations_file_path = annotated_file ,transform =False, to_tensor=True, norm=True,apply_attention = model_parameter['apply_attention'] ) }
        dataloaders_dict = {'train': DataLoader(image_datasets_train['train'], batch_size=model_parameter['batch_size'], sampler =samplers['sampler']) ,
                            'val': DataLoader(image_datasets_train['val'], batch_size=model_parameter['batch_size'])}
    elif which == 'weighted_dataloader' :
        image_datasets_train = {x: CardioDataset(data_dir,annotations_img_id = split_data[x],annotations_file_path = annotated_file,
                                   transform =None,to_tensor=True,norm=True,apply_attention = model_parameter['apply_attention'] ) for x in ['train', 'val']}
        dataloaders_dict = {'train': DataLoader(image_datasets_train['train'], batch_size=model_parameter['batch_size'], sampler =samplers['sampler']) ,
                            'val': DataLoader(image_datasets_train['val'], batch_size=model_parameter['batch_size'])}
    else :
        image_datasets_train = {x: CardioDataset(data_dir,annotations_img_id = split_data[x],annotations_file_path = annotated_file,
                                transform =None,to_tensor=True,norm=True ,apply_attention = model_parameter['apply_attention']) for x in ['train', 'val']}
        dataloaders_dict = {x: DataLoader(image_datasets_train[x], batch_size=model_parameter['batch_size'],shuffle=True ) for x in ['train', 'val']}

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        # finetunning learn all parameter but slower rate for base parmeter
        to_learn = [model_ft.reduce_input,model_ft.backbone.fc]
        ignored_params = []
        for module in to_learn : 
            ignored_params += list(map(id, module.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params, model_ft.parameters())
        optimizer_ft = optim.SGD([
                {'params': base_params},
                {'params': model_ft.reduce_input.parameters(), 'lr': model_parameter['lr']},
                {'params': model_ft.backbone.fc.parameters(), 'lr': model_parameter['lr']}], lr=model_parameter['lr']*0.1, momentum=0.9,weight_decay=model_parameter['wd'])

    elif mode == 'scratch' : 
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
    modes = ['scratch','ft_imgnet','ft_best']

    folds_train, folds_test, folds_test_label, folds_train_label = k_fold(num_fold,annotated_file,results_save,seed)

    for mode in modes:
        for k in range(num_fold):

            split_data, samplers = cv_iterate(k,folds_train, folds_test, folds_train_label) 
            perf = run(data_dir,annotated_file,split_data,model_parameter,mode, samplers , which)
            name_result = which+'_perf_' + mode + '_cv_'+str(k+1) + '.pkl'

            with open(os.path.join(results_save, name_result ), 'wb') as fp:
                pickle.dump(perf, fp)

def main_CV_tunning(data_dir,results_save, model_parameter,num_fold,which,seed,experiment,modes):
    annotated_file = os.path.join(data_dir , model_parameter['annotated_file'])

    
    folds_train, folds_test, folds_test_label, folds_train_label = k_fold(num_fold,annotated_file,results_save,seed)
    
    for mode in modes:
        print(mode,which)
        for k in range(num_fold):

            split_data, samplers = cv_iterate(k,folds_train, folds_test, folds_train_label)
            perf = run(data_dir,annotated_file,split_data,model_parameter,mode, samplers , which)
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
                perf = run(data_dir,annotated_file,split_data,model_parameter,mode, samplers , which)
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
    parser.add_argument('--apply_filter', type=bool, default=False, metavar='P',
                        help='apply the custom attention to the model',required=False)
    parser.add_argument('--input_channel', type=int, default=2, metavar='P',
                        help='number of input channel',required=False)
    args = parser.parse_args()

    data_path = '/data/cardio/SPUM/CVD_detection_code/Data_CVD'
    results_dir = '/data/cardio/SPUM/CVD_detection_code/contrastive_learning/results'
    # Cross validation parameters
    num_fold = 5
    seed = 42
    # Model parameters
    model_parameter = {'name': None ,
                       'annotated_file': None ,
                       'num_classes' : 2 ,
                       'batch_size' : 64 ,
                       'num_epochs' : 20 ,
                       'dropout' : False,
                       'lr' : 0.002 ,
                       'wd' : 0 ,
                       'input_channel' : 2,
                       'path_save_model': None ,
                       'apply_attention' : False } 

    model_parameter['name'] = args.model
    model_parameter['annotated_file'] = args.path_annot
    model_parameter['input_channel'] = args.input_channel
    model_parameter['apply_attention'] = args.apply_filter
    print('Input_channel, model_parameter',model_parameter['input_channel'], model_parameter['apply_attention'])
    print('model_parameter input_channel',model_parameter['input_channel'])
    
    if args.typerun == 'baseline' :
        main_CV(os.path.join(data_path, 'data_CNN_ATT' ),os.path.join(results_dir, args.model ),
                     model_parameter,num_fold,args.typerun,seed) 

    if args.typerun == 'weighted_dataloader' :
        main_CV(os.path.join(data_path, 'data_CNN_ATT' ),os.path.join(results_dir, args.model),
                     model_parameter,num_fold,args.typerun,seed)

    if args.typerun == 'weighted_loss' :
        main_CV(os.path.join(data_path, 'data_CNN_ATT' ),os.path.join(results_dir, args.model ),
                     model_parameter,num_fold,args.typerun,seed)

    if args.typerun == 'augmented_culprit' :
        model_parameter['annotated_file'] = args.path_annot + '_aug'
        main_CV(os.path.join(data_path, 'data_CNN_ATT' ),os.path.join(results_dir, args.model ),
                     model_parameter,num_fold,args.typerun,seed)

    if args.typerun == 'test_augmented_culprit' :
        model_parameter['annotated_file'] = args.path_annot + '_aug'
        main_CV(os.path.join(data_path, 'data_CNN_ATT' ),os.path.join(results_dir, args.typerun ),
                     model_parameter,num_fold,'augmented_culprit',seed)
    

    if args.typerun == 'tuning_balance_channel':

        num_fold = 4
        model_parameter['name'] = args.model
        model_parameter['annotated_file'] = args.path_annot
        model_parameter['input_channel'] = 2
        model_parameter['apply_attention'] = False

        dict_hyper_tuning = {}
        lrs= [ 0.004, 0.007,0.01, 0.03]
        wds = [ 0, 0.01,0.055, 0.1, 0.3 ]
        
        which = 'weighted_dataloader'
        mode = 'ft_imgnet'
        exp_id = 0
        for lr in lrs:
            for wd in wds:
                exp_id +=1
                exp = 'exp_%d'%(exp_id)
                param = {}
                param['lr'] = lr
                param['wd'] = wd
                
                save_names = []
                for k in range(num_fold):
                    save_names.append(exp+'_'+which+'_perf_'+mode+'_cv_'+str(k+1) + '.pkl')
                param['paths'] = save_names
                dict_hyper_tuning[exp] = param

                model_parameter['lr'] = lr
                model_parameter['wd'] = wd
                
                main_CV_tunning(data_dir=os.path.join(data_path, 'data_CNN_ATT' ),results_save = os.path.join(results_dir, args.typerun),
                model_parameter=model_parameter ,num_fold = num_fold ,which = which ,seed = seed,experiment = exp, modes = [mode])

        with open(os.path.join(results_dir, args.typerun,'config_prameter_tuning.json'), 'w') as outfile:
            json.dump(dict_hyper_tuning, outfile,indent=4)

    if args.typerun == 'tuning_balance_apply':

        num_fold = 4
        model_parameter['name'] = args.model
        model_parameter['annotated_file'] = args.path_annot
        model_parameter['input_channel'] = 1
        model_parameter['apply_attention'] = True

        dict_hyper_tuning = {}
        lrs= [0.004, 0.007,0.01, 0.03]
        wds = [0, 0.01,0.055, 0.1, 0.3]
        
        which = 'weighted_dataloader'
        mode = 'ft_imgnet'
        exp_id = 0
        for lr in lrs:
            for wd in wds:
                exp_id +=1
                exp = 'exp_%d'%(exp_id)
                param = {}
                param['lr'] = lr
                param['wd'] = wd
                
                save_names = []
                for k in range(num_fold):
                    save_names.append(exp+'_'+which+'_perf_'+mode+'_cv_'+str(k+1) + '.pkl')
                param['paths'] = save_names
                dict_hyper_tuning[exp] = param

                model_parameter['lr'] = lr
                model_parameter['wd'] = wd
                
                main_CV_tunning(data_dir=os.path.join(data_path, 'data_CNN_ATT' ),results_save = os.path.join(results_dir, args.typerun),
                model_parameter=model_parameter ,num_fold = num_fold ,which = which ,seed = seed,experiment = exp, modes = [mode])

        with open(os.path.join(results_dir, args.typerun,'config_prameter_tuning.json'), 'w') as outfile:
            json.dump(dict_hyper_tuning, outfile,indent=4)
