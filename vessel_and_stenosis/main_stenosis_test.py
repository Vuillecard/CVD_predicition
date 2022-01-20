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

def run(data_dir, split_data, model_parameter, mode, samplers ,which): 
    
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
        model_ft = initialize_model(model_parameter['name'], model_parameter['num_classes'],num_inputs_chanel=3, feature_extract= False, 
                                pretrained_model=pretrained_model,load_best_model=load_best_model , dropout = model_parameter['dropout'])

    print("Initializing Datasets and Dataloaders...")

    if which == 'augmentation' : 
        # create some transformations to augment the culprit class
        transform_stenosis = T.Compose([T.RandomCrop(size=(224,224)),T.RandomHorizontalFlip(),T.RandomVerticalFlip()])
        # Create training and validation datasets
        
        # Create training and validation dataloaders
        image_datasets_train = {'train': CardioDataset(data_dir,annotations_file = split_data['train'],
                                    transform =None,to_tensor=True,norm=True,norm_imgnet = True, transform_stenosis = transform_stenosis ) ,
                                'val': CardioDataset(data_dir,annotations_file = split_data['val'],
                                    transform =None,to_tensor=True,norm=True,norm_imgnet = True, transform_stenosis = T.Compose([T.CenterCrop(size=224)]) )}
        dataloaders_dict = {x: DataLoader(image_datasets_train[x], batch_size=model_parameter['batch_size'] ) for x in ['train', 'val']}
    else :
        # create some transformations to augment the culprit class
        transform_stenosis = T.Compose([T.CenterCrop(size=224)])
        # Create training and validation datasets
        
        # Create training and validation dataloaders
        image_datasets_train = {'train': CardioDataset(data_dir,annotations_file = split_data['train'],
                                    transform =None,to_tensor=True,norm=True,norm_imgnet = True, transform_stenosis = transform_stenosis ) ,
                                'val': CardioDataset(data_dir,annotations_file = split_data['val'],
                                    transform =None,to_tensor=True,norm=True,norm_imgnet = True, transform_stenosis = T.Compose([T.CenterCrop(size=224)]) )}
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

def prepare_data( train_path, test_path):
    files_train = pd.read_pickle(train_path)
    files_test = pd.read_pickle(test_path)
    #shuffle the training data: 
    files_train = files_train.sample(frac=1).reset_index(drop=True)
    
    # save the train and test img id
    train = list(files_train.img_id.values)
    test = list(files_test.img_id.values)
    # take the taget of the train to generate a sampler
    train_label = list(files_train.target.values)
    class_sample_count =np.unique(train_label, return_counts=True)[1]
    weight = 1. / class_sample_count
    samples_weight = weight[train_label]
    samples_weight = torch.from_numpy(samples_weight)
    weight = torch.from_numpy(weight)
    #samples_weigth = samples_weight.double()
    sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight, len(samples_weight))
    samplers = {'weight': weight,
                'sampler': sampler}
    
    split_data = {'train': train,
                  'val': test }

    return samplers , split_data

def main_test(data_dir,results_save, model_parameter,num_runs,which,seed, modes):
    """ Main to test the model the data is split according to the train and test files,
        the """
    annotated_file = os.path.join(data_dir , model_parameter['annotated_file_train'])
    annotated_file_test = os.path.join(data_dir , model_parameter['annotated_file_test'])

    samplers , split_data = prepare_data( annotated_file, annotated_file_test)

    for mode in modes:
        print(mode,which)
        for k in range(num_runs):

            perf = run(data_dir,split_data,model_parameter,mode, samplers, which)
            name_result = which+'_perf_' + mode + '_run_'+str(k+1) + '.pkl'

            with open(os.path.join(results_save, name_result ), 'wb') as fp:
                pickle.dump(perf, fp)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run baseline for CVD detection')
    parser.add_argument('--typerun', type=str, default='None', metavar='V',
                        help='version of the code',required=True)
    parser.add_argument('--model', type=str, default='None', metavar='M',
                        help='version of the model',required=True)
    parser.add_argument('--path_annot', type=str, default='None', metavar='P',
                        help='annotated version',required=False)
    args = parser.parse_args()

    data_path = '/data/cardio/SPUM/CVD_detection_code/Data_CVD'
    results_dir = '/data/cardio/SPUM/CVD_detection_code/cnn/results'
    # Cross validation parameters
    num_fold = 5
    seed = 6 # yield the best split among the patient 
    # Model parameters
    model_parameter = {'name': None ,
                       'annotated_file': None ,
                       'num_classes' : 2 ,
                       'batch_size' : 64 ,
                       'num_epochs' : 30 ,
                       'dropout' : False,
                       'lr' : 0.01 ,
                       'wd' : 0 ,
                       'path_save_model': None} 

    model_parameter['name'] = args.model
    model_parameter['annotated_file'] = args.path_annot
    
    if args.typerun == 'stenosis_baseline_test':
        print('Begin testing ...')
        num_runs = 5
        model_parameter['annotated_file_train'] = 'annotated_box_cv'
        model_parameter['annotated_file_test']  = 'annotated_box_test'

        # from the hyper parameter tuning 
        modes = ['scratch']
        model_parameter['lr'] = 0.01
        model_parameter['wd'] = 0.0
        model_parameter['dropout'] = False
        main_test(data_dir = os.path.join(data_path, 'data_CNN_stenosis' ),results_save = os.path.join(results_dir, args.typerun ), 
                    model_parameter = model_parameter ,num_runs = num_runs,which = 'baseline',seed = seed,modes=modes)
        
        modes = ['ft_imgnet']
        model_parameter['lr'] = 0.007
        model_parameter['wd'] = 0.055
        model_parameter['dropout'] = False
        main_test(data_dir = os.path.join(data_path, 'data_CNN_stenosis' ),results_save = os.path.join(results_dir, args.typerun ), 
                    model_parameter = model_parameter ,num_runs = num_runs,which = 'baseline',seed = seed,modes=modes)
        
        modes = ['ft_best']
        model_parameter['lr'] = 0.003
        model_parameter['wd'] = 0.0
        model_parameter['dropout'] = False
        main_test(data_dir = os.path.join(data_path, 'data_CNN_stenosis' ),results_save = os.path.join(results_dir, args.typerun ), 
                    model_parameter = model_parameter ,num_runs = num_runs,which = 'baseline',seed = seed,modes=modes)


    if args.typerun == 'stenosis_augmentation_test' :
        print('Begin testing ...')
        num_runs = 5
        model_parameter['annotated_file_train'] = 'annotated_box_cv'
        model_parameter['annotated_file_test']  = 'annotated_box_test'

        # from the hyper parameter tuning 
        modes = ['scratch']
        model_parameter['lr'] = 0.03
        model_parameter['wd'] = 0.0
        model_parameter['dropout'] = False
        main_test(data_dir = os.path.join(data_path, 'data_CNN_stenosis' ),results_save = os.path.join(results_dir, args.typerun ), 
                    model_parameter = model_parameter ,num_runs = num_runs,which = 'augmentation',seed = seed,modes=modes)
        
        modes = ['ft_imgnet']
        model_parameter['lr'] = 0.004
        model_parameter['wd'] = 0.055
        model_parameter['dropout'] = False
        main_test(data_dir = os.path.join(data_path, 'data_CNN_stenosis' ),results_save = os.path.join(results_dir, args.typerun ), 
                    model_parameter = model_parameter ,num_runs = num_runs,which = 'augmentation',seed = seed,modes=modes)
        
        modes = ['ft_best']
        model_parameter['lr'] = 0.004
        model_parameter['wd'] = 0.01
        model_parameter['dropout'] = False
        main_test(data_dir = os.path.join(data_path, 'data_CNN_stenosis' ),results_save = os.path.join(results_dir, args.typerun ), 
                    model_parameter = model_parameter ,num_runs = num_runs,which = 'augmentation',seed = seed,modes=modes)

    