import torch 
import numpy as np 
import time
import copy
from sklearn import metrics
from matplotlib import pyplot as plt
import seaborn as sns
import os
import pickle 

"""
This file contain the implementation of the function to train the model and save the metric.
It also implement the visualization of the plots.
"""
def get_data_dir():
    if os.path.dirname(__file__):
        data_path = '//files9.epfl.ch/data/vuilleca/git/Data_CVD'
    else : 
        data_path = '/data/git/Data_CVD'
    return data_path
#*******************************************************
#                   Training 
#******************************************************* 

def compute_metrics(y_true, y_pred, y_pred_prob):
    """ Computing different metric using sklearn implementation based on :
        https://analyticsindiamag.com/evaluation-metrics-in-ml-ai-for-classification-problems-wpython-code/"""
    recall_ = metrics.recall_score(y_true, y_pred, pos_label=1)
    specificity_ = metrics.recall_score(y_true, y_pred, pos_label=0)
    precision_ = metrics.precision_score(y_true, y_pred, pos_label=1)
    f1_ = metrics.f1_score(y_true,y_pred)
    lr_precision_, lr_recall_, _ = metrics.precision_recall_curve(y_true, y_pred_prob)
    auc_ = metrics.auc(lr_recall_, lr_precision_)
    return recall_ , specificity_ ,precision_ ,f1_ ,lr_precision_ ,lr_recall_,auc_

def train_epoch(model,dataloader,criterion,optimizer,device):

    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()   # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0.0
        y_true = []
        y_pred_prob = []
        y_pred = []
        # Iterate over data.
        for inputs, labels in dataloader[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device).long()
            #print(inputs.shape)
            # zero the parameter gradients
            optimizer.zero_grad()
            print(labels.sum()/len(labels))

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                # Get model outputs and calculate loss
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # get the true and prediction for each batch
                if phase == 'val':
                    y_true.append(labels.detach())
                    y_pred_prob.append(outputs[:,1].detach())
                    y_pred.append(preds.detach())

            # statistics
            running_loss += loss.item()*inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        #epoch_loss = running_loss / len(dataloader[phase].dataset)
        
        if phase == 'val':
            epoch_acc = (running_corrects.double() / len(dataloader[phase].dataset)).cpu().numpy()
            epoch_loss = running_loss / len(dataloader[phase].dataset)
            y_true = torch.cat(y_true, dim=0).cpu().numpy()
            y_pred_prob = torch.cat(y_pred_prob, dim=0).cpu().numpy()
            y_pred = torch.cat(y_pred, dim=0).cpu().numpy()
            _ , _ ,_ ,epoch_f1 ,_ ,_,epoch_auc= compute_metrics(y_true, y_pred, y_pred_prob)

    return epoch_loss,epoch_acc ,epoch_f1, epoch_auc
                

def train_model(model, dataloaders, criterion, optimizer,device,keep_pred = False,path_save_model=None, num_epochs=25):
    since = time.time()

    perf = {
    'loss_val' : [], 'loss_train' : [],
    'acc_val' : [], 'acc_train' : [],
    'recall' : [], 'specificity' : [],
    'precision' : [],'f1_score' : [],
    'auc' : [], 'lr_precision_best' : None,
    'lr_recall_best' : None, 'best_epoch' : None,
    'time_epoch':None }
    
    best_f1 = 0
    #best_model_wts = copy.deepcopy(model.state_dict())
    print('starting training')
    for epoch in range(num_epochs):
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            y_true = []
            y_pred_prob = []
            y_pred = []
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device).long()
                #print(inputs.shape)
                # zero the parameter gradients
                optimizer.zero_grad()
                #print(labels.sum()/len(labels))
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # get the true and prediction for each batch
                    if phase == 'val':
                        y_true.append(labels.detach())
                        y_pred_prob.append(outputs[:,1].detach())
                        y_pred.append(preds.detach())

                # statistics
                running_loss += loss.item()*inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = (running_corrects.double() / len(dataloaders[phase].dataset)).cpu().numpy()
            if epoch == 0 :
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val':
                y_true = torch.cat(y_true, dim=0).cpu().numpy()
                y_pred_prob = torch.cat(y_pred_prob, dim=0).cpu().numpy()
                y_pred = torch.cat(y_pred, dim=0).cpu().numpy()
                recall_ , specificity_ ,precision_ ,f1_ ,lr_precision_ ,lr_recall_,auc_ = compute_metrics(y_true, y_pred, y_pred_prob)

            # Save the computed metric
            if phase == 'val' and f1_ >= best_f1:
                # we save the best model based on the f1 metric
                best_f1 = f1_
                perf['lr_precision_best'] = lr_precision_
                perf['lr_recall_best'] = lr_recall_
                perf['best_epoch'] = epoch
                if path_save_model:
                    torch.save(model.state_dict(), path_save_model)
            if phase == 'val':
                perf['loss_val'].append(epoch_loss)
                perf['acc_val'].append(epoch_acc)
                perf['specificity'].append(specificity_)
                perf['precision'].append(precision_)
                perf['recall'].append(recall_)
                perf['f1_score'].append(f1_)
                perf['auc'].append(auc_)
                if keep_pred :
                    perf['y_true'] = y_true
                    perf['y_pred'] = y_pred
            else:
                perf['loss_train'].append(epoch_loss)
                perf['acc_train'].append(epoch_acc)

    time_elapsed = time.time() - since
    perf['time_epoch'] = (time_elapsed // 60) + (time_elapsed % 60)*0.01
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return perf

#*******************************************************
#                   Visualization
#*******************************************************
def visualize_results(perf,path_fig):
    sns.set_theme()
    fig, axs = plt.subplots(2,2,figsize = (10,10))
    axs = axs.ravel()
    epochs =[ i for i in range( 1,len(perf['loss_train'])+ 1) ] 
    axs[0].plot(epochs, perf['loss_train'],label = 'train')
    axs[0].plot(epochs, perf['loss_val'],label = 'val')
    axs[0].set_title('Cross entropy loss')
    axs[0].set_xlabel('nb epochs')
    axs[0].set_ylabel('loss')

    axs[1].plot(epochs, perf['acc_train'],label = 'train')
    axs[1].plot(epochs, perf['acc_val'],label = 'val')
    axs[1].set_title('Accuracy')
    axs[1].set_xlabel('nb epochs')
    axs[1].set_ylabel('accuracy')

    metric = ['specificity', 'precision', 'recall','f1_score','auc']
    for m in metric:
        axs[2].plot(epochs, perf[m],label = m)
    axs[2].set_title('Different metrics on validation set')
    axs[2].set_xlabel('nb epochs')

    axs[3].plot(perf['lr_recall_best'],perf['lr_precision_best'])
    axs[3].set_title('Precision and recal curve')
    axs[3].set_ylabel('precision')
    axs[3].set_xlabel('recall')
    for i in range(3):
        axs[i].legend()
    plt.tight_layout()
    fig.savefig(path_fig)

def visualize_results_runs(name_files,results_dir,path_fig):
    sns.set_theme()
    fig, axs = plt.subplots(2,5,figsize = (20,8),sharex=True)
    fig.suptitle('Patient CV-5 summaries ',fontsize=16)
    axs = axs.ravel()
    colors = ['b','r','g','orange']
    title = ['Loss training', 'Loss validation','Accuracy training','Accuracy validation',
                'F1 score','Specificity', 'Precision', 'Recall','AUC precision recall curve']
    key = ['loss_train','loss_val','acc_train','acc_val','f1_score','specificity', 'precision', 'recall','auc']
    
    for i ,(k , v) in enumerate(name_files.items()):
        # accumulate 
        perf_avg,perf_var = accumulate_results(v,results_dir)
        epochs =[ i for i in range( 1,len(perf_avg['loss_train'])+ 1) ] 
        for j,m in enumerate(key):
            axs[j].plot(epochs, perf_avg[m], color=colors[i] ,label = k )
            plot_confience(epochs,perf_avg,perf_var,m,colors[i],k,axs[j],los_val = False)
            axs[j].legend()
            axs[j].set_xlabel('nb epochs')
            axs[j].set_title(title[j])

    plt.tight_layout()
    fig.savefig(path_fig)

def visualize_results_runs_2(name_files,results_dir,path_fig,tilte_plot):
    sns.set_theme()
    fig, axs = plt.subplots(2,4,figsize = (18,8),sharex=True)
    fig.suptitle(tilte_plot,fontsize=18)
    axs = axs.ravel()
    colors = ['b','r','g','orange']
    title = ['Loss','Accuracy training','Accuracy validation',
                'F1 score','Specificity', 'Precision', 'Recall','AUC precision recall curve']
    key = ['loss_train','acc_train','acc_val','f1_score','specificity', 'precision', 'recall','auc']
    
    for i ,(k , v) in enumerate(name_files.items()):
        # accumulate 
        perf_avg,perf_var = accumulate_results(v,results_dir)
        epochs =[ i for i in range( 1,len(perf_avg['loss_train'])+ 1) ] 
        for j,m in enumerate(key):
            plot_(epochs,perf_avg ,m, colors[i],k,axs[j])
            plot_confience(epochs,perf_avg,perf_var,m,colors[i],k,axs[j],los_val = True)
            axs[j].legend()
            axs[j].set_xlabel('nb epochs')
            axs[j].set_title(title[j])

    fig.tight_layout()
    fig.savefig(path_fig)

def plot_(x,perf_avg ,key, color ,label , ax): 
    if key == 'loss_train' :
        ax.plot(x, perf_avg[key], color=color ,label = label + ' train')
        ax.plot(x, perf_avg['loss_val'],'--', color=color ,label = label +' test')
        ax.set_yscale('log')
    else : 
        ax.plot(x, perf_avg[key], color=color ,label = label)

def plot_confience(x,perf_avg,perf_var,key,color,label,ax,los_val= False):
    ax.fill_between(x, perf_avg[key] -perf_var[key] , perf_avg[key] + perf_var[key], color=color, alpha=.1)
    if key == 'loss_train' and los_val :
        ax.fill_between(x, perf_avg['loss_val'] -perf_var['loss_val'] , perf_avg['loss_val'] + perf_var['loss_val'], color=color, alpha=.1)

def accumulate_results(name_files,results_dir):
    if results_dir == None: 
        with open( name_files[0], "rb") as fp:
            perf_acc = pickle.load(fp)
    else : 
        with open(os.path.join(results_dir, name_files[0] ), "rb") as fp:
            perf_acc = pickle.load(fp)
    print(perf_acc['time_epoch'])
    perf_acc.pop('best_epoch', None)
    perf_acc.pop('time_epoch', None)
    perf_acc.pop('lr_precision_best', None)
    perf_acc.pop('lr_recall_best', None)
    perf_acc.pop('y_true',None)
    perf_acc.pop('y_pred',None)
    for i in range(1,len(name_files)):
        if results_dir == None: 
            with open( name_files[i], "rb") as fp:
                perf_tmp = pickle.load(fp)
        else : 
            with open(os.path.join(results_dir, name_files[i] ), "rb") as fp:
                perf_tmp = pickle.load(fp)
        
        for metric in list(perf_acc.keys()) :
            perf_acc[metric] = np.c_[perf_acc[metric],perf_tmp[metric][:30]]
    
    perf_var = {}
    perf_avg = {}
    for metric in list(perf_acc.keys()):
        perf_avg[metric] = np.mean(perf_acc[metric],axis=1)
        perf_var[metric] = 1.96*np.std(perf_acc[metric],axis=1)
    
    return perf_avg,perf_var


            
