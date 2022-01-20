import torchvision.models as modelsT
import torch.nn as nn
import torch
import os 

"""
This file contains the models definition 
"""
PATH_BEST_MODEL = '/data/cardio/SPUM/CVD_detection_code/cnn/model_pretrained_latest1.pth'

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


class Siamese(nn.Module):
    def __init__(self, in_channel = 2 , num_classes=2,dropout = False,pretrained_model= False,load_best_model=False ):
        super(Siamese, self).__init__()

        if pretrained_model :
            print('No pretrained model')
            self.backbone = modelsT.resnet18(pretrained=False)
        else : 
            print('Load Imgnet pretrained model')
            self.backbone = modelsT.resnet18(pretrained=True)

        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        self.dim_out = self.backbone.fc.in_features
        self.classifier = nn.Linear(self.dim_out*2,num_classes)

        if load_best_model : 
            print('Load best pretrained model')
            state_dict = torch.load(PATH_BEST_MODEL)
            state_dict['fc.weight'] = state_dict.pop('fc.1.weight')
            state_dict['fc.bias'] =state_dict.pop('fc.1.bias')
            self.backbone.load_state_dict(state_dict)

        if dropout:
            self.backbone.fc = nn.Sequential(nn.Dropout(0.5), nn.Identity())
            print('dropout')
        else:
            self.backbone.fc = nn.Identity()

    def forward(self,x):
        #channel 1
        x1 = torch.cat((x[:,:1],x[:,:1],x[:,:1]),dim=1)
        #channel 2
        x2 = torch.cat((x[:,1:],x[:,1:],x[:,1:]),dim=1)

        out1 = self.backbone(x1)
        out2 = self.backbone(x2)
        emb = torch.cat((out1,out2),dim = 1)
        out = self.classifier(emb)
        return out 
         

def initialize_model(modelName, num_classes,feature_extract, num_inputs_chanel=2, pretrained_model = True, 
                        load_best_model=False , dropout = False):
    ''' List of different models with their respective
        initializations. Here we modify the final output layer to 
        2 classes. If you want to use the model as a feature extractor set feature_extract to True ,
        If you want the best model parameter from previous year try load_best_model to True '''
    model = None

    if pretrained_model == load_best_model:
            print("Warrning pretrained_model and load_best_model are equal :(")

    if modelName == 'dense121':
        model = modelsT.densenet121(pretrained=pretrained_model)
        set_parameter_requires_grad(model, feature_extract)
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, num_classes)

    elif modelName == 'dense161':
        model = modelsT.densenet161(pretrained=pretrained_model)
        set_parameter_requires_grad(model, feature_extract)
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, num_classes)
  
    elif modelName == 'dense201':
        model = modelsT.densenet201(pretrained=pretrained_model)
        set_parameter_requires_grad(model, feature_extract)
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, num_classes)

    elif modelName == 'resnet18':
        model = modelsT.resnet18(pretrained=pretrained_model)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

        if load_best_model: 
            state_dict = torch.load(PATH_BEST_MODEL)
            state_dict['fc.weight'] = state_dict.pop('fc.1.weight')
            state_dict['fc.bias'] =state_dict.pop('fc.1.bias')
            model.load_state_dict(state_dict)
    
        set_parameter_requires_grad(model, feature_extract)
        # modify the input if not 3
        if num_inputs_chanel == 2 :

            model.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3,
                                    bias=False)
            model.bn1 = nn.BatchNorm2d(64)
            input_l = [ model.conv1 ,model.bn1 ]
            for layer in input_l:
                for param in layer.parameters():
                    param.requires_grad = True

        if dropout:
            model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(model.fc.in_features, num_classes))
            #print(dropout)
        else:
            model.fc = nn.Linear(model.fc.in_features, num_classes)


    elif modelName == 'resnet101':
        model =modelsT.resnet101(pretrained=pretrained_model)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    elif modelName == 'resnet152':
        model = modelsT.resnet152(pretrained=pretrained_model)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    elif modelName == 'vgg11_bn':
        model = modelsT.vgg11_bn(pretrained=pretrained_model)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs,num_classes)
  
    elif modelName == 'vgg13_bn':
        model = modelsT.vgg13_bn(pretrained=pretrained_model)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs,num_classes)

    elif modelName == 'vgg19_bn':
        model = modelsT.vgg19_bn(pretrained=pretrained_model)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs,num_classes)

    else:
        raise NameError('Model name not included in our list.')

    return model


