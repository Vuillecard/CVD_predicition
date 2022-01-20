#!/bin/bash
# Run hyperparameter tunning for the patches classification
python main_hyperparameter_tunning.py --typerun tuning_augment --model resnet18 --path_annot annotated_box_MI_CV_aug

# Run hyperparameter tunning for the stenosis 
python main_stenosis.py --typerun tuning_base_stenosis_2 --model resnet18 --path_annot annotated_box_cv
python main_stenosis.py --typerun tuning_base_stenosis_2_scratch --model resnet18 --path_annot annotated_box_cv
python main_stenosis.py --typerun tuning_base_stenosis_2_best --model resnet18 --path_annot annotated_box_cv
python main_stenosis.py --typerun tuning_aug_stenosis --model resnet18 --path_annot annotated_box_cv
python main_stenosis.py --typerun tuning_aug_stenosis_scratch --model resnet18 --path_annot annotated_box_cv
python main_stenosis.py --typerun tuning_aug_stenosis_best --model resnet18 --path_annot annotated_box_cv