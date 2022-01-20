#!/bin/bash
# Run ResNet18 model
python main_vessels.py --typerun baseline --model resnet18 --path_annot annotated_box_MI_CV
python main_vessels.py --typerun weighted_dataloader --model resnet18 --path_annot annotated_box_MI_CV
python main_vessels.py --typerun weighted_loss --model resnet18 --path_annot annotated_box_MI_CV
python main_vessels.py --typerun augmented_culprit --model resnet18 --path_annot annotated_box_MI_CV

# Run Siamese model
python main_vessels.py --typerun baseline --model siamese --path_annot annotated_box_MI_CV
python main_vessels.py --typerun weighted_dataloader --model siamese --path_annot annotated_box_MI_CV
python main_vessels.py --typerun weighted_loss --model siamese --path_annot annotated_box_MI_CV
python main_vessels.py --typerun augmented_culprit --model siamese --path_annot annotated_box_MI_CV

# test to run Adam optimizer
python main_vessels.py --typerun test_augmented_culprit --model resnet18 --path_annot annotated_box_MI

# Run on stenosis data 
python main_stenosis.py --typerun baseline --model resnet18 --path_annot annotated_box
python main_stenosis.py --typerun augmentation --model resnet18 --path_annot annotated_box
python main_stenosis.py --typerun aug_reg --model resnet18 --path_annot annotated_box