#!/bin/bash
# command to run model with the custom attention as an input channel 
python main.py --typerun baseline --model resnet18 --path_annot annotated_box_MI_cv
python main.py --typerun weighted_dataloader --model resnet18 --path_annot annotated_box_MI_cv
python main.py --typerun weighted_loss --model resnet18 --path_annot annotated_box_MI_cv
python main.py --typerun augmented_culprit --model resnet18 --path_annot annotated_box_MI_cv

# command to run models with the custom attention apply to the images 
python main.py --typerun baseline --model resnet18_apply --path_annot annotated_box_MI_cv --input_channel 1 --apply_filter True
python main.py --typerun weighted_dataloader --model resnet18_apply --path_annot annotated_box_MI_cv --input_channel 1 --apply_filter True
python main.py --typerun weighted_loss --model resnet18_apply --path_annot annotated_box_MI_cv --input_channel 1 --apply_filter True
python main.py --typerun augmented_culprit --model resnet18_apply --path_annot annotated_box_MI_cv --input_channel 1 --apply_filter True

# command to run the hyper parameter tunning 
python main.py --typerun tuning_balance_channel --model resnet18 --path_annot annotated_box_MI_cv
python main.py --typerun tuning_balance_apply --model resnet18 --path_annot annotated_box_MI_cv

# command to test the model 
python main_test.py --typerun test_balance_channel --model resnet18 