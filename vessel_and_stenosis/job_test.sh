#!/bin/bash
# Run test for the patches classification 
python main_vessels_test.py --typerun testing --model resnet18
python main_vessels_test.py --typerun testing_dataloader --model resnet18

# Run test for the setnosis classifiaction
python main_stenosis_test.py --typerun stenosis_baseline_test --model resnet18
python main_stenosis_test.py --typerun stenosis_augmentation_test --model resnet18