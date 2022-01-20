#!/bin/bash
# If runing a docker container 
# docker: pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime cf pod .yaml
pip install sklearn
pip install matplotlib
pip install seaborn
pip install opencv-python
apt-get update 
apt-get install ffmpeg libsm6 libxext6  -y # cv2 dependencies that are normally present on the local machine