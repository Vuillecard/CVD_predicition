# Myocardial infarction prediction from angiographies based on deep learning
Master thesis, Pierre Vuillecard

## Summary : 
Stenosis is a coronary artery disease that can lead to heart attack or myocardial infarction (MI). It is one of the main cause of death in the past two decades. Predicting whether a stenosis will lead to a heart attack or not is crucial for modern society. The signal of the vessel surrounding the stenosis might give valuable information on the stenosis condition. Computer-aided diagnosis systems gained in popularity since they could help and assist experts in critical decision-making. Previous researches developed several methods to detect significant stenosis in X-ray coronary angiography image. They demonstrated excellent performances based on classification from extracted vessels patches or based on detection from full X-ray images, via object detection models. In this thesis, a special effort is put to evaluate the prediction of future MI based on extracted vessel segment from X-ray coronary angiograms. Doctors have annotated angiography images for this work. It brings many challenges that need to be addressed as the limited amount of data, the class imbalance, the annotation specificities, and the qualities of the image. In order to understand if there is some predictive signal in the angiograms, we approach the problem from different angles. First, we predict MI from segment extracted from the annotated angiograms. Then, we try to simultaneously detect the segment and predict if it will lead to an MI. Next, we try to classify segments but taking into account not only the segment information but the entire image, with an attention map on the segment level. Finally, we simplify the problem by exploiting the exact position of the stenosis on the vessel, and predict MI from patches extracted around the stenosis. The different techniques demonstrated poor predictions of vessels leading to a MI. Based on a clinical study of 469 patients, out of which only 58 had MI, we present evidence that the signal of vessels responsible for heart attacks might not be sufficient to make a clear distinction between a severe and benign stenosis inside the vessels.

## Code Structure: 
The code used in this project is structured as follow: 
- */vessel_and_stenosis*: contains the code for the MI patches classification and the MI stenosis patches classification
- */custom_attention*: contains the code for the MI custom attention vessel segment prediction.
- */vessel_detection*: contains the code for the MI vessel segment detection using an object detection framework

## Installation: 
There two main installation one for the */vessel_and_stenosis* and */custom_attention* code and one for the */vessel_detection*. We used docker that ensures a good code reproducibility.
###### For */vessel_and_stenosis* and */custom_attention* :
- docker image : pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime
- run */vessel_and_stenosis/installation.sh*
    ```sh
    cd vessel_and_stenosis
    bash installation.sh
    ```

###### For */vessel_detection* :
- docker image : pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-devel
- install apex:
    ```sh
    cd mkdir apexonly
    cd apexonly
    git clone https://github.com/NVIDIA/apex
    cd apex
    git checkout f3a960f80244cf9e80558ab30f7f7e8cbf03c0a0
    python setup.py install --cuda_ext --cpp_ext
    ```
- install cython and matplotlib : pip install cython matplotlib
- install cocoapi:
    ```sh
    git clone https://github.com/cocodataset/cocoapi.git
    cd cocoapi/PythonAPI
    python setup.py build_ext install
    ```
- install rotated detectron: 
    ```sh
    cd vessel_detection
    python setup.py build develop
    ```
- upgrade the opencv package
    ```sh
    pip upgrade 
    pip install opencv-python==4.2.0.32
    apt-get install libglib2.0-0
    apt install -y libsm6 libxext6
    apt-get install -y libxrender-dev
    ```
    read */vessel_detection* README for further information.

## Files: 
###### */vessel_and_stenosis* and */custom_attention* : 
- */utiles.py*: handle data creation and algorithm to extract the information from the annotated images.
- */dataset.py*: handle the dataset, and preprocessing for the model.
- */implementation.py*: handle the training of the model and results visualization. 
- */model_init.py*: Initialize the models.
- */genereate_data.py*: Generate the annotation and images from the annotated images.
- */main_<name>.py*: code to run the experiment.
- */job_<name>.py*: run the experiment.

###### */custom_attention* :
- */utiles.py*: handle data creation and algorithm to extract the information from the annotated images.
- */gaussian_attention.py*: Create a gaussian attention 
- */dataset.py*: handle the dataset, and preprocessing for the model.
- */implementation.py*: handle the training of the model and results visualization. 
- */model_init.py*: Initialize the models.
- */genereate_data.py*: Generate the annotation and images from the annotated images.
- */main_<name>.py*: code to run the experiment.
- */job_<name>.py*: run the experiment.

###### */vessel_detection* : 
- The strucutured of the code could be find in the readme.
- The model configuration could be find in */configs/rotated/< name with cvd >.yaml* like the FPN architecture.




