# Infant Action Recognition with DMD

Codes and experiments for the following paper:
Yanjun Zhu and Pooria daneshvar Kakhaki, Agata Lapedriza, Sarah Ostadabbas, “Diffusion-based Motion Denoising for Robust Infant Action Recognition” [NeurIPS 2024 UNDER REVIEW]

Contact:  
**Yanjun Zhu**  
Email: [ya.zhu@northeastern.edu](mailto:ya.zhu@northeastern.edu)  
**Pooria daneshvar Kakhaki**  
Email: [daneshvarkakhaki.p@northeastern.edu](mailto:daneshvarkakhaki.p@northeastern.edu)  
**Sarah Ostadbass**  
Email: [s.ostadabbas@northeastern.edu](mailto:s.ostadabbas@northeastern.edu)



### Table of Contents
1. [Introduction](#introduction)
2. [Environment](#environment)
3. [Data Preparation](#data-preparation)
    1. [Preprocessing Data](#preprocessing-data)
    2. [Data Download](#data-download)
4. [Infant Action Recognition](#infant-action-recognition)
    1. [Training](#training)
    2. [Testing](#testing)
    3. [Visualizing](#visualizing)
5. [Diffusion-based Motion Denoising](#diffusion-based-motion-denoising)

## Introduction

This repository contains the code for the paper "Diffusion-based Motion Denoising for Robust Infant Action Recognition". We propose a novel method for infant action recognition that combines a graph convolutional network (GCN) with a diffusion-based motion denoising (DMD) model. The GCN is used to extract the spatial and temporal features of the infant's skeleton data, while the DMD model is used to denoise the motion data.

![Qualitative samples](figs\qualitative_results.png)

The code is divided into two main parts: Infant Action Recognition and Diffusion-based Motion Denoising. The former is responsible for training and testing the action recognition model, while the latter is responsible for training and testing the diffusion-based motion denoising model.

## Environment

Prepare the virtual environment:

You can use the provided YAML file to create the environemt:
```shell
conda env create -f environment.yml
conda activate infant_denoising
```

Otherwise, you will need to ensure that you have the following packages:
```shell
pytorch
torchvision
tqdm
matplotlib
scikit-learn
seaborn
numpy
pandas
```

This code base is tested with python=3.11.0 and PyTorch==2.3.0

## Data preparation

### Preprocessing Data
For object detection and multi-person tracking, please refer to [Ultralytics YOLO Docs](https://docs.ultralytics.com/modes/track/ "Multi-Object Tracking with Ultralytics YOLO"). For our implementation, we used "Yolov8n-pose" and "BOT-SoRT" tracker.

For infant-specific pose estimation, please refer to [Fine-tuned Domain-adapted Infant Pose (FiDIP)](https://github.com/ostadabbas/Infant-Pose-Estimation "Github Repository of FiDiP")

### Data Download
<a name="INFANTS"></a>
We have also added the processed skeletons of the INFANTS dataset directly for download.
Preprocessed  INFANTS skeleton data can be downloaded 
[here](https://drive.google.com/file/d/10z5dbOXk76nOhmeLpYDnNtnOT1xYGvkc/view?usp=sharing)<br/>
Put downloaded data into the following directory structure:

```
- Data/
  - INFANTS/
    -INFANTS.pkl
```


## Validation of Data Processing

In this session, we used two datasets to verify the effectiveness of our data processing pipeline:

- **SyRIP Test100**  
- **InfAct+Unseen**  

Please download the data from [GoogleDrive](https://drive.google.com/drive/u/1/folders/1jG8kK8ZqZttuscyt1HhM1bR-pqF4EwDc). The raw data are stored in the 'custom_data' folder in **COCO format**. Additional folders contain:

- **2D pose predictions:** [FiDIP](https://github.com/ostadabbas/Infant-Pose-Estimation) and [YOLOv8-Pose](https://github.com/autogyro/yolo-V8)  
- **3D pose predictions:** [HW-HuP](https://github.com/ostadabbas/HW-HuP)  
- **Posture results:** [Posture classifier](https://github.com/ostadabbas/Infant-Posture-Estimation)  

For more details, please refer to the original paper.

### Evaluation Results
1. **2D Pose Estimation Comparison:** FiDIP vs. YOLOv8-Pose for infant 2D pose estimation on frames from InfAct+Unseen and SyRIP Test100 datasets.

   ![Pose estimation](figs\pose_comp.png)



2. **Posture Classification Accuracy Comparison (%):** Using different input data sources:  

   - 2D pose from YOLOv8-Pose  
   - 2D pose from FiDIP  
   - 2D pose ground truth  
   - 3D pose from HW-HuP  
   
   ![Posture Classification](figs\posture_comp.png)



## Infant action recognition

### Training
To train the action recognition model, the train script must be executed
assuming the code root as ``${REC_ROOT}``, please navigate into the InfantGCN directory by
```shell
cd ${REC_ROOT}/InfantGCN
```
To start the training, use the following script:

```shell
python train.py [--epochs] [--model] [--base_lr] [--repeat] [--output_folder] [--exp_name]
```

#### Arguments

- `model`: The model which is used as the backbone of the recognition model. Please use either `CTRGCN` or `STCGN`
- `base_lr`: The learning rate of the optimizer. Default value is `0.1`
- `repeat`: Number of times that the training dataset is repeated to increase the size of the dataset
- `output_folder`: The parent folder where the results will be saved
- `exp_name`: The folder where the results will be saved

#### Example:

```shell
python train.py --epochs 15 --model CTRGCN --base_lr 0.1 --repeat 1 --output_folder ../Results --exp_name CTRGCN_REC
```

The results of the following experiments will be saved in ``${REC_ROOT}/Results/CTRGCN_REC``<br/>
In each experiments, weights after each training epoch will be save as ``$epoch_{epoch_num}.pth``<br/>
The weights which yields the best validation accuracy will be saved as ``best_results.pth``

### Testing
To test the action recognition model, the test script must be executed
assuming the code root as ``${REC_ROOT}``, please navigate into the InfantGCN directory by
```shell
cd ${REC_ROOT}/InfantGCN
```
To start the inference, use the following script:

```shell
python test.py [--model] [--weights] [--output_folder] [--exp_name]
```

#### Arguments

- `model`: The model which is used as the backbone of the recognition model. Please use either `CTRGCN` or `STCGN`
- `weights`: The weights which will be loaded into the model
- `output_folder`: The parent folder where the results have been previously be saved
- `exp_name`: The folder where the results have been previously be saved

#### Example:

```shell
python test.py --model CTRGCN --weights ../Results/CTRGCN_REC/best_results.pth --output_folder ../Results --exp_name CTRGCN_REC
```

The results of the test will be save as ``${REC_ROOT}/Results/CTRGCN_REC/eval.pkl``<br/>
This pickle file contains the predicted and ground truth labels for each samples in the test dataset, and the accuracy accross the dataset

### Visualizing
To visualize the action recognition model, the test script must be executed
assuming the code root as ``${REC_ROOT}``, please navigate into the InfantGCN directory by
```shell
cd ${REC_ROOT}/InfantGCN
```
To start the visualization, use the following script:

```shell
python visualize.py [--eval_file]
```

#### Arguments

- `eval_file`: the result of the execution of a previous test command saved as a pickle file

#### Example:

```shell
python visualize.py --eval_file ../Results/CTRGCN_REC/eval.pkl
```

The visualized confusion matrix will be save as  ``${REC_ROOT}/Results/CTRGCN_REC/cm.png``<br/>

## Diffusion-based motion Denoising