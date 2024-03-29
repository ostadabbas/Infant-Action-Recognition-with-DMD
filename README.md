# Infant Action Recognition

## Environment

Prepare the virtual environment:
#SUBJECT TO CHANGE
```shell
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
conda install pytorch torchvision -c pytorch  # This command will automatically install the latest version PyTorch and cudatoolkit, please check whether they match your environment.
pip install -U openmim
mim install mmengine
mim install mmcv
mim install mmdet  # optional
mim install mmpose  # optional
cd mmaction2
pip install -v -e .
```
### Data preparation


<a name="InfActPrimitive"></a>
Preprocessed  infant 2D and 3D skeleton data can be downloaded 
[here](https://drive.google.com/file/d/10z5dbOXk76nOhmeLpYDnNtnOT1xYGvkc/view?usp=sharing)<br/>
Put downloaded data into the following directory structure:

```
- Data/
  - InfAct_plus/
    -InfAct_plus_2d.pkl
    -InfAct_plus_3d.pkl
```

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
To start the training, use the following script:

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
To start the training, use the following script:

```shell
python visualize.py [--eval_file]
```

#### Arguments

- `model`: the result of the execution of a previous test command

#### Example:

```shell
python visualize.py --eval_file ../Results/CTRGCN_REC/eval.pkl
```

The visualized confusion matrix will be save as  ``${REC_ROOT}/Results/CTRGCN_REC/cm.png``<br/>




