# Infant Action Recognition

## InfActPrimitive Dataset
<a name="InfActPrimitive"></a>
Preprocessed  infant 2D and 3D skeleton data can be downloaded 
[here](https://drive.google.com/file/d/1TiuTul5b5XtJgKZeOCnrAH8WKmxb6Rld/view?usp=sharing)


## Environment

Prepare the virtual environment:

```shell
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
conda install pytorch torchvision -c pytorch  # This command will automatically install the latest version PyTorch and cudatoolkit, please check whether they match your environment.
pip install -U openmim
mim install mmengine
mim install mmcv
mim install mmdet  # optional
mim install mmpose  # optional
git clone https://github.com/open-mmlab/mmaction2.git
cd mmaction2
pip install -v -e .
```
### Data preparation

Add the infact dataset to:

\Data\InfAct_plus\2d\primitive

Put downloaded data into the following directory structure:

```
- Data/
  - InfAct_plus/
    - 2d/
      -primitive/
        -InfAct_plus.pkl
```

### Training
To train the model on InfAct

```
"""mim train mmaction configs/ctrgcn_infact_plus_2d_primitive.py 
--work-dir ../../../Results/ctrgcn_infact_plus_2d_primitive"""
```

### Inference


```
mim test mmaction 
configs/ctrgcn_infact_plus_2d_primitive.py 
--checkpoint ../../../Results/ctrgcn_infact_plus_2d_primitive/best_acc_top1_epoch_{}.pth 
--dump ../../../Results/ctrgcn_infact_plus_2d_primitive/eval.pkl
```




