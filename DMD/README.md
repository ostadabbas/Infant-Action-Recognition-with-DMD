# DMD for 3D pose
This a an implementation of DMD for 3D motion denoising on Human 3.6M dataset. After obtaining 3D infant poses, we need to transform them to the same format of Human 3.6M.


## Installation


### 1. Environment

<details> 
<summary>Python/conda environment</summary>
<p>

```
pytorch
einops
zarr
pandas
scipy
numpy
```
</p>
</details> 


### 2. Datasets

#### [**> Human3.6M**](http://vision.imar.ro/human3.6m/description.php)

We follow https://github.com/wei-mao-2019/gsps for Human3.6M dataset preparation. 
All data needed can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1sb1n9l0Na5EqtapDVShOJJ-v6o-GZrIJ?usp=sharing) and place all the dataset in ``data`` folder inside the root of this repo.

### **Infact+**


## Training

Human3.6M:
```
python train.py --cfg h36m
```

Infact+:
```
python train.py --cfg infact
```

## Evaluation

Human3.6M:
```
python train.py --cfg h36m --test
```

Infact+:
```
python test.py --cfg infact --test
```
