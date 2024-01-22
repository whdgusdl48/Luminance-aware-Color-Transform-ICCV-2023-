# LACT : Luminance-aware Color Transform for Multiple Exposure Correction

## Introduction
This repo is the official Code of  Luminance-aware Color Transform for Multiple Exposure Correction (**ICCV 2023**). 

Supplementary paper will be update!

## Qualitative Result on ME Dataset
![sample](figures/Result_image.png)

## Qualitative Result on SICE Dataset
![sample2](figures/Result_image2.png)

## Installation
We recommend you to use an Anaconda virtual environment and memory of GPU RAM 24G.
### Requirements
- Linux or macOS with Python ≥ 3.6
- tensorflow ≥ 2.5 
- OpenCV is optional but needed by demo and visualization
- scikit-learn is need to save image file

### Example conda environment setup
```bash
conda create --name lact python=3.8 -y
conda activate lact
conda install tensorflow==2.5.0
pip install -U opencv-python

# Prepare Datasets for LACT

$DATASETS/
  multi_exposure/
  SICE/
```

## Expected dataset structure for [multi_exposure](https://github.com/mahmoudnafifi/Exposure_Correction):

```
multi_exposure/
  testing/
  traiing/
  multi_adobe5k-train.json
  multi_adobe5k-test.json
```

## Expected dataset structure for SICE:
```
cityscapes/
  sice_middle/
  sice_middle_label/
  sice_middle_train.json
  sice_middle_test.json
```
## Getting Started with LACT

### Develop Environment
OS: Ubuntu 18.04

GPU: Nvidia RTX A6000

### Train
```python -m train.py --dataset dataset_path --train_db dataset_path/train.json```</br>
```                   --test_db dataset_path/test.json --batch_size = 2```</br>
```                   --epochs 20 --learning_rate = 3e-4```</br>
```                   --warmup_steps 8765 decay_steps = 40000 --log_dir result/```</br>

### Structure of Project Folder
```
$ tree

├─datasets
├─libs
│  ├─Metric.py
│  ├─model.py
│  ├─multi_datasets.py
│  ├─util.py
├─logs
└─train.py
```

### Result
We achieved the state-of-the art on ME Dastset and SICE Dataset.
![sample](figures/Result2.png)


### Model Output
You can access the results for the dataset via the link below.

Keep in mind that there are five exposure levels for a sequence: 0, N1.5, N1, P1.5, and P1.

ME Dataset : https://drive.google.com/file/d/13yprihRtu4d6Leb2nPUGvuW1qz671q5X/view?usp=sharing

### bibtex
```
@InProceedings{Baek_2023_ICCV,
    author    = {Baek, Jong-Hyeon and Kim, DaeHyun and Choi, Su-Min and Lee, Hyo-jun and Kim, Hanul and Koh, Yeong Jun},
    title     = {Luminance-aware Color Transform for Multiple Exposure Correction},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {6156-6165}
}
```

