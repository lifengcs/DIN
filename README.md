# DIN
## Introduction 
This is the implementation of DIN base on [*Deep Interleaved Network for Image Super-Resolution With Asymmetric Co-Attention*](https://arxiv.org/abs/2004.11814) and [*Learning Deep Interleaved Networks with Asymmetric Co-Attention for Image Restoration*](https://arxiv.org/abs/2010.15689). The ```Conference_version``` is for the former which we take for example to illustrate and the ```Enhance_version``` is for the later. 
The architecture of our proposed DIN.
## Environment
+ Python3
+ pytorch
## Installations
+ skimage
+ imageio
+ tqdm
+ pandas
+ numpy
+ opencv-python
+ Matlab
## Train
1. Download trainning set [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/).
2. Prepare training data. Run ```./scripts/Prepare_TrainData_HR_LR.py``` or ```./scripts/Prepare_TrainDate_HR_LR.m``` to generate HR/LR pairs with corresponding degradation models and scale factor. Modify ```./scripts/flags.py``` to configure ```traindata_path``` and ```savedata_path```.
3. Test data preparation is as same as train data preparation.
4. Configure ```./options/train/train_DIN_x2.json``` for your training.
5. Run the command
```
CUDA_VISIBLE_DEVICES=0 python train.py -opt options/train/train_DIN_x2.json
```
## Test
1. Prepare testing data. Choose public standard benchmark datasets and run ```./scripts/Prepare_TrainData_HR_LR.py``` or ```./scripts/Prepare_TrainDate_HR_LR.m``` to generate HR/LR pairs with corresponding degradation models and scale factor. Modify ```./scripts/flags.py``` to configure ```traindata_path``` and ```savedata_path```.
2. 
