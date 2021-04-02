# DIN
## Introduction 
This is the implementation of DIN base on [*Deep Interleaved Network for Image Super-Resolution With Asymmetric Co-Attention*](https://arxiv.org/abs/2004.11814) and [*Learning Deep Interleaved Networks with Asymmetric Co-Attention for Image Restoration*](https://arxiv.org/abs/2010.15689). The ```Conference_version``` is for the former paper, which we take for example to edit the following instructions,  and the ```Enhance_version``` is for the later. 
The architecture of our proposed DIN.  
![image](https://github.com/lifengshiwo/DIN/blob/d24fa5fb41de20c3578db619b43fedecaca15cab/Conference_version/figures/2.PNG)
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
1. Download trainning set [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) and [Flickr2K](https://github.com/sanghyun-son/EDSR-PyTorch)
2. Prepare training data. Run ```./scripts/Prepare_TrainData_HR_LR.py``` or ```./scripts/Prepare_TrainDate_HR_LR.m``` to generate HR/LR pairs with corresponding degradation models and scale factor. Modify ```./scripts/flags.py``` to configure ```traindata_path``` and ```savedata_path```.
3. Test data preparation is as same as train data preparation.
4. Configure ```./options/train/train_DIN_x4.json``` for your training.
5. Run the command:
```
CUDA_VISIBLE_DEVICES=0 python train.py -opt options/train/train_DIN_x4.json
```
## Test
1. Prepare testing data. Choose public standard benchmark datasets and run ```./scripts/Prepare_TrainData_HR_LR.py``` or ```./scripts/Prepare_TrainDate_HR_LR.m``` to generate HR/LR pairs with corresponding degradation models and scale factor. Modify ```./scripts/flags.py``` to configure ```traindata_path``` and ```savedata_path```.
2. Configure ```./options/test/test_DIN_x4_BI.json``` for your testing.
3. Run the command and PSNR/SSIM values are printed and you can find the reconstructed images in ```./result```.
```
CUDA_VISIBLE_DEVICES=0 python test.py -opt options/test/test_DIN_x4_BI.json
```
## Results
Here are the visual results.  
![image](https://github.com/lifengshiwo/DIN/blob/d24fa5fb41de20c3578db619b43fedecaca15cab/Conference_version/figures/1.PNG)
## Citation
If you find our work useful in your research or publications, please consider citing:
```
@inproceedings{International Joint Conference on Artificial Intelligence(ijcai),
author = {Li, Feng and Cong, Runmin and Bai, Huihui and He, Yifan},
year = {2020},
month = {07},
pages = {537-543},
title = {Deep Interleaved Network for Single Image Super-Resolution with Asymmetric Co-Attention},
doi = {10.24963/ijcai.2020/75}
}

@article{arXiv:2010.15689,
author = {Li, Feng and Cong, Runmin and Bai, Huihui and He, Yifan and Zhao, Yao and Zhu, Ce},
year = {2020},
month = {10},
pages = {},
title = {Learning Deep Interleaved Networks with Asymmetric Co-Attention for Image Restoration}
}
```
## Acknowledgement
This code is built on [SRFBN](https://github.com/Paper99/SRFBN_CVPR19)(Pytorch), we thank the authors for sharing their code.
