### Spatial-Frequency Domain Collaboration for Efficient and Detail-Preserving Infrared Image Super-Resolution
---
<p align="center">
  <img width="800" src="./figs/framework.jpg">
</p>

*Overall architecture of the proposed DDFHN. The network consists of three components: a shallow feature extraction module, a series of dual domain feature extraction blocks (DDFEBs), and a lightweight image reconstruction module. Each DDFEB integrates a spatial-frequency feature mixer unit (SFFMU) and a hybrid feature feed-forward unit (HFFU) to progressively extract and refine representative features across both spatial and frequency domains. The baseline DDFHN consists of 8 DDFEB blocks with 36 feature channels, while its larger variant DDFHN+ is equipped with 12 DDFEB blocks and 48 feature channels.
---
### Requirements
> - Python 3.8, PyTorch >= 2.3.0
> - BasicSR 1.4.2
> - GPU: NVIDIA RTX 3090
> - Platforms: Ubuntu 18.04, cuda-11.8 or higher

### Installation
```
# Clone the repo
git clone https://github.com/smilenorth1/DDFHN-main
# Install dependent packages
cd DDFHN-main
pip install -r requirements.txt
# Install BasicSR
python setup.py develop
```
You can also refer to this [INSTALL.md](https://github.com/XPixelGroup/BasicSR/blob/master/docs/INSTALL.md) for installation

## Dataset Information
### Training Dataset
- **CVC‑09‑1K**: Derived from CVC‑09 FIR Sequence Pedestrian Dataset, containing 1000 infrared images with resolution 640×480 for network training. 
  - Reference: Kyaw, H., & Hogg, D. (2016). Adapting pedestrian detectors to new domains: A comprehensive review. *Engineering Applications of Artificial Intelligence*.
  - The training daraset can be obtained from https://opendatalab.org.cn/OpenDataLab/CVC-09/tree/main.

### Testing Datasets
We evaluate model performance on five public infrared benchmark datasets: resultsA, resultsC, M3FD, IR700, FLIR ADAS (2022).
1. resultsA & resultsC: Public infrared test sets, see Liu et al.(2018), Bai et al.(2024).
2. M3FD: Multi‑modal infrared‑visible detection benchmark, Liu et al.(2022).
3. IR700: Infrared super‑resolution test dataset, Zou et al.(2021).

### Data Preprocessing
1. Generate low-resolution (LR) training samples by bicubic down-sampling from original high-resolution (HR) infrared images.
2. Randomly crop 64×64 image patches from LR-HR image pairs during training.
3. Data augmentation: random horizontal flip, random rotation.

## Code Information
This repository contains the official PyTorch implementation of FCMANet.
- Framework: Built on BasicSR 1.4.2 open‑source image restoration toolbox.
- Core modules implementation: DDFEB, SFFMU, HFFU, FFT‑based frequency loss function.
- Included files: training configuration files, test configuration files, network definition scripts, dataset preprocessing & reproduction scripts.
- Pretrained weights for ×2 and ×4 infrared super‑resolution are provided.

## Usage Instructions

### 1. Prepare Datasets
1. Download training dataset CVC-09 and put it under `./datasets/train/`.
2. Download test datasets and place them under `./datasets/test/`.
3. Modify dataset path parameters in corresponding `.yml` config files under `options/train/` and `options/test/`.

### 2. Model Weights
The pretrained model weights results are stored under `./experiments/pretrained_models/`.

### Calculate FLOPs
Run the following commands:
```
python basicsr/archs/ddfhn_arch.py
```

### Runtime and GPU memory consumption
1. Download test datasets and place them under `./inference/datasets/test/`.
2. Download pretrained model weights place them under `./inference/pretrained_model/`.
Run the following commands:
```
python inference/inference_ddfhn.py
```
- Output results will be saved in `./inference/results/` directory.

### Training
Run the following commands for training:
```
# train DDFHN for x2 effieicnt InfraredSR
python basicsr/train.py -opt options/train/train_DDFHN_x2.yml
# train DDFHN for x4 effieicnt InfraredSR
python basicsr/train.py -opt options/train/train_DDFHN_x4.yml
```
### Testing
- Run the following commands:
```
# test DDFHN for x2 efficient SR
python basicsr/test.py -opt options/test/test_DDFHN_x2.yml
# test DDFHN for x4 efficient SR
python basicsr/test.py -opt options/test/test_DDFHN_x4.yml
```
- Output reconstruction results will be saved in `./results/` directory.

## Citation
If you find this work useful for your research, please cite our paper:
```
@article{DDFHN,
  title={Spatial-Frequency Domain Collaboration for Efficient and Detail-Preserving Infrared Image Super-Resolution},
  author={Zepeng Liu, Zhizhong Fu},
  journal={Scientific Reports},
  year={2026}
}
 ```
Please also cite related datasets and BasicSR toolbox as mentioned in the manuscript.

## License
This project is open‑source under **MIT License**.
> See the `LICENSE` file for full license terms.

### Contact
If you have any questions, please feel free to reach me out at 202311012318@std.uestc.edu.cn

### Acknowledgement
This code is based on [BasicSR](https://github.com/XPixelGroup/BasicSR) toolbox. Thanks for the awesome work.



