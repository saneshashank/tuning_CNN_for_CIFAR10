# Tuning CNN for CIFAR10 with Depthwise & Dilated Convolutions

This repository demonstrates the systematic tuning of a Convolutional Neural Network (CNN) for the CIFAR10 image classification dataset, leveraging **depthwise separable convolutions** and **dilated convolutions** to optimize both accuracy and computational efficiency.

## Overview

- **Goal**: Achieve high CIFAR10 accuracy using lightweight CNN design principles.
- **Core Features**: Depthwise-separable convolutions, dilation for receptive field expansion, modern data augmentation, robust training schedule.

## Features

- **Custom PyTorch CNN** (`tiny_cifar10_net.py`)
    - Depthwise separable layers throughout
    - Dilated convolutions (dilation rates 2 & 3) in critical blocks
    - ~36K trainable parameters
    - No maxpooling, downsampling by stride

- **Data Preparation**
    - Computes and uses exact CIFAR10 mean & std for normalization
    - Uses both TorchVision transforms and **Albumentations** (Flip, ShiftScaleRotate, CoarseDropout/CutOut)
    - Custom augmentation pipeline for robust training

- **Training Procedure**
    - SGD optimizer: `lr=0.045`, momentum=0.9, weight_decay=5e-4, nesterov=True
    - Dynamic LR: ReduceLROnPlateau scheduler
    - Early stopping with patience
    - Epoch-level tracking of accuracy/loss

- **Results**
    - Achieves **87.31% test accuracy** (val loss: 0.3716) after 50 epochs on CIFAR10[1]
    - Detailed logs, training curves, and architecture summary included


## Usage

- **Training & Experiments**: Run the Jupyter notebook `tuning_CNN_CIFAR10.ipynb` for the full workflow: loading data, computing stats, model training, and result visualization.
- **Model Definition**: Import and use `NetCIFAR10_Tiny` from `tiny_cifar10_net.py` for your own pipelines.

## Architecture Details

- **Model Blocks:**
    - **C1:** Standard convs with BN+ReLU
    - **C2:** Two depthwise-separable blocks (one dilated, d=2)
    - **C3:** Two more depthwise-separable blocks
    - **C4:** Series of depthwise-separable blocks, with downsampling (stride=2), and dilated convolutions (dilation 2 & 3)[2]
    - **Head:** Adaptive AvgPool + 1x1 Conv for softmax logits

- **Receptive field before GAP:** 45[2]
- **Total trainable parameters:** ~36,170

## Experiments & Results

- **Augmentations:**
    - Horizontal flips, random shifts, rotations, CutOut/CoarseDropout

- **Training Schedule:**
    - Batch size: 128
    - Epochs: 50, with early stopping based on val loss
    - ReduceLROnPlateau for adaptive learning rate drops
    - Achieves >87% test set classification accuracy with strong data augmentation and careful regularization[1]

- **Metrics:**
    - Best validation loss: 0.3716
    - Final test accuracy: 87.31%

## Citation

If you use this repo or the contained techniques/model, please cite the repository and reference the author.

```markdown
@software{tuning_CNN_for_CIFAR10,
  author = {Sanesh Ashank},
  title = {Tuning CNN for CIFAR10 using Depthwise and Dilated Convolutions},
  year = {2025},
  url = {https://github.com/saneshashank/tuning_CNN_for_CIFAR10}
}
```

## License

This repository is released under the MIT License.

***

**Authors and Contact:**  
Shashank Sane 

**References:**  
- Code and architecture:[2]
- Training notebook and results:[1]

[1](https://github.com/saneshashank/tuning_CNN_for_CIFAR10/blob/main/tuning_CNN_CIFAR10.ipynb)
[2](https://github.com/saneshashank/tuning_CNN_for_CIFAR10/blob/main/tiny_cifar10_net.py)
[3](https://github.com/saneshashank/tuning_CNN_for_CIFAR10)
[4](https://github.com/saneshashank/tuning_CNN_for_CIFAR10/blob/main/README.md)
[5](https://github.com/albumentations-team/albumentations)
