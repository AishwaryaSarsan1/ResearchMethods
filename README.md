Lung Cancer Detection Project

This package contains source codes for training deep learning models (ResNet50 and ConvNeXt) on histopathological lung cancer images (LC25000 dataset).

Files:
- ResNet50.py: Training script for ResNet50
- ConvNeXt.py: Training script for ConvNeXt Base

Dataset Expected Directory:
- lung_cancer_data/train
- lung_cancer_data/val
- Link to dataset  https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images

Requirements:
- PyTorch
- torchvision
- timm (for ConvNeXt)
