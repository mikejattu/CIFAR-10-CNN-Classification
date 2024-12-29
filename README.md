# CIFAR-10 Classification with CNN and Pretrained Model

This repository contains a project for image classification on the CIFAR-10 dataset. It implements two models:

1. A custom-built Convolutional Neural Network (CNN) from scratch.
2. A fine-tuned pretrained ConvNeXt-Tiny model.

The goal of the project is to classify the 10 classes of CIFAR-10 images with high accuracy. The custom CNN achieves an accuracy of over 65%, while the pretrained ConvNeXt-Tiny model achieves an accuracy of 90% or higher after fine-tuning.

## Project Overview

### 1. Classification with CNN (Custom Model)
- **Architecture**:  A convolutional neural network consisting of:
  - 3 Convolutional Layers (with increasing filter sizes: 16, 32, 64)
  - 3 Convolutional Layers (with increasing filter sizes: 16, 32, 64)
  - 1 Fully Connected Layer (with 120 output units)
- **Validation & Overfitting**: The model was trained on the CIFAR-10 dataset, with validation metrics monitored to assess performance and prevent overfitting.

### 2. Classification with Pretrained Model (ConvNeXt-Tiny)
- **Model**: The pretrained ConvNeXt-Tiny model was fine-tuned for CIFAR-10 classification.
- **Training**: The pretrained weights were adapted to the CIFAR-10 dataset, achieving an accuracy of over 90% on the test set.

## Setup and Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/cifar-10-classification.git
   cd cifar-10-classification
   ```
2. 	Install dependencies:
   ```bash
    pip install -r requirements.txt
```
3.	To train the custom CNN model:
```bash
 python CNN_main.py
```
4. To fine-tune the pretrained ResNet-18 model:
   ```bash
   python CNN_main.py pretrained=1
   ```
5. To load and test the fine-tuned model:
```bash
python CNN_main.py pretrained=1 load_model=1
```

## Files in the Repository
	•	CNN_main.py: Main script for training and testing the CNN models.
	•	CNN_submission.py: Template code for model submission.
	•	best_model.pth: Fine-tuned pretrained model saved after training.
