# Overview
This project aims to detect pneumonia using chest X-ray images. The dataset contains a total of 26,684 images with a resolution of 1024x1024 pixels, gathered from the RSNA dataset. The images are categorized as follows:

## Project Structure
- [frontend/](frontend/): Contains the React frontend application.
- [backend/](backend/): Contains the Python files application for model training and preprocessing
 
20,672 images without pneumonia (~77%)
6,012 images with pneumonia (~22%)
To address data imbalance and accommodate hardware constraints, images are resized to 244x244 pixels. These images are then normalized and standardized (pixel values scaled to [0,1]). The dataset is divided into 24,000 images for training and 2,684 images for validation.

Images are organized into folders based on their class:

0: Healthy lungs
1: Lungs with pneumonia
> [!TIP]
> Approaches for Handling Imbalanced Data
- Do nothing
- Use a weighted loss function
- Oversample the minority class


Data Storage
The X-ray images are converted to numpy arrays. The mean and standard deviation of pixel values are computed for normalization purposes. The images are stored in respective folders according to their binary labels.

This project leverages PyTorch Lightning for efficient and scalable model training, along with TensorBoard for monitoring and visualizing training progress.

# Goal
🔬 My ongoing goal is to develop a user-friendly interface for my pneumonia classifier using React, ensuring it is accessible to everyone. This project aims to address real-world healthcare challenges by providing an intuitive tool for detecting pneumonia from X-ray images.

#

> There are more information and detailed explanation at the jupyter notebook on each file...
