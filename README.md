# Overview
This project aims to leverage the power of machine learning and convolutional neural networks(CNNs) to detect pneumonia in chest X-ray (thorax), scans, By utilizing advanced image processing techniques and Class Activation Mapping (CAM) to highlight affected areas. A user-friendly React frontend allows for easy x-ray images uploads, while the backend processes the images, This integration provides a seamless and efficient diagnostic tool to aid in early pneumonia detection and localization.

# Why is it important?
Pneumonia is one of the most common causes of death worldwide. It is a condition of the inflammation of the lungs, specifically in the alveoli, When an individual has pneumonia, the alveoli are filled with pus and fluid, which makes breathing painful and limits oxygen intake, according to the National Institutes of Health, pneumonia is the eighth leading cause of death and first among infectious causes of death. The mortality rate is as high as 23% for patients admitted to the intensive care unit for severe pneumonia. Therefore, early diagnosis is crucial.


## Project Structure
- [frontend/](frontend/): Contains the React frontend application.
- [backend/](backend/): Contains the Python files application for model preprocessing, model training, x-ray preprocessing and finally the output prediction with Class activation mapping (CAM)


## Model training
Using the Radiological Society of North America(RSNA) Dataset, the model was trained on:
20,672 images without pneumonia (~77%)
6,012 images with pneumonia (~23%)
To address data imbalance and accommodate hardware constraints, images are resized to 244x244 pixels. The X-ray images are converted to numpy arrays. The mean and standard deviation of pixel values are computed for normalization purposes. (pixel values scaled to [0,1]).
The dataset is divided into 24,000 images for training and 2,684 images for validation, 

The images are stored in respective folders according to their binary labels:
0: Healthy lungs
1: Lungs with pneumonia

## Model Evaluation
After training the model, it is essential to thoroughly evaluate its accuracy and performance. This involves assessing how well the model can generalize to new, unseen data, which helps ensure that it is not just memorizing the training data but learning underlying patterns that can be applied to real-world scenarios. Evaluation metrics such as precision, recall, F1 score, and confusion matrix are used to provide a comprehensive understanding of the model's strengths and weaknesses.





