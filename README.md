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

<h3 align="left">Languages and Tools used:</h3>

Programming Languages:

<a href="https://developer.mozilla.org/en-US/docs/Web/JavaScript" target="_blank" rel="noreferrer"> 
    <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/javascript/javascript-original.svg" alt="javascript" width="40" height="40"/> 
</a> 
<a href="https://www.python.org" target="_blank" rel="noreferrer"> 
    <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="40" height="40"/> 
</a> 

Frameworks and Libraries:

<a href="https://flask.palletsprojects.com/" target="_blank" rel="noreferrer"> 
    <img src="https://www.vectorlogo.zone/logos/pocoo_flask/pocoo_flask-icon.svg" alt="flask" width="40" height="40"/> 
</a> 
<a href="https://reactjs.org/" target="_blank" rel="noreferrer"> 
    <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/react/react-original-wordmark.svg" alt="react" width="40" height="40"/> 
</a> 
<a href="https://pytorch.org/" target="_blank" rel="noreferrer"> 
    <img src="https://www.vectorlogo.zone/logos/pytorch/pytorch-icon.svg" alt="pytorch" width="40" height="40"/> 
</a> 
<a href="https://opencv.org/" target="_blank" rel="noreferrer"> 
    <img src="https://www.vectorlogo.zone/logos/opencv/opencv-icon.svg" alt="opencv" width="40" height="40"/> 
</a> 
<a href="https://pandas.pydata.org/" target="_blank" rel="noreferrer"> 
    <img src="https://raw.githubusercontent.com/devicons/devicon/2ae2a900d2f041da66e950e4d48052658d850630/icons/pandas/pandas-original.svg" alt="pandas" width="40" height="40"/> 
</a> 

Tools:

<a href="https://git-scm.com/" target="_blank" rel="noreferrer"> 
    <img src="https://www.vectorlogo.zone/logos/git-scm/git-scm-icon.svg" alt="git" width="40" height="40"/> 
</a> 
<a href="https://postman.com" target="_blank" rel="noreferrer"> 
    <img src="https://www.vectorlogo.zone/logos/getpostman/getpostman-icon.svg" alt="postman" width="40" height="40"/> 
</a> 

Styling Tools:

<a href="https://www.w3schools.com/css/" target="_blank" rel="noreferrer"> 
    <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/css3/css3-original-wordmark.svg" alt="css3" width="40" height="40"/> 
</a> 
<a href="https://tailwindcss.com/" target="_blank" rel="noreferrer"> 
    <img src="https://www.vectorlogo.zone/logos/tailwindcss/tailwindcss-icon.svg" alt="tailwind" width="40" height="40"/> 
</a> 

Markup Languages:

<a href="https://www.w3.org/html/" target="_blank" rel="noreferrer"> 
    <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/html5/html5-original-wordmark.svg" alt="html5" width="40" height="40"/> 
</a>




