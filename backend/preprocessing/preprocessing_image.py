import torch
from pathlib import Path
import pydicom
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision 
from torchvision import transforms
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from models.model import Predict, cam, visualize_cam

def retrieve_image():
    try:
        img = cv2.imread("image/xray.jpeg")
    except:
        print("Erro processar o arquivo")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (224, 224))
    return img

def preprocess_image():
    img = retrieve_image() 
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(0.49, 0.248),
    ])

    img = transform(img)
    img = img.unsqueeze(0)
    return img


original_image = retrieve_image()
preprocess_img = preprocess_image()

    
activationmap, output = cam(preprocess_img)

visualize_cam(original_image, activationmap, output)
    
