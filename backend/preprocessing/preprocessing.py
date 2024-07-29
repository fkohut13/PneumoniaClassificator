from pathlib import Path
import pydicom
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

labels = pd.read_csv("./dataset/stage_2_train_labels.csv")
labels = labels.drop_duplicates('patientId')
##print(labels.head())

ROOT_PATH = Path("./dataset/stage_2_train_images/")
SAVE_PATH = Path("./dataset/Processed")

#Visualization purposes
def Visualize():
    fig, axis = plt.subplots(3, 3, figsize=(9,9))
    c=0
    for i in range(3):
        for j in range(3):
            patient_id = labels.patientId.iloc[c]
            dcm_path = ROOT_PATH/patient_id
            dcm_path = dcm_path.with_suffix(".dcm")
            dcm = pydicom.read_file(dcm_path).pixel_array
            label = labels["Target"].iloc[c]
      
            if (label == 1):
                pn = "Pneumonia"
            else:
                pn = "Healthy"
            ##--
            axis[i][j].imshow(dcm, cmap="gray")
            axis[i][j].set_title(pn)
            c+=1 
    plt.show()      

def Preprocess():
    sums = 0
    sums_squared = 0
    for c, patient_id in enumerate(tqdm(labels.patientId)):
        dcm_path = ROOT_PATH/patient_id
        dcm_path = dcm_path.with_suffix(".dcm")
    
        dcm = pydicom.read_file(dcm_path).pixel_array / 255

        dcm_array = cv2.resize(dcm, (224, 224)).astype(np.float16)
        label = labels.Target.iloc[c]
    
        #Validation split
        train_or_val = "train" if c < 24000 else "val"
    
        current_save_path = SAVE_PATH/train_or_val/str(label)
        current_save_path.mkdir(parents=True, exist_ok=True)
        np.save(current_save_path/patient_id, dcm_array)
    
        normalizer = dcm_array.shape[0] * dcm_array.shape[1]
        if train_or_val == "train":
            sums += np.sum(dcm_array) / normalizer
            sums_squared += (np.power(dcm_array, 2).sum())
            
    mean = sums / 2400
    std = np.sqrt(sums_squared / 24000 - (mean**2))
    print(f"Mean: {mean}, Standard Deviation: {std}")