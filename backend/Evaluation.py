import os, random
import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '')))

PNEUMONIA_FOLDER = r"/home/fkohut/Code/projects/PneumoniaClassificator/backend/test/PNEUMONIA"

HEALTHY_FOLDER = r"/home/fkohut/Code/projects/PneumoniaClassificator/backend/test/NORMAL"

SAVE_PATH = Path("Processed/")

fig, axis = plt.subplots(3, 3, figsize=(9, 9))
c = 0
for i in range(3):
    for j in range(3):
        a = os.listdir(PNEUMONIA_FOLDER)
        print(a[c])
        file = PNEUMONIA_FOLDER+'//'+a[c]
        file = cv2.imread(file)
        file = cv2.cvtColor(file, cv2.COLOR_BGR2GRAY)
        file = cv2.resize(file, (224, 224))
        axis[i][j].imshow(file, cmap="bone")
        c+=1
plt.show()