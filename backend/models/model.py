import torch
from pathlib import Path
import pydicom
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision ##For easy model training without the for loops.
from torchvision import transforms
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

def load_file(path):
    return np.load(path).astype(np.float32)

def data_Augmentation():
    train_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.49, 0.248),
    transforms.RandomAffine(degrees=(-5, -5), translate=(0, 0.05), scale=(0.9, 1.1)),
    transforms.RandomResizedCrop((224, 224), scale=(0.35, 1))
    ])

    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.49, 0.248),
    ])

    train_dataset = torchvision.datasets.DatasetFolder("./dataset/Processed/train/", loader=load_file, extensions="npy", transform=train_transforms)

    val_dataset = torchvision.datasets.DatasetFolder("./dataset/Processed/val/", loader=load_file, extensions="npy", transform=val_transforms)
    
    return train_dataset, val_dataset

train_dataset, val_dataset = data_Augmentation()

def visualize():
    fig, axis = plt.subplots(2,2, figsize=(9, 9))
    for i in range(2):
        for j in range(2):
            random_index = np.random.randint(0, 24000)
            x_ray, label = train_dataset[random_index]
            axis[i][j].imshow(x_ray[0], cmap="gray")
            if (label == 1):
                pn = "Pneumonia"
            else:
                pn = "Healthy"
            axis[i][j].set_title(pn)
    plt.show()

batch_size = 64
num_workers = 4
gpus = 0

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)


class PneumoniaModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet18()
        self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)#Modify first conv layer from resnet18
        self.model.fc = torch.nn.Linear(in_features=512, out_features=1)#Modify fully connected layer from resnet18
        
        self.feature_map = torch.nn.Sequential(*list(self.model.children())[:-2])
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3]))
        
        ##Accuracy
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        
        ##Precision
        self.train_precision = torchmetrics.Precision()
        self.val_precision = torchmetrics.Precision()
        
    def forward(self, data):
        feature_map = self.feature_map(data)
        avg_pool_output = torch.nn.functional.adaptive_avg_pool2d(input=feature_map, output_size=(1,1))
        avg_output_flattened = torch.flatten(avg_pool_output)
        pred = self.model.fc(avg_output_flattened)
        return pred, feature_map
    
    def training_step(self, batch, batch_idx):
        xray, label = batch
        label = label.float()
        pred = self(xray)[:,0]
        loss = self.loss_fn(pred, label)
        self.log("Train Loss", loss)
        self.log("Step Train ACC", self.train_acc(torch.sigmoid(pred), label.int()))
        self.log("Step Train Precision", self.train_precision(torch.sigmoid(pred), label.int()))
        return loss
    def training_epoch_end(self, outs):
        self.log("Train ACC", self.train_acc.compute())
        
    def validation_step(self, batch, batch_idx):
        xray, label = batch
        label = label.float()
        pred = self(xray)[:,0]
        loss = self.loss_fn(pred, label)
        
        self.log("Val Loss", loss)
        self.log("Step Val ACC", self.val_acc(torch.sigmoid(pred), label.int()))
        self.log("Step Val Precision", self.val_precision(torch.sigmoid(pred), label.int()))
    def validation_epoch_end(self, outs):
        self.log("Val ACC", self.val_acc.compute())
        
    def configure_optimizers(self):
        return [self.optimizer]
    


def train():
    #Configs
    model = PneumoniaModel()
    checkpoint_callback = ModelCheckpoint(monitor="Val ACC", save_top_k=10, mode="max")
    trainer = pl.Trainer(gpus=gpus, logger=TensorBoardLogger(save_dir="./logs"), log_every_n_steps=1, callbacks=checkpoint_callback, max_epochs=35)
    #Train model
    trainer.fit(model, train_loader, val_loader)
    
    
def Model_evaluation(Path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = PneumoniaModel.load_from_checkpoint(Path)
    model.eval()
    model.to(device)
    preds = []
    labels = []
    with torch.no_grad():
        for data, label in tqdm(val_dataset):
            data = data.to(device).float().unsqueeze(0)
            pred = torch.sigmoid(model(data)[0].cpu())
            preds.append(pred)
            labels.append(label)
        preds = torch.tensor(preds)
        labels = torch.tensor(labels).int()
    acc = torchmetrics.Accuracy()(preds, labels)
    precision = torchmetrics.Precision()(preds, labels)
    recall = torchmetrics.Recall()(preds, labels)
    cm = torchmetrics.ConfusionMatrix(num_classes=2)(preds, labels)
    print(f"Val Acurracy {acc}")
    print(f"Val Precision {precision}")
    print(f"Val Recall {recall}")
    print(f"Confusion Matrix {cm}")

def Predict(Path, image):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = PneumoniaModel.load_from_checkpoint(Path)
    model.eval()
    model.to(device)
    with torch.no_grad():
        image = image.to(device)
        output = model(image)
        output = torch.sigmoid(output)
        prediction = (output > 0.5).int().item()
        print(f"Output: {output.item()}, Prediction: {'Pneumonia' if prediction == 1 else 'No pneumonia'}")

def cam(img):
    model = PneumoniaModel.load_from_checkpoint("weights/weights_3.ckpt", strict=False)
    model.eval()
    with torch.no_grad():
        pred, features = model(img)
    features = features.reshape((512, 49))
    weight_params = list(model.model.fc.parameters())[0]
    weight = weight_params[0].detach()
    
    cam = torch.matmul(weight, features)
    cam_img = cam.reshape(7,7).cpu()
    return cam_img, torch.sigmoid(pred)

        
def visualize_cam(originalimg ,cam, pred):
    cam = transforms.functional.resize(cam.unsqueeze(0), (224, 224))[0]
    fig, axis = plt.subplots(1,1)
    axis.imshow(originalimg, cmap="bone")
    axis.imshow(cam, alpha=0.5, cmap="jet")
    if (pred > 0.5):
        plt.title("Pneumonia")
        plt.suptitle(f'Prediction: {pred}')
    else:
        plt.title("Healthy")
        plt.suptitle(f'Prediction: {pred}')
    plt.show()
