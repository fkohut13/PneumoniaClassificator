import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms
from models.model_training.train import PneumoniaModel

## Predicts a single image. (Not used in flask)
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
        
        
## Predicts a batch of images. (Not used in flask) (in progress)   
def Predict_batch(Path, batch):
    predictions = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = PneumoniaModel.load_from_checkpoint(Path)
    model.eval()
    model.to(device)
    with torch.no_grad():
        for file in tqdm(batch):
            file = file.to(device).float().unsqueeze(0)
            single_prediction = torch.sigmoid(model(file)[0].cpu())
            predictions.append(single_prediction)
    predictions = torch.tensor(predictions)
    
## Predicts with the Class Activation Map         
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

## Visualizes the Class Activation Map        
def visualize_cam(originalimg ,cam, pred):
    cam = transforms.functional.resize(cam.unsqueeze(0), (224, 224))[0]
    fig, axis = plt.subplots(1,1)
    axis.imshow(originalimg, cmap="bone")
    axis.imshow(cam, alpha=0.5, cmap="jet")
    axis.axis('off')
    plt.savefig("image/xray_processed.jpeg",
    bbox_inches='tight', pad_inches=0)
