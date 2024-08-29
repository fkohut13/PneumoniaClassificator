import cv2
from torchvision import transforms
from models.model import cam, visualize_cam

def retrieve_image():
    try:
        img = cv2.imread("image/xray.jpeg")
    except:
        print("Erro processar o arquivo")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (224, 224))
    return img

def preprocess_image():
    preprocessed_img = retrieve_image()
    original_img = retrieve_image() 
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(0.49, 0.248),
    ])

    preprocessed_img = transform(preprocessed_img)
    preprocessed_img = preprocessed_img.unsqueeze(0)
    return original_img, preprocessed_img


original_img, preprocessed_img = preprocess_image()

    
activationmap, output = cam(preprocessed_img)

visualize_cam(original_img, activationmap, output)
    
