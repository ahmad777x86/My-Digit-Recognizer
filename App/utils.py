import torch
import numpy as np
import cv2
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
from Model.Model import SequentialModel


model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"Model","model.pth")

def load_model(path : str = model_path):
    model = SequentialModel()
    model.load_state_dict(torch.load(path))
    print("Model Successfully Loaded")
    return model

def preprocess_image(image_data):
    array = np.frombuffer(image_data,dtype=np.uint8)
    print(f"Raw bytes length: {len(array)}")

    image = cv2.imdecode(array, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image,(28,28))
    print(f"Decoded image shape: {image.shape}")
    
    image_tensors = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()
    print(f"Image Tensors Size: {image_tensors.size()}")
    print("Image has successfully been decoded")
    return image_tensors