import torch
import numpy as np
import cv2

def load_model(path : str = "../Model/model.pth"):
    model = torch.load(path, weights_only=False)
    print("Model Successfully Loaded")
    return model

def preprocess_image(image_data):
    array = np.frombuffer(image_data,dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_GRAYSCALE)
    print("Image has successfully been decoded")