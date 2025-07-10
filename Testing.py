from DataLoading import test_data
import torch

import matplotlib.pyplot as plt


# Testing

index = 1

model = torch.load('model.pth',weights_only=False)
img , label = test_data[index]
with torch.no_grad():
    prediction = model(img)
    predicted_n = prediction.argmax(dim=1).item()

    print(f"Predicted digit: {predicted_n}, Actual digit: {label}")
    plt.title(f"Predicted digit: {predicted_n}, Actual digit: {label}")
    plt.imshow(img.squeeze(),cmap='gray')
    plt.show()

