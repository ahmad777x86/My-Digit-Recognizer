import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import mnist


# Data Loading

x_train = mnist.train_images
y_train = mnist.train_labels

x_test= mnist.test_images
y_test = mnist.test_labels

# Model definition

model = nn.Sequential([
        nn.Flatten(28,28),
        nn.ReLU(128),
        nn.ReLU(128),
        nn.Softmax(10)
])


# model compilation

model = torch.compile(model=model)

# Loss and Optimizer

criteria = nn.CrossEntropyLoss()
Optimizer = torch.optim.Adam()

# model training

for epochs in range(3):
    pass
