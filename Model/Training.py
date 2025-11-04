from Model.Model import model
from DataLoading import train_loader
import torch.optim as optim
import torch.nn as nn
import torch

# Training

optimizer = optim.SGD(model.parameters(),lr=0.01)
criterion = nn.CrossEntropyLoss()

model.train()
for batch_idx , (features,target) in enumerate(train_loader):
    optimizer.zero_grad() # As gradients accumulate it is necessary to zero them

    prediction = model(features)
    loss = criterion(prediction,target)
    print(f"Loss={loss.item()},Batch={batch_idx}")
    loss.backward()
    optimizer.step()

print("Training Complete!")

# Saving model

torch.save(model,'model.pth')

