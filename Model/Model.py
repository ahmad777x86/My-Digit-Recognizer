import torch.nn as nn


class SequentialModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(28*28,512)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(512,10)

    def forward(self,x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x
    
model = SequentialModel()
