from torch.utils.data import DataLoader
import torchvision

train_data = torchvision.datasets.MNIST(
    root = './data', train=True, download=False, transform= torchvision.transforms.ToTensor()
)
# Set download = True, if you don't have the dataset downloaded
test_data = torchvision.datasets.MNIST(
    root = './data', train=False, download=False, transform= torchvision.transforms.ToTensor()
)

train_loader = DataLoader(dataset=train_data,shuffle=True, batch_size=64)
