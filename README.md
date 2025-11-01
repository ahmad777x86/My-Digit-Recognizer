# My-Digit-Recognizer
---
## Overview

This project is a simple digit recognizer built using PyTorch. The neural network model is trained on the MNIST handwritten digits dataset and can predict the digit from an image.

---
## Model Architecture

The model is a sequential neural network with the following layers:
1.  **Flatten Layer**: Flattens the 28x28 image into a 1D tensor of size 784.
2.  **Linear Layer 1**: A fully connected layer with 784 input features and 512 output features.
3.  **ReLU Activation**: Applies the Rectified Linear Unit activation function.
4.  **Linear Layer 2**: A fully connected layer with 512 input features and 10 output features (for the 10 digits).

---
## Dataset

The model is trained on the [MNIST dataset](http://yann.lecun.com/exdb/mnist/), which is a large database of handwritten digits that is commonly used for training and testing in the field of machine learning. The dataset contains 60,000 training images and 10,000 testing images.

---
## Installation and Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    ```
2.  **Install Python packages:**
    ```bash
    pip install torch torchvision matplotlib
    ```
3.  **Download the dataset:**
    Run the `DataLoading.py` script with `download=True` in the `torchvision.datasets.MNIST` function to download the dataset.
    ```python
    train_data = torchvision.datasets.MNIST(
        root = './data', train=True, download=True, transform= torchvision.transforms.ToTensor()
    )
    test_data = torchvision.datasets.MNIST(
        root = './data', train=False, download=True, transform= torchvision.transforms.ToTensor()
    )
    ```
---
## Training

To train the model, run the `Training.py` script.
```bash
python Training.py
```
This will train the model on the MNIST training data and save the trained model to `model.pth`.

---
## Usage

To test the model, run the `Testing.py` script.
```bash
python Testing.py
```
You can change the `index` variable in `Testing.py` to an integer value between 0 and 9999 to test the model's prediction on a specific image from the test set. The script will print the predicted digit and the actual digit, and it will display the image with the prediction as the title.