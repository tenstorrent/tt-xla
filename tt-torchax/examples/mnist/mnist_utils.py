
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def TestLoader(batch_size=10, shuffle=False):
    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))])
    # Load the test dataset
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False,
                                           download=True, transform=transform)
    # Create a data loader for the test dataset
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

def TrainLoader(batch_size=64, shuffle=True):
    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))])
    # Load the training dataset
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform)
    # Create a data loader for the training dataset
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
