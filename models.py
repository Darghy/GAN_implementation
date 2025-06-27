import torch.nn as nn

# Note - the models expect flattened 784-dimensional input from our data pipeline

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256), # we will be sampling from 100-dimensional noise
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Tanh(), #this normalizes the pixels to be between -1 and 1
        )
    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(), # gives us a probability
        )
    def forward(self, x):
        return self.model(x)