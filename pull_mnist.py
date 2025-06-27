import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms

data_dir = './data'
os.makedirs(data_dir, exist_ok=True)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) # normalize to [-1,1]
    # note - MNIST itself is already normalized to [0,1]
])

train_dataset = datasets.MNIST(
    root=data_dir,
    train=True,
    download=True,
    transform=transform,
)

test_dataset = datasets.MNIST(
    root=data_dir,
    train=False,
    download=True,
    transform=transform,
)
