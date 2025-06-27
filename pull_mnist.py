import os
import torchvision
import torchvision.transforms as transforms

def download_mnist():
    # Create data directory if it doesn't exist
    data_dir = './data'
    os.makedirs(data_dir, exist_ok=True)
    
    # Define transform to convert PIL images to tensors
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])
    
    # Download training set
    train_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    
    # Download test set
    test_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )
    
    print(f"MNIST dataset downloaded to {data_dir}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_dataset, test_dataset

if __name__ == "__main__":
    download_mnist()
