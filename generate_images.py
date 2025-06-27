import torch
import matplotlib.pyplot as plt
import numpy as np
from training_loop import load_models, get_noise

def generate_and_visualize(num_images=16, filepath="gan_models.pth"):
    """Generate and visualize MNIST images using trained generator"""
    
    # Load the trained models
    generator, discriminator = load_models(filepath)
    generator.eval()  # Set to evaluation mode
    
    # Generate noise
    noise = get_noise(num_images)
    
    # Generate fake images
    with torch.no_grad():
        fake_images = generator(noise)
    
    # Convert to numpy and reshape for visualization
    fake_images = fake_images.cpu().numpy()
    fake_images = fake_images.reshape(num_images, 28, 28)
    
    # Normalize from [-1, 1] to [0, 1] for display
    fake_images = (fake_images + 1) / 2
    
    # Create a grid visualization
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    fig.suptitle('Generated MNIST Images', fontsize=16)
    
    for i, ax in enumerate(axes.flat):
        ax.imshow(fake_images[i], cmap='gray')
        ax.set_title(f'Image {i+1}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return fake_images

if __name__ == "__main__":
    print("Generating 16 MNIST images...")
    generate_and_visualize()
