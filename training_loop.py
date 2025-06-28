import torch
import torchvision
import matplotlib.pyplot as plt
import os

from models import Generator, Discriminator

def save_models(generator, discriminator, total_epochs_trained, filepath="gan_models.pth"):
    """Save both generator and discriminator models along with training progress"""
    torch.save({
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'total_epochs_trained': total_epochs_trained,
    }, filepath)
    print(f"Models saved to {filepath} (trained for {total_epochs_trained} epochs total)")

def load_models(filepath="gan_models.pth"):
    """Load both generator and discriminator models and return training progress"""
    checkpoint = torch.load(filepath)
    
    generator = Generator()
    discriminator = Discriminator()
    
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    
    # Get total epochs trained (default to 0 for old checkpoints)
    total_epochs_trained = checkpoint.get('total_epochs_trained', 0)
    
    print(f"Models loaded from {filepath} (previously trained for {total_epochs_trained} epochs)")
    return generator, discriminator, total_epochs_trained

# models
generator = Generator()
discriminator = Discriminator()

loss = torch.nn.BCELoss()

def get_noise(m):
    '''
    returns a tensor of shape (m, 100), with all of its entries Unif[-1,1]
    '''
    return 2 * torch.rand(size=(m, 100)) - 1

def get_data(m):
    '''
    returns a tensor of shape (m, 784) consisting of m random MNIST samples
    the MNIST pixels are already normalized to [-1,1]
    '''
    from pull_mnist import train_dataset
    indices = torch.randint(low=0, high=len(train_dataset), size=(m,))
    samples = torch.stack([train_dataset[i][0] for i in indices])
    return samples.view(m, -1)

def train_model(generator, discriminator, disc_opt, gen_opt, epochs=50, start_epoch=0, k=1, m=128):
    '''
    k: number of discriminator updates per generator update, k = 1 in the original paper
    m: batch size, m = 128 in the original paper, o3 recommends 256 or "maximum that fits in VRAM"
    start_epoch: starting epoch number for continuous training
    '''
    for epoch in range(epochs):
        current_epoch = start_epoch + epoch + 1  # Actual epoch number for display/saving
        
        for _ in range(k):
            # 1. sample m noise samples
            noise = get_noise(m)
            # 2. samples m images from the dataset
            data = get_data(m)
            # 3. update discriminator
            # generate fake data
            fake_data = generator(noise)
            
            # get discriminator predictions
            real_preds = discriminator(data)
            fake_preds = discriminator(fake_data.detach())  # detach to avoid generator gradients
            
            # calculate losses
            # label smoothing: real=0.9, fake=0.1
            real_targets = torch.full_like(real_preds, 0.9)
            fake_targets = torch.full_like(fake_preds, 0.1)

            real_loss = loss(real_preds, real_targets)
            fake_loss = loss(fake_preds, fake_targets)
            
            # update discriminator
            disc_opt.zero_grad()
            total_disc_loss = real_loss + fake_loss
            total_disc_loss.backward()
            disc_opt.step()    
        # 4. sample m noise samples
        noise = get_noise(m)
        # 5. update generator
        # generate fake data (no detach - we want gradients to flow to generator)
        fake_data = generator(noise)
        
        # get discriminator predictions on fake data
        fake_preds = discriminator(fake_data)
        
        gen_targets = torch.full_like(fake_preds, 0.9)  # generator tries to hit smoothed real label
        gen_loss = loss(fake_preds, gen_targets)
        
        # update generator
        gen_opt.zero_grad()
        gen_loss.backward()
        gen_opt.step()

        # calculate discriminator accuracy for monitoring
        with torch.no_grad():
            real_accuracy = (real_preds > 0.5).float().mean()
            fake_accuracy = (fake_preds < 0.5).float().mean()  # correct when < 0.5
            
        print(f"Epoch {current_epoch}/{start_epoch + epochs}")
        print(f"  Disc Loss: {total_disc_loss:.4f} | Gen Loss: {gen_loss:.4f}")
        print(f"  Disc Acc - Real: {real_accuracy:.2f} | Fake: {fake_accuracy:.2f}")
        print(f"  Balance: {(real_accuracy + fake_accuracy)/2:.2f}")
        print("-" * 50)
        
        # Save sample images every 250 epochs for progress tracking
        if current_epoch % 250 == 0:
            save_sample_images(generator, current_epoch)

    # Save models after training with total epochs trained
    total_epochs_after_training = start_epoch + epochs
    save_models(generator, discriminator, total_epochs_after_training)
    print("Training complete!")

def save_sample_images(generator, epoch, num_samples=16):
    """Save sample images for progress tracking"""
    # Create progress_images directory if it doesn't exist
    os.makedirs('progress_images', exist_ok=True)
    
    generator.eval()
    with torch.no_grad():
        noise = get_noise(num_samples)
        fake_images = generator(noise)
        fake_images = fake_images.cpu().numpy().reshape(num_samples, 28, 28)
        fake_images = (fake_images + 1) / 2  # normalize to [0,1]
        
        # Create grid
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        fig.suptitle(f'Generated Images - Epoch {epoch}', fontsize=14)
        
        for i, ax in enumerate(axes.flat):
            ax.imshow(fake_images[i], cmap='gray')
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'progress_images/progress_epoch_{epoch:04d}.png', dpi=100, bbox_inches='tight')
        plt.close()  # Don't display, just save
        
    generator.train()  # Back to training mode
    print(f"  → Saved progress image: progress_images/progress_epoch_{epoch:04d}.png")

if __name__ == "__main__":
    # Try to load existing models, otherwise start fresh
    try:
        generator, discriminator, total_epochs_trained = load_models()
        print("Continuing training from saved models...")
    except FileNotFoundError:
        print("No saved models found, starting fresh training...")
        total_epochs_trained = 0
    
    # Create optimizers AFTER loading models (critical!)
    disc_opt = torch.optim.AdamW(discriminator.parameters(), lr=1e-4, betas=(0.3, 0.999))
    gen_opt = torch.optim.AdamW(generator.parameters(), lr=1e-4, betas=(0.3, 0.999))
    
    train_model(generator, discriminator, disc_opt, gen_opt, epochs=15_000, start_epoch=total_epochs_trained)
