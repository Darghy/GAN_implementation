import torch
import torchvision

from models import Generator, Discriminator

# models
generator = Generator()
discriminator = Discriminator()

# optimizers, loss functions
disc_opt = torch.optim.AdamW(discriminator.parameters(), lr = 2e-4, betas=(0.5, 0.999)) # claude says 2e-4 is good
gen_opt = torch.optim.AdamW(generator.parameters(), lr = 2e-4, betas=(0.5, 0.999))

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

def train_model(generator, discriminator, epochs=50, k=1, m=128):
    '''
    k: number of discriminator updates per generator update, k = 1 in the original paper
    m: batch size, m = 128 in the original paper, o3 recommends 256 or "maximum that fits in VRAM"
    '''
    for epoch in range(epochs):
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
            real_loss = loss(real_preds, torch.ones_like(real_preds))  # target = 1 for real
            fake_loss = loss(fake_preds, torch.zeros_like(fake_preds))  # target = 0 for fake
            total_disc_loss = real_loss + fake_loss
            
            # update discriminator
            disc_opt.zero_grad()
            total_disc_loss.backward()
            disc_opt.step()    
        # 4. sample m noise samples
        noise = get_noise(m)
        # 5. update generator
        # generate fake data (no detach - we want gradients to flow to generator)
        fake_data = generator(noise)
        
        # get discriminator predictions on fake data
        fake_preds = discriminator(fake_data)
        
        # generator loss: we want discriminator to think fake data is real (target = 1)
        gen_loss = loss(fake_preds, torch.ones_like(fake_preds))
        
        # update generator
        gen_opt.zero_grad()
        gen_loss.backward()
        gen_opt.step()