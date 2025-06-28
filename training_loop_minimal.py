# the point of this file is for me to rewrite the training loop from scratch,
# including only the minimal relevant parts

import torch

from models import Generator, Discriminator


gen = Generator()
disc = Discriminator()

gen_opt = torch.optim.AdamW(gen.parameters(), lr=2e-4, betas=(0.5, 0.999))
disc_opt = torch.optim.AdamW(disc.parameters(), lr=2e-4, betas=(0.5, 0.999))

bce = torch.nn.BCELoss() 

def get_noise(m):
    return torch.rand(size=(m, 100)) * 2 - 1

def get_data(m):
    from pull_mnist import train_dataset
    indices = torch.randint(low=0, high=len(train_dataset), size=(m, ))
    samples = torch.stack([train_dataset[i][0] for i in indices])
    return samples.view((m, 784))

m = 128
for epoch in range(500):
    for k in range(1):
        # 1. sample m noise samples
        noise = get_noise(m)
        # 2. samples m real data samples from MNIST
        data = get_data(m)
        # 3. update discriminator
        predicted_real = disc(data)
        actual_real = torch.full(size=predicted_real.size(), fill_value=1.0)

        generated_fake = gen(noise).detach()
        predicted_fake = disc(generated_fake)
        actual_fake = torch.full(size=predicted_fake.size(), fill_value=0.0)

        loss_real = bce(predicted_real, actual_real)
        loss_fake = bce(predicted_fake, actual_fake)
        disc_loss = loss_real + loss_fake
        disc_opt.zero_grad()
        disc_loss.backward()
        disc_opt.step()
    
    # 4. sample m noise samples
    noise = get_noise(m)
    # 5. update generator
    generated_fake = gen(noise)  # NO detach here - we want gradients to flow
    predicted_fake = disc(generated_fake)
    actual_real = torch.full(size=predicted_fake.size(), fill_value=1.0)  # here we want disc to think that the fake images are real
    gen_loss = bce(predicted_fake, actual_real)
    gen_opt.zero_grad()
    gen_loss.backward()
    gen_opt.step()
    
    if epoch % 50 == 0:
        print(f'Epoch {epoch} done')
        print(f'Disc loss: {disc_loss:.4f}, Gen loss: {gen_loss:.4f}\n')

#save file
torch.save({
    'generator': gen.state_dict(),
    'discriminator': disc.state_dict()
}, 'minimal_models.pth'
)
