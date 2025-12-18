import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

# 1. Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

lr = 0.0002
batch_size = 64
z_dim = 64
image_dim = 28 * 28 * 1
num_epochs = 50

# Create directories
os.makedirs("dataset", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# 2. Generator (The Forger)
class Generator(nn.Module):
    def __init__(self):  # Fixed: was init, should be __init__
        super().__init__()  # Fixed: was super().init()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, image_dim),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.net(x)

# 3. Discriminator (The Detective)
class Discriminator(nn.Module):
    def __init__(self):  # Fixed: was init, should be __init__
        super().__init__()  # Fixed: was super().init()
        self.net = nn.Sequential(
            nn.Linear(image_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

# Initialize models
gen = Generator().to(device)
disc = Discriminator().to(device)

print(f"Generator parameters: {sum(p.numel() for p in gen.parameters()):,}")
print(f"Discriminator parameters: {sum(p.numel() for p in disc.parameters()):,}")

# 4. Optimizers and Loss
opt_gen = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))
criterion = nn.BCELoss()

# 5. Data Loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

dataset = datasets.MNIST(
    root="dataset/", 
    transform=transform, 
    download=True
)

loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Fixed noise for visualization
fixed_noise = torch.randn(64, z_dim).to(device)

# Training Loop
print("\n" + "="*50)
print("Starting Training Loop...")
print("="*50 + "\n")

for epoch in range(num_epochs):
    for batch_idx, (real_images, _) in enumerate(loader):
        # Prepare data
        real_images = real_images.view(-1, 784).to(device)
        curr_batch_size = real_images.shape[0]
        
        ### STEP 1: Train Discriminator
        noise = torch.randn(curr_batch_size, z_dim).to(device)
        fake_images = gen(noise)
        
        # Real images
        disc_real = disc(real_images).view(-1)
        loss_d_real = criterion(disc_real, torch.ones_like(disc_real))
        
        # Fake images
        disc_fake = disc(fake_images.detach()).view(-1)
        loss_d_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        
        # Total discriminator loss
        loss_d = (loss_d_real + loss_d_fake) / 2
        
        disc.zero_grad()
        loss_d.backward()
        opt_disc.step()
        
        ### STEP 2: Train Generator
        output = disc(fake_images).view(-1)
        loss_g = criterion(output, torch.ones_like(output))
        
        gen.zero_grad()
        loss_g.backward()
        opt_gen.step()
    
    # Print progress
    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}] | Loss D: {loss_d:.4f} | Loss G: {loss_g:.4f}")
        
        # Generate and save sample images
        with torch.no_grad():
            fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
            fake = fake.cpu().numpy()
            
            fig, axes = plt.subplots(8, 8, figsize=(10, 10))
            for i, ax in enumerate(axes.flat):
                ax.imshow(fake[i][0], cmap='gray')
                ax.axis('off')
            
            plt.tight_layout()
            plt.savefig(f'results/epoch_{epoch+1}.png')
            plt.close()
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'gen_state_dict': gen.state_dict(),
            'disc_state_dict': disc.state_dict(),
            'opt_gen_state_dict': opt_gen.state_dict(),
            'opt_disc_state_dict': opt_disc.state_dict(),
        }, f'checkpoints/checkpoint_epoch_{epoch+1}.pth')

print("\n" + "="*50)
print("Training Complete!")
print("="*50)

# Generate final samples
with torch.no_grad():
    fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
    fake = fake.cpu().numpy()
    
    fig, axes = plt.subplots(8, 8, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        ax.imshow(fake[i][0], cmap='gray')
        ax.axis('off')
    
    plt.suptitle('Final Generated Images', fontsize=16)
    plt.tight_layout()
    plt.savefig('results/final_generated.png')
    print("\nFinal results saved to: results/final_generated.png")