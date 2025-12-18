import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
z_dim = 64
image_dim = 28 * 28 * 1

# Generator (same as training)
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
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

# Load trained model
gen = Generator().to(device)

checkpoint_path = 'checkpoints/checkpoint_epoch_50.pth'  # Change epoch as needed
checkpoint = torch.load(checkpoint_path, map_location=device)
gen.load_state_dict(checkpoint['gen_state_dict'])
gen.eval()

print(f"Loaded model from: {checkpoint_path}")
print(f"Trained for {checkpoint['epoch']+1} epochs")

# Generate new images
num_images = 100
noise = torch.randn(num_images, z_dim).to(device)

with torch.no_grad():
    fake_images = gen(noise).reshape(-1, 1, 28, 28)
    fake_images = fake_images.cpu().numpy()

# Display grid
fig, axes = plt.subplots(10, 10, figsize=(15, 15))
for i, ax in enumerate(axes.flat):
    ax.imshow(fake_images[i][0], cmap='gray')
    ax.axis('off')

plt.suptitle('Generated MNIST Digits', fontsize=20)
plt.tight_layout()
plt.savefig('results/generated_samples.png', dpi=150)
plt.show()

print("Generated 100 new images!")
print("Saved to: results/generated_samples.png")