# 🎨 Simple GAN for MNIST Digit Generation

A PyTorch implementation of a Generative Adversarial Network (GAN) that learns to generate handwritten digits similar to the MNIST dataset. This project demonstrates the fundamental concepts of adversarial training with GPU acceleration support.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![CUDA](https://img.shields.io/badge/CUDA-Supported-76B900.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

---

## 📋 Table of Contents

- [Overview](#overview)
- [How GANs Work](#how-gans-work)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Training Details](#training-details)
- [Results](#results)
- [Troubleshooting](#troubleshooting)
- [Advanced Configuration](#advanced-configuration)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)

---

## 🎯 Overview

This project implements a **simple feedforward GAN** (Generative Adversarial Network) to generate handwritten digit images. The model consists of two neural networks:

- **Generator** 🎨: Creates fake images from random noise vectors
- **Discriminator** 🔍: Distinguishes between real MNIST images and generated fakes

Through adversarial training, the Generator learns to produce increasingly realistic digits that can fool the Discriminator.

### Key Highlights

- ✅ Simple and easy-to-understand architecture
- ✅ Automatic CUDA/GPU acceleration
- ✅ Real-time training visualization
- ✅ Checkpoint saving and resuming
- ✅ Clean, well-documented code
- ✅ Beginner-friendly implementation

---

## 🧠 How GANs Work

### The Game Theory Analogy

Think of a GAN as a game between two players:

1. **The Generator (Forger)** 🎨
   - Creates fake money (images)
   - Tries to make them look real
   - Learns from the Detective's feedback

2. **The Discriminator (Detective)** 🔍
   - Examines money to determine if it's real or fake
   - Gets better at spotting fakes over time
   - Forces the Forger to improve

### Training Process

```
                    ┌─────────────┐
                    │ Random Noise│
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │  Generator  │ ← Learns to create realistic images
                    └──────┬──────┘
                           │
                   Fake Images
                           │
        ┌──────────────────┴──────────────────┐
        │                                      │
        ▼                                      ▼
┌───────────────┐                    ┌─────────────────┐
│  Real Images  │                    │  Fake Images    │
│  from MNIST   │                    │ from Generator  │
└───────┬───────┘                    └────────┬────────┘
        │                                      │
        └──────────────┬───────────────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Discriminator   │ ← Learns to detect fakes
              └────────┬─────────┘
                       │
                       ▼
              Real (1) or Fake (0)?
```

### Mathematical Foundation

**Generator Loss:**
```
L_G = -log(D(G(z)))
```
Goal: Maximize the probability that the Discriminator classifies fake images as real.

**Discriminator Loss:**
```
L_D = -[log(D(x)) + log(1 - D(G(z)))]
```
Goal: Correctly classify real images as real (1) and fake images as fake (0).

---

## ✨ Features

- **Automatic Dataset Download**: MNIST dataset is downloaded automatically on first run
- **GPU Acceleration**: Automatic CUDA detection and utilization
- **Progress Visualization**: Generated samples saved every 5 epochs
- **Checkpoint System**: Save and resume training from any epoch
- **Memory Efficient**: Optimized batch processing
- **Extensible Architecture**: Easy to modify for other datasets
- **Clean Code**: Well-commented and follows best practices

---

## 📦 Requirements

### System Requirements

- **OS**: Linux, macOS, or Windows
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum (8GB+ recommended)
- **Storage**: ~200MB for dataset and checkpoints
- **GPU** (Optional but recommended): NVIDIA GPU with CUDA support

### Software Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
matplotlib>=3.5.0
numpy>=1.21.0
```

### CUDA Support

For GPU acceleration, ensure you have:
- NVIDIA GPU (GTX 10 series or newer recommended)
- CUDA Toolkit 11.8 or 12.1
- cuDNN (usually comes with PyTorch)

Check CUDA availability:
```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

---

## 🚀 Installation

### Step 1: Clone or Download the Repository

```bash
# If using git
git clone https://github.com/yourusername/simple-gan-mnist.git
cd simple-gan-mnist

# Or simply create a new directory
mkdir gan_project && cd gan_project
```

### Step 2: Install Dependencies

**Option A: Using pip**
```bash
pip install -r requirements.txt
```

**Option B: Using conda**
```bash
conda create -n gan_env python=3.9
conda activate gan_env
pip install -r requirements.txt
```

**Option C: Install with CUDA support**
```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install remaining packages
pip install matplotlib numpy
```

### Step 3: Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

Expected output:
```
PyTorch: 2.0.1
CUDA: True
```

---

## 📁 Project Structure

```
gan_project/
│
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── train_gan.py             # Main training script
├── generate.py              # Generate images from trained model
│
├── dataset/                 # Auto-created on first run
│   └── MNIST/
│       ├── raw/             # Downloaded dataset files
│       └── processed/       # Preprocessed tensors
│
├── results/                 # Auto-created: Generated images
│   ├── epoch_5.png
│   ├── epoch_10.png
│   ├── epoch_15.png
│   ├── ...
│   └── final_generated.png
│
└── checkpoints/             # Auto-created: Model checkpoints
    ├── checkpoint_epoch_5.pth
    ├── checkpoint_epoch_10.pth
    └── checkpoint_epoch_50.pth
```

---

## 🎮 Usage

### Basic Training

Run the training script with default settings:

```bash
python train_gan.py
```

### What Happens During Training

1. **Initialization** (First run only)
   - Downloads MNIST dataset (~12MB)
   - Creates necessary directories
   - Initializes models on GPU/CPU

2. **Training Loop**
   - Trains for 50 epochs (configurable)
   - Prints loss every 5 epochs
   - Saves generated images every 5 epochs
   - Saves model checkpoints

3. **Output**
   ```
   Using device: cuda
   GPU: NVIDIA GeForce RTX 3080
   Generator parameters: 533,776
   Discriminator parameters: 533,777
   
   ==================================================
   Starting Training Loop...
   ==================================================
   
   Epoch [5/50] | Loss D: 0.6234 | Loss G: 0.8912
   Epoch [10/50] | Loss D: 0.5821 | Loss G: 1.0234
   Epoch [15/50] | Loss D: 0.6102 | Loss G: 0.9456
   ...
   ```

### Generating New Images

After training, generate new digits:

```bash
python generate.py
```

This will:
- Load the trained model from the latest checkpoint
- Generate 100 new digit images
- Display them in a 10×10 grid
- Save to `results/generated_samples.png`

### Resume Training from Checkpoint

Modify `train_gan.py` to resume training:

```python
# Add this after model initialization
checkpoint = torch.load('checkpoints/checkpoint_epoch_30.pth')
gen.load_state_dict(checkpoint['gen_state_dict'])
disc.load_state_dict(checkpoint['disc_state_dict'])
opt_gen.load_state_dict(checkpoint['opt_gen_state_dict'])
opt_disc.load_state_dict(checkpoint['opt_disc_state_dict'])
start_epoch = checkpoint['epoch'] + 1

# Update the training loop
for epoch in range(start_epoch, num_epochs):
    # ... rest of training code
```

---

## 🎓 Training Details

### Model Architecture

**Generator**
```
Input: Random noise vector (64 dimensions)
   ↓
Linear(64 → 256) + ReLU
   ↓
Linear(256 → 512) + ReLU
   ↓
Linear(512 → 784) + Tanh
   ↓
Output: Image (28×28 pixels)
```

**Discriminator**
```
Input: Image (28×28 = 784 pixels)
   ↓
Linear(784 → 512) + LeakyReLU(0.2)
   ↓
Linear(512 → 256) + LeakyReLU(0.2)
   ↓
Linear(256 → 1) + Sigmoid
   ↓
Output: Probability (0=Fake, 1=Real)
```

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Learning Rate | 0.0002 | Adam optimizer learning rate |
| Beta1 | 0.5 | Adam momentum parameter |
| Beta2 | 0.999 | Adam momentum parameter |
| Batch Size | 64 | Images per training batch |
| Latent Dimension | 64 | Size of noise vector |
| Epochs | 50 | Total training epochs |
| Loss Function | Binary Cross Entropy | BCE Loss |

### Training Time

| Hardware | Approximate Time |
|----------|------------------|
| NVIDIA RTX 3080 | ~5 minutes |
| NVIDIA GTX 1060 | ~10 minutes |
| CPU (Intel i7) | ~30 minutes |
| CPU (Intel i5) | ~45 minutes |

### Memory Usage

- **GPU VRAM**: ~500MB - 1GB
- **System RAM**: ~2GB during training
- **Disk Space**: ~200MB (dataset + checkpoints)

---

## 📊 Results

### Training Progress Visualization

The quality of generated images improves over time:

**Epoch 5**: Noisy, random patterns
```
[Blurry, unclear shapes]
```

**Epoch 20**: Recognizable digit shapes
```
[Rough outlines of digits emerging]
```

**Epoch 50**: Clear, realistic digits
```
[Well-formed handwritten digits]
```

### Loss Interpretation

**Healthy Training:**
- Discriminator Loss: 0.5 - 0.7 (balanced)
- Generator Loss: 0.7 - 1.5 (learning)

**Warning Signs:**
- D_loss → 0: Discriminator too strong (Generator can't learn)
- D_loss → 1: Generator fooling Discriminator too easily (mode collapse)
- Losses oscillating wildly: Unstable training (reduce learning rate)

### Sample Outputs

Check the `results/` folder for:
- `epoch_X.png`: 8×8 grids of generated digits at epoch X
- `final_generated.png`: Final results after training
- `generated_samples.png`: 10×10 grid from `generate.py`

---

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. CUDA Out of Memory

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
```python
# Reduce batch size in train_gan.py
batch_size = 32  # or even 16
```

#### 2. Module Not Found Error

**Error:**
```
ModuleNotFoundError: No module named 'torch'
```

**Solution:**
```bash
pip install torch torchvision matplotlib numpy
```

#### 3. CUDA Not Available

**Error:**
```
Using device: cpu
```

**Solutions:**
- Check if you have an NVIDIA GPU: `nvidia-smi`
- Install CUDA-enabled PyTorch:
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  ```
- Verify CUDA installation: `nvcc --version`

#### 4. Poor Quality Generated Images

**Symptoms:**
- Images look random even after 50 epochs
- All images look the same (mode collapse)

**Solutions:**
1. Train longer (100-200 epochs)
2. Adjust learning rate:
   ```python
   lr = 0.0001  # Lower learning rate
   ```
3. Use label smoothing:
   ```python
   real_labels = torch.ones_like(disc_real) * 0.9  # Instead of 1.0
   fake_labels = torch.zeros_like(disc_fake) + 0.1  # Instead of 0.0
   ```

#### 5. Training Takes Too Long on CPU

**Solution:**
Consider using:
- Google Colab (free GPU)
- Kaggle Notebooks (free GPU)
- AWS/Azure GPU instances
- Reduce `num_epochs` to 20-30 for faster results

#### 6. Dataset Download Fails

**Error:**
```
HTTP Error 503: Service Unavailable
```

**Solution:**
Manually download MNIST from: http://yann.lecun.com/exdb/mnist/
- Place files in `dataset/MNIST/raw/`
- Set `download=False` in code

---

## ⚙️ Advanced Configuration

### Modify Hyperparameters

Edit `train_gan.py`:

```python
# Experiment with these values
lr = 0.0001              # Lower = more stable, slower
batch_size = 128         # Higher = faster, needs more VRAM
z_dim = 128              # Higher = more expressive generator
num_epochs = 100         # Train longer for better results
```

### Add Learning Rate Scheduling

```python
from torch.optim.lr_scheduler import StepLR

# After optimizer initialization
scheduler_gen = StepLR(opt_gen, step_size=30, gamma=0.5)
scheduler_disc = StepLR(opt_disc, step_size=30, gamma=0.5)

# In training loop (after optimizer.step())
scheduler_gen.step()
scheduler_disc.step()
```

### Monitor Training with TensorBoard

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/gan_experiment')

# In training loop
writer.add_scalar('Loss/Generator', loss_g.item(), epoch)
writer.add_scalar('Loss/Discriminator', loss_d.item(), epoch)
writer.add_images('Generated', fake, epoch)

# View with: tensorboard --logdir=runs
```

### Use Convolutional Layers (DCGAN)

For better image quality, upgrade to convolutional architecture:

```python
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 256, 7, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )
```

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

### Ways to Contribute

1. **Report Bugs**: Open an issue with details
2. **Suggest Features**: Propose new ideas or improvements
3. **Submit Pull Requests**: Fix bugs or add features
4. **Improve Documentation**: Help make this README better
5. **Share Results**: Post your generated images!

### Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/yourusername/simple-gan-mnist.git

# Create a new branch
git checkout -b feature/your-feature-name

# Make changes and commit
git commit -m "Add: your feature description"

# Push and create a pull request
git push origin feature/your-feature-name
```

---

## 📄 License

This project is licensed under the MIT License.

```
MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

---

## 📚 References

### Original Papers

- **GANs**: [Generative Adversarial Networks (Goodfellow et al., 2014)](https://arxiv.org/abs/1406.2661)
- **DCGAN**: [Unsupervised Representation Learning with Deep Convolutional GANs (Radford et al., 2015)](https://arxiv.org/abs/1511.06434)

### Useful Resources

- [PyTorch Official Documentation](https://pytorch.org/docs/stable/index.html)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [GAN Hacks - How to Train a GAN](https://github.com/soumith/ganhacks)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)

### Tutorials

- [Understanding GANs (Distill.pub)](https://distill.pub/)
- [A Beginner's Guide to GANs](https://wiki.pathmind.com/generative-adversarial-network-gan)
- [PyTorch DCGAN Tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)

---

## 🌟 Acknowledgments

- **MNIST Dataset**: Yann LeCun, Corinna Cortes, Christopher J.C. Burges
- **PyTorch Team**: For the excellent deep learning framework
- **Ian Goodfellow et al.**: For inventing GANs

---

## 📞 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/abhinave-p-b/simple-gan-mnist/issues)
- **Discussions**: [GitHub Discussions](https://github.com/abhinave-p-b/MNIST-GAN-Project/discussions)
- **Email**: abhinavepb92@gmail.com

---

## 🎉 Quick Start Summary

```bash
# 1. Install dependencies
pip install torch torchvision matplotlib numpy

# 2. Train the model
python train_gan.py

# 3. Generate new images
python generate.py
```

That's it! You should see generated MNIST digits improving over time.

---

**Happy Training! 🚀**

If this project helped you, please consider giving it a ⭐ on GitHub!
