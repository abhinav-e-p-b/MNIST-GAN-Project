from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import io
import base64
import os
import sys

# Add parent directory to path to import your models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_gan import Generator, Discriminator

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
training_active = False
current_gen = None
current_disc = None

# Load model checkpoint
def load_model(checkpoint_path):
    global current_gen
    current_gen = Generator().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    current_gen.load_state_dict(checkpoint['gen_state_dict'])
    current_gen.eval()
    return current_gen

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'online',
        'device': str(device),
        'cuda_available': torch.cuda.is_available()
    })

@app.route('/api/train', methods=['POST'])
def start_training():
    global training_active
    data = request.json
    
    training_active = True
    
    # Extract parameters
    lr = data.get('learningRate', 0.0002)
    batch_size = data.get('batchSize', 64)
    z_dim = data.get('latentDim', 64)
    epochs = data.get('epochs', 50)
    
    # Start training in background thread
    socketio.start_background_task(
        train_gan_async, lr, batch_size, z_dim, epochs
    )
    
    return jsonify({'message': 'Training started', 'status': 'training'})

def train_gan_async(lr, batch_size, z_dim, epochs):
    global training_active, current_gen, current_disc
    
    # Import training components
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    import torch.optim as optim
    
    # Initialize models
    current_gen = Generator().to(device)
    current_disc = Discriminator().to(device)
    
    opt_gen = optim.Adam(current_gen.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_disc = optim.Adam(current_disc.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCELoss()
    
    # Load data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    dataset = datasets.MNIST(
        root="../dataset/",
        transform=transform,
        download=True
    )
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training loop
    for epoch in range(epochs):
        if not training_active:
            break
            
        for batch_idx, (real_images, _) in enumerate(loader):
            real_images = real_images.view(-1, 784).to(device)
            curr_batch_size = real_images.shape[0]
            
            # Train Discriminator
            noise = torch.randn(curr_batch_size, z_dim).to(device)
            fake_images = current_gen(noise)
            
            disc_real = current_disc(real_images).view(-1)
            loss_d_real = criterion(disc_real, torch.ones_like(disc_real))
            
            disc_fake = current_disc(fake_images.detach()).view(-1)
            loss_d_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            
            loss_d = (loss_d_real + loss_d_fake) / 2
            
            current_disc.zero_grad()
            loss_d.backward()
            opt_disc.step()
            
            # Train Generator
            output = current_disc(fake_images).view(-1)
            loss_g = criterion(output, torch.ones_like(output))
            
            current_gen.zero_grad()
            loss_g.backward()
            opt_gen.step()
        
        # Emit progress
        socketio.emit('training_progress', {
            'epoch': epoch + 1,
            'total_epochs': epochs,
            'generator_loss': float(loss_g.item()),
            'discriminator_loss': float(loss_d.item())
        })
        
        # Generate sample images every 5 epochs
        if (epoch + 1) % 5 == 0:
            samples = generate_sample_images(8)
            socketio.emit('sample_images', {
                'epoch': epoch + 1,
                'images': samples
            })
    
    training_active = False
    socketio.emit('training_complete', {'message': 'Training finished'})

@app.route('/api/train/stop', methods=['POST'])
def stop_training():
    global training_active
    training_active = False
    return jsonify({'message': 'Training stopped', 'status': 'stopped'})

@app.route('/api/generate', methods=['POST'])
def generate_images():
    data = request.json
    num_images = data.get('numImages', 64)
    checkpoint = data.get('checkpoint', 'epoch_50')
    
    # Load model if not already loaded
    if current_gen is None:
        checkpoint_path = f'../checkpoints/checkpoint_{checkpoint}.pth'
        load_model(checkpoint_path)
    
    # Generate images
    grid_size = int(np.sqrt(num_images))
    images = generate_sample_images(grid_size)
    
    return jsonify({
        'images': images,
        'count': len(images)
    })

def generate_sample_images(grid_size):
    """Generate grid_size x grid_size sample images"""
    global current_gen
    
    if current_gen is None:
        checkpoint_path = '../checkpoints/checkpoint_epoch_50.pth'
        load_model(checkpoint_path)
    
    z_dim = 64
    num_images = grid_size * grid_size
    
    noise = torch.randn(num_images, z_dim).to(device)
    
    with torch.no_grad():
        fake_images = current_gen(noise).reshape(-1, 1, 28, 28)
        fake_images = fake_images.cpu().numpy()
    
    # Convert to base64 strings
    image_list = []
    for img in fake_images:
        # Denormalize from [-1, 1] to [0, 255]
        img_array = ((img[0] + 1) * 127.5).astype(np.uint8)
        
        # Convert to PIL Image
        pil_img = Image.fromarray(img_array, mode='L')
        
        # Convert to base64
        buffer = io.BytesIO()
        pil_img.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        image_list.append(f'data:image/png;base64,{img_base64}')
    
    return image_list

@app.route('/api/checkpoints', methods=['GET'])
def list_checkpoints():
    checkpoint_dir = '../checkpoints'
    if not os.path.exists(checkpoint_dir):
        return jsonify({'checkpoints': []})
    
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    return jsonify({'checkpoints': checkpoints})

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5000)