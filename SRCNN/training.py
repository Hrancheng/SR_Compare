# training.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from model import SRCNN
from PIL import Image
from datasets import load_dataset
import os

# Dataset class
class DIV2KDataset(torch.utils.data.Dataset):
    def __init__(self, lr_dir, hr_dir, transform=None):
        self.lr_images = [os.path.join(lr_dir, img) for img in os.listdir(lr_dir)]
        self.hr_images = [os.path.join(hr_dir, img) for img in os.listdir(hr_dir)]
        self.transform = transform

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, idx):
        lr_image = Image.open(self.lr_images[idx]).convert("RGB")
        hr_image = Image.open(self.hr_images[idx]).convert("RGB")

        if self.transform:
            lr_image = self.transform(lr_image)
            hr_image = self.transform(hr_image)

        return lr_image, hr_image

# Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

    # Load dataset
def train_model():
    lr_dir = './lr/images'
    hr_dir = './hr/images'
    dataset = DIV2KDataset(lr_dir=lr_dir, hr_dir=hr_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Model, Loss, and Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SRCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training Loop
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        for lr_images, hr_images in dataloader:
            lr_images, hr_images = lr_images.to(device), hr_images.to(device)

            optimizer.zero_grad()
            outputs = model(lr_images)
            loss = criterion(outputs, hr_images)
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')
