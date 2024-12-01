import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import save_image
import random

class ComplexMNISTNet(nn.Module):
    def __init__(self):
        super(ComplexMNISTNet, self).__init__()

        # First convolutional block
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)  # Output: (batch_size, 32, 28, 28)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Output: (batch_size, 64, 28, 28)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=5, padding=2)  # Output: (batch_size, 64, 28, 28)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)  # Output: (batch_size, 64, 14, 14)
        self.dropout1 = nn.Dropout(0.2)

        # Second convolutional block
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # Output: (batch_size, 128, 14, 14)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=5, padding=2)  # Output: (batch_size, 128, 14, 14)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)  # Output: (batch_size, 128, 14, 14)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)  # Output: (batch_size, 128, 7, 7)
        self.dropout2 = nn.Dropout(0.3)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 7 * 7, 256)  # Flattened size to 256
        self.batch_norm_fc = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 10)  # Output: 10 classes

    def forward(self, x):
        # First convolutional block
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool1(x)
        x = self.dropout1(x)

        # Second convolutional block
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool2(x)
        x = self.dropout2(x)

        # Fully connected layers
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.batch_norm_fc(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def save_augmented_samples(dataset, num_samples=5):
    """Save samples of augmented images"""
    os.makedirs('augmented_samples', exist_ok=True)
    
    # Get random samples
    indices = random.sample(range(len(dataset)), num_samples)
    
    # Create a figure
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    
    for idx, sample_idx in enumerate(indices):
        # Get original image (before transforms)
        original_image = transforms.ToTensor()(dataset.dataset.data[sample_idx].numpy())
        
        # Get augmented image
        augmented_image = dataset[sample_idx][0]
        
        # Save individual images
        save_image(original_image, f'augmented_samples/original_{idx}.png')
        save_image(augmented_image, f'augmented_samples/augmented_{idx}.png')
        
        # Display in subplot
        axes[0, idx].imshow(original_image.squeeze(), cmap='gray')
        axes[0, idx].axis('off')
        axes[0, idx].set_title('Original')
        
        axes[1, idx].imshow(augmented_image.squeeze(), cmap='gray')
        axes[1, idx].axis('off')
        axes[1, idx].set_title('Augmented')
    
    plt.tight_layout()
    plt.savefig('augmented_samples/comparison.png')
    plt.close()

def train_model():
    try:
        print(f"PyTorch version: {torch.__version__}")
        os.makedirs('./data', exist_ok=True)

        # Modified transform sequence - ToTensor() must come before other tensor-based transforms
        transform = transforms.Compose([
            transforms.ToTensor(),  # Convert PIL Image to tensor first
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.RandomErasing(p=0.1),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        subset_size = len(train_dataset) // 5
        indices = torch.randperm(len(train_dataset))[:subset_size]
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        # Save augmented samples before training
        save_augmented_samples(train_dataset)

        model = ComplexMNISTNet()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()
        
        # Modified scheduler for 2 epochs
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=0.005,
            epochs=2,  # Changed to 2 epochs
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            div_factor=10.0,
            final_div_factor=100.0
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params}")

        # Training loop for 2 epochs
        for epoch in range(2):  # Run for 2 epochs
            print(f'\nEpoch {epoch+1}/2')
            model.train()
            correct = 0
            total = 0
            running_loss = 0.0
            
            pbar = tqdm(train_loader, desc=f'Training Epoch {epoch+1}')
            
            for batch_idx, (data, target) in enumerate(pbar):
                optimizer.zero_grad()
                
                output = model(data)
                loss = criterion(output, target)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += len(data)
                running_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    current_accuracy = 100. * correct / total
                    current_loss = running_loss / (batch_idx + 1)
                    pbar.set_postfix({
                        'loss': f'{current_loss:.4f}',
                        'accuracy': f'{current_accuracy:.2f}%',
                        'lr': f'{scheduler.get_last_lr()[0]:.6f}'
                    })

            epoch_accuracy = 100. * correct / total
            print(f'Epoch {epoch+1} Accuracy: {epoch_accuracy:.2f}%')

        final_accuracy = epoch_accuracy  # Use the last epoch's accuracy
        print(f'\nFinal Training Accuracy: {final_accuracy:.2f}%')

        return model

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    model = train_model() 