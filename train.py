import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model.network import SimpleCNN
from datetime import datetime
import os
from tqdm import tqdm
import torchvision.utils as vutils
import matplotlib.pyplot as plt

def save_augmentation_examples(train_transform, num_examples=5):
    """Save examples of original and augmented images"""
    # Create directory for visualization
    os.makedirs('visualization', exist_ok=True)
    
    # Basic transform for original images
    basic_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load some training examples
    dataset = datasets.MNIST('data', train=True, download=True, transform=None)
    
    plt.figure(figsize=(10, 4))
    for idx in range(num_examples):
        # Get original image
        img, label = dataset[idx]
        
        # Create original and augmented versions
        orig_img = basic_transform(img)
        aug_img = train_transform(img)
        
        # Plot original
        plt.subplot(2, num_examples, idx + 1)
        plt.imshow(orig_img.squeeze(), cmap='gray')
        plt.axis('off')
        plt.title(f'Original {label}')
        
        # Plot augmented
        plt.subplot(2, num_examples, num_examples + idx + 1)
        plt.imshow(aug_img.squeeze(), cmap='gray')
        plt.axis('off')
        plt.title(f'Augmented {label}')
    
    plt.tight_layout()
    plt.savefig('visualization/augmentation_examples.png')
    plt.close()

def evaluate(model, device, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

def train():
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Reduced augmentations to help achieve higher training accuracy faster
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),  # Reduced rotation
        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),  # Reduced translation
            scale=(0.95, 1.05),      # Reduced scaling
        ),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Save examples of augmented images
    save_augmentation_examples(train_transform)
    
    # Load train and test datasets
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=test_transform)
    
    # Increased batch size further
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # Initialize model
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    # Increased learning rate further
    optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.9)
    
    # Print total number of parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")
    
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss = 0.9 * running_loss + 0.1 * loss.item()
            pbar.set_postfix({'loss': f'{running_loss:.4f}'})
        
        # Evaluate after each epoch
        train_accuracy = evaluate(model, device, train_loader)
        test_accuracy = evaluate(model, device, test_loader)
        print(f"\nEpoch {epoch+1}")
        print(f"Training Accuracy: {train_accuracy:.2f}%")
        print(f"Test Accuracy: {test_accuracy:.2f}%")
        
        # Early stopping if we achieve our target
        if train_accuracy >= 95.0:
            print(f"\nReached target training accuracy of 95%+ in epoch {epoch+1}")
            break
    
    # Save model with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not os.path.exists('models'):
        os.makedirs('models')
    torch.save(model.state_dict(), f'models/model_{timestamp}.pth')
    
    # Final evaluation
    final_accuracy = evaluate(model, device, test_loader)
    print(f"\nFinal Test Accuracy: {final_accuracy:.2f}%")

if __name__ == "__main__":
    train() 