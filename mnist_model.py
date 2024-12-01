import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

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

def train_model():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")

        os.makedirs('./data', exist_ok=True)

        # Enhanced data augmentation
        transform = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)  # Reduced batch size

        model = ComplexMNISTNet().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)  # Added weight decay
        criterion = nn.CrossEntropyLoss()
        
        # Modified learning rate scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=0.005,  # Reduced max learning rate
            epochs=1,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,  # Increased warm-up period
            div_factor=10.0,
            final_div_factor=100.0
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params}")

        model.train()
        correct = 0
        total = 0
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc='Training')
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            output = model(data)
            loss = criterion(output, target)
            
            loss.backward()
            # Gradient clipping
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

        final_accuracy = 100. * correct / total
        print(f'\nFinal Training Accuracy: {final_accuracy:.2f}%')
        return model

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    model = train_model() 