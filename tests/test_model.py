import torch
import pytest
from model.network import SimpleCNN
from torchvision import transforms, datasets
import os
import time

@pytest.fixture
def model():
    return SimpleCNN()

def test_model_output_shape():
    model = SimpleCNN()
    batch_size = 4
    x = torch.randn(batch_size, 1, 28, 28)
    output = model(x)
    assert output.shape == (batch_size, 10), "Output shape is incorrect"

def test_parameter_count():
    """Test that model has less than 25000 parameters"""
    model = SimpleCNN()
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 25000, f"Model has {total_params} parameters, should be less than 25000"

def test_input_size():
    model = SimpleCNN()
    batch_sizes = [4, 8, 32]
    for batch_size in batch_sizes:
        x = torch.randn(batch_size, 1, 28, 28)
        output = model(x)
        assert output.shape == (batch_size, 10), f"Failed for batch size {batch_size}"

def test_basic_training():
    model = SimpleCNN()
    x = torch.randn(32, 1, 28, 28)
    y = torch.randint(0, 10, (32,))
    
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    
    model.train()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    
    assert not torch.isnan(loss), "Training step produced NaN loss"

def test_model_inference_time():
    """Test that model inference is reasonably fast"""
    model = SimpleCNN()
    model.eval()
    x = torch.randn(1, 1, 28, 28)
    
    with torch.no_grad():
        start_time = time.time()
        _ = model(x)
        end_time = time.time()
    
    inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
    assert inference_time < 100, f"Inference too slow: {inference_time:.2f}ms"

def test_model_training_accuracy():
    """Test model achieves reasonable accuracy in quick test"""
    model = SimpleCNN()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load a larger subset of training data
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    subset_size = 5000  # Increased subset size
    indices = torch.randperm(len(train_dataset))[:subset_size]
    subset = torch.utils.data.Subset(train_dataset, indices)
    train_loader = torch.utils.data.DataLoader(subset, batch_size=128)  # Increased batch size
    
    # Train for multiple epochs on subset
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    model.train()
    # Train for more epochs
    for epoch in range(5):  # Increased to 5 epochs
        correct = 0
        total = 0
        
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += len(data)
    
    accuracy = 100. * correct / total
    assert accuracy > 60.0, f"Model achieved only {accuracy:.2f}% accuracy"  # Adjusted threshold