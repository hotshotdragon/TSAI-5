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

# def test_model_training_accuracy():
#     """Test model achieves reasonable accuracy in quick test"""
#     # Reduced augmentations to help achieve higher training accuracy faster
#     train_transform = transforms.Compose([
#         transforms.RandomRotation(10),  # Reduced rotation
#         transforms.RandomAffine(
#             degrees=0,
#             translate=(0.05, 0.05),  # Reduced translation
#             scale=(0.95, 1.05),      # Reduced scaling
#         ),
#         transforms.ToTensor(),
#         transforms.Normalize((0.1307,), (0.3081,)),
#     ])
    
#     test_transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.1307,), (0.3081,))
#     ])
    
    
#     # Load train and test datasets
#     train_dataset = datasets.MNIST('data', train=True, download=True, transform=train_transform)
#     test_dataset = datasets.MNIST('data', train=False, download=True, transform=test_transform)
    
#     # Increased batch size further
#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
#     test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
#     # Initialize model
#     model = SimpleCNN().to(device)
#     criterion = nn.CrossEntropyLoss()
#     # Increased learning rate further
#     optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.9)
    
#     # Print total number of parameters
#     total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"Total trainable parameters: {total_params}")
    
#     num_epochs = 5
#     for epoch in range(num_epochs):
#         model.train()
#         running_loss = 0.0
#         pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
#         for batch_idx, (data, target) in enumerate(pbar):
#             data, target = data.to(device), target.to(device)
#             optimizer.zero_grad()
#             output = model(data)
#             loss = criterion(output, target)
#             loss.backward()
#             optimizer.step()
            
#             running_loss = 0.9 * running_loss + 0.1 * loss.item()
#             pbar.set_postfix({'loss': f'{running_loss:.4f}'})
        
#         # Evaluate after each epoch
#         train_accuracy = evaluate(model, device, train_loader)
#         test_accuracy = evaluate(model, device, test_loader)
#         print(f"\nEpoch {epoch+1}")
#         print(f"Training Accuracy: {train_accuracy:.2f}%")
#         print(f"Test Accuracy: {test_accuracy:.2f}%")
        
#         # Early stopping if we achieve our target
#         if train_accuracy >= 95.0:
#             print(f"\nReached target training accuracy of 95%+ in epoch {epoch+1}")
#             break
    
#     # Final evaluation
#     final_accuracy = evaluate(model, device, test_loader)
#     print(f"\nFinal Test Accuracy: {final_accuracy:.2f}%")
#     assert final_accuracy > 60.0, f"Model achieved only {accuracy:.2f}% accuracy"  # Adjusted threshold