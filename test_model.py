import torch
import pytest
from mnist_model import ComplexMNISTNet

@pytest.fixture
def model():
    return ComplexMNISTNet()

def test_model_output_shape():
    model = ComplexMNISTNet()
    batch_size = 4
    x = torch.randn(batch_size, 1, 28, 28)
    output = model(x)
    assert output.shape == (batch_size, 10), "Output shape is incorrect"

def test_model_parameters():
    model = ComplexMNISTNet()
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 1000000, "Model has too many parameters"

def test_basic_training():
    model = ComplexMNISTNet()
    x = torch.randn(32, 1, 28, 28)
    y = torch.randint(0, 10, (32,))
    
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    
    model.train()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    
    assert not torch.isnan(loss), "Training step produced NaN loss" 