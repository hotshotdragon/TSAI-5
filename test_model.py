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

def test_input_size():
    model = ComplexMNISTNet()
    # Test with different batch sizes
    batch_sizes = [1, 4, 32]
    for batch_size in batch_sizes:
        x = torch.randn(batch_size, 1, 28, 28)
        output = model(x)
        assert output.shape == (batch_size, 10), f"Failed for batch size {batch_size}"

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