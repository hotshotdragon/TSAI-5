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
    batch_sizes = [4, 8, 32]
    for batch_size in batch_sizes:
        x = torch.randn(batch_size, 1, 28, 28)
        output = model(x)
        assert output.shape == (batch_size, 10), f"Failed for batch size {batch_size}"

def test_inference_mode():
    model = ComplexMNISTNet()
    model.eval()
    # Now we can test with batch_size=1
    x = torch.randn(1, 1, 28, 28)
    with torch.no_grad():
        output = model(x)
    assert output.shape == (1, 10), "Failed for single sample inference"

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

def test_model_gradient_flow():
    """Test if gradients are flowing properly through the model"""
    model = ComplexMNISTNet()
    x = torch.randn(4, 1, 28, 28)
    y = torch.randint(0, 10, (4,))
    
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    
    # Initial backward pass
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    
    # Check if gradients exist and are not zero
    has_grad = False
    has_non_zero_grad = False
    
    for param in model.parameters():
        if param.requires_grad and param.grad is not None:
            has_grad = True
            if torch.sum(torch.abs(param.grad)) > 0:
                has_non_zero_grad = True
                break
    
    assert has_grad and has_non_zero_grad, "Model gradients are not flowing properly"

def test_model_overfitting_single_batch():
    """Test if model can overfit to a single batch (sanity check)"""
    model = ComplexMNISTNet()
    x = torch.randn(4, 1, 28, 28)
    y = torch.randint(0, 10, (4,))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Try to overfit
    initial_loss = None
    final_loss = None
    
    for _ in range(10):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        if initial_loss is None:
            initial_loss = loss.item()
        loss.backward()
        optimizer.step()
        final_loss = loss.item()
    
    assert final_loss < initial_loss, "Model is not able to overfit to a single batch"

def test_model_batch_norm_behavior():
    """Test if BatchNorm layers behave differently in train and eval modes"""
    model = ComplexMNISTNet()
    x = torch.randn(4, 1, 28, 28)
    
    # Training mode
    model.train()
    out_train = model(x)
    
    # Eval mode
    model.eval()
    out_eval = model(x)
    
    # Outputs should be different due to BatchNorm behavior
    assert not torch.allclose(out_train, out_eval), "BatchNorm layers are not functioning properly"

def test_model_activation_ranges():
    """Test if activations are in reasonable ranges"""
    model = ComplexMNISTNet()
    x = torch.randn(4, 1, 28, 28)
    
    model.eval()
    with torch.no_grad():
        output = model(x)
    
    # Log softmax output should be negative
    assert torch.all(output <= 0), "Log softmax outputs are not all negative"
    
    # Probabilities (exp of log softmax) should sum to 1
    probs = torch.exp(output)
    sums = torch.sum(probs, dim=1)
    assert torch.allclose(sums, torch.ones_like(sums)), "Probabilities don't sum to 1" 