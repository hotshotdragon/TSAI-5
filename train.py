import torch
from mnist_model import ComplexMNISTNet, train_model
from datetime import datetime
import os

def main():
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Train the model
    model = train_model()
    
    # Save the model with timestamp and accuracy
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Get the final accuracy from the model training
    dummy_input = torch.randn(1, 1, 28, 28)
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
        # Just to verify model is working
        assert output.shape == (1, 10), "Model output shape is incorrect"
    
    model_path = f'models/mnist_model_{timestamp}.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'timestamp': timestamp,
        'architecture': 'ComplexMNISTNet'
    }, model_path)
    
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main() 