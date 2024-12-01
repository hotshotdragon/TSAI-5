# MNIST Model Training with CI/CD

[![ML Pipeline](https://github.com/{username}/TSAI_5/actions/workflows/ml_pipeline.yml/badge.svg)](https://github.com/{username}/TSAI_5/actions/workflows/ml_pipeline.yml)

This project implements a CNN model for MNIST digit classification with a complete CI/CD pipeline using GitHub Actions. The model is designed to achieve high accuracy while maintaining efficient training on CPU environments.

## Project Structure 
TSAI_5/
├── .github/
│ └── workflows/
│ └── ml_pipeline.yml
├── models/ # Created during training
├── mnist_model.py # Model architecture and training
├── train.py # Training script
├── test_model.py # Testing suite
├── requirements.txt # Dependencies
└── README.md

## Features

- CNN model for MNIST digit classification
- Automated testing and validation
- CPU-compatible training
- Model versioning with timestamps
- GitHub Actions integration
- Test coverage reporting

## Requirements

- Python 3.8+
- PyTorch (CPU version)
- pytest for testing
- Other dependencies listed in requirements.txt

## Local Setup

1. Create and activate virtual environment:
bash
python -m venv venv
On Windows:
venv\Scripts\activate
On Unix or MacOS:
source venv/bin/activate

2. Install dependencies:
bash
pip install -r requirements.txt

3. Run tests:
bash
pytest test_model.py -v --cov=.

4. Train model:

bash
python train.py


## Model Architecture

The ComplexMNISTNet architecture includes:
- Multiple convolutional layers with batch normalization
- Dropout for regularization
- Fully connected layers
- Input shape: [batch_size, 1, 28, 28]
- Output shape: [batch_size, 10]

## CI/CD Pipeline

The GitHub Actions workflow:
1. Runs on push to main and pull requests
2. Sets up Python environment
3. Installs CPU-only dependencies
4. Runs automated tests
5. Performs training validation
6. Saves model artifacts
7. Generates test coverage report

## Testing

Tests include:
- Model output shape verification
- Multi-batch size compatibility
- Inference mode testing
- Basic training functionality

## Model Deployment

Models are saved with timestamps in the `models/` directory:
models/mnist_model_YYYYMMDD_HHMMSS.pth


## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Notes

- Training is optimized for CPU environments
- Model artifacts are retained for 5 days in GitHub Actions
- Test coverage reports are generated automatically
- Batch size restrictions apply during training (minimum batch size: 4)

## Image Augmentation

The model uses several augmentation techniques:
- Random rotation (±15 degrees)
- Random affine transformations
- Random perspective
- Random erasing

Sample augmented images are saved in the `augmented_samples` directory during training.
