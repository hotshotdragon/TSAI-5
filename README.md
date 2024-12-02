# MNIST Model Training Pipeline

[![Build Status](https://github.com/hotshotdragon/TSAI_5/actions/workflows/ml_pipeline.yml/badge.svg)](https://github.com/hotshotdragon/TSAI_5/actions/workflows/ml_pipeline.yml)

This project implements a CNN model for MNIST digit classification with complete CI/CD pipeline. The model achieves >95% accuracy in one epoch while maintaining less than 25,000 parameters.

## Features

- Efficient CNN architecture (<25k parameters)
- High accuracy (>95%) in single epoch
- Data augmentation with visualizations
- Automated testing and validation
- GitHub Actions integration

## Model Architecture

- Input: 28x28 grayscale images
- 2-block CNN with efficient parameter usage
- Dropout regularization
- Output: 10 classes (digits 0-9)

## Tests

1. Parameter count verification (<25k)
2. Training accuracy check (>95%)
3. Augmentation consistency
4. Model inference speed
5. Input/output shape validation
6. Basic training functionality

## Setup and Usage

1. Create virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run tests:

```bash
pytest test_model.py -v --cov=.
```

4. Train model:

```bash
python train.py
```

## CI/CD Pipeline

The GitHub Actions workflow:
1. Runs all tests
2. Verifies parameter count
3. Checks training accuracy
4. Generates augmentation samples
5. Saves model artifacts
