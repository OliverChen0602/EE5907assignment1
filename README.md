# EE5907 Assignment 1: Neural Networks & RBF

Binary classification using MLP and RBF networks on a 2D Gaussian dataset.

## Files

- **assignment1.py** - Main implementation with MLP (2-3-1 architecture) and RBF networks
- **Generate_data.py** - Data generation utility
- **Assignment 1.pdf** - Assignment specifications

## Overview

### Part 1: MLP (Multi-Layer Perceptron)
- Architecture: 2 inputs → 3 hidden (ReLU) → 1 output (sigmoid)
- Total parameters: 9 weight connections
- Training: Backpropagation with batch gradient descent
- Features: Data normalization, L2 regularization

### Part 2: RBF Networks (Radial Basis Function)
- Design: Gaussian basis functions as features
- Two variants: M=3 centers and M=6 centers
- Training: Least-squares solution via normal equation

## Results

- **MLP (trained)**: ~99% accuracy
- **RBF (M=3)**: ~58% accuracy
- **RBF (M=6)**: ~60% accuracy

## Usage

```bash
python assignment1.py
```

This runs:
1. Data generation and visualization
2. MLP initial evaluation and training
3. Decision boundary and ROC curve plots
4. RBF networks with different center counts

## Dependencies

- numpy
- matplotlib

## How It Works

### Classification

Models output continuous scores (0-1 via sigmoid). Binary predictions are made using:
```
y_pred = 1 if score ≥ 0.5 else 0
```

Metrics computed: Accuracy, Precision, Recall, AUC-ROC

### Data

- Class 0: 300 samples from N([10, 14], σ=2)
- Class 1: 80 samples from N([14, 18], σ=2)
- Preprocessed: Normalized to zero-mean, unit-variance

## Future Improvements

- K-Means center selection for RBF
- Adaptive sigma tuning
- Cross-validation for hyperparameter optimization
- More RBF centers (M > 6)
