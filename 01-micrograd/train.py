"""
Linear Regression with Gradient Descent

This script demonstrates training a linear regression model using PyTorch.
Your task is to implement the same training loop using your MicroGrad Value class.
"""

import torch
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler


def load_data():
    """Load and preprocess the California Housing dataset."""
    # Load dataset
    housing = fetch_california_housing()
    X, y = housing.data, housing.target

    # Use only first 2 features for simplicity (MedInc and HouseAge)
    X = X[:, :2]

    # Use a subset for faster training
    X, y = X[:500], y[:500]

    # Normalize features and targets
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X = scaler_X.fit_transform(X)
    y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    return X, y


def pytorch_training():
    """Train linear regression using PyTorch."""
    print("=" * 50)
    print("PyTorch Linear Regression")
    print("=" * 50)

    # Load data
    X_np, y_np = load_data()
    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)

    # Initialize weights and bias
    torch.manual_seed(42)
    w = torch.randn(2, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    # Training hyperparameters
    learning_rate = 0.1
    n_epochs = 100

    print(f"\nDataset size: {len(X)}")
    print(f"Features: 2 (MedInc, HouseAge)")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {n_epochs}")
    print("\nTraining...")
    print("-" * 40)

    for epoch in range(n_epochs):
        # Forward pass: y_pred = X @ w + b
        y_pred = X @ w + b

        # MSE loss
        loss = ((y_pred - y) ** 2).mean()

        # Backward pass
        loss.backward()

        # Update weights (gradient descent)
        with torch.no_grad():
            w -= learning_rate * w.grad
            b -= learning_rate * b.grad

        # Zero gradients
        w.grad.zero_()
        b.grad.zero_()

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Loss: {loss.item():.6f}")

    print("-" * 40)
    print(f"\nFinal weights: w = [{w[0].item():.4f}, {w[1].item():.4f}], b = {b.item():.4f}")
    print(f"Final loss: {loss.item():.6f}")

    return w.detach().numpy(), b.item(), loss.item()


def micrograd_training():
    """
    TODO: Implement linear regression training using your MicroGrad Value class.

    Use the same dataset, hyperparameters, and random seed as the PyTorch version.
    Your final loss should match PyTorch's final loss (up to small numerical differences).

    Hints:
    - Use load_data() to get the same dataset
    - Initialize weights the same way (use random.seed(42) and random.gauss(0, 1))
    - The forward pass computes: y_pred = x1*w1 + x2*w2 + b for each sample
    - MSE loss is: sum((y_pred - y_true)^2) / n_samples
    - After loss.backward(), update each weight: w.data -= learning_rate * w.grad
    - Don't forget to zero gradients before each backward pass!
    """
    from micrograd import Value
    import random

    print("\n" + "=" * 50)
    print("MicroGrad Linear Regression")
    print("=" * 50)

    # Load data
    X_np, y_np = load_data()

    # TODO: Initialize weights and bias using random.seed(42) and random.gauss(0, 1)
    # random.seed(42)
    # w1 = Value(...)
    # w2 = Value(...)
    # b = Value(...)

    # TODO: Training loop
    # learning_rate = 0.1
    # n_epochs = 100
    # for epoch in range(n_epochs):
    #     # Forward pass: compute predictions for all samples
    #     # Compute MSE loss
    #     # Backward pass
    #     # Update weights
    #     # Zero gradients

    print("\nTODO: Implement this function!")
    print("Your implementation should produce the same loss as PyTorch.")


if __name__ == "__main__":
    # Run PyTorch version (reference implementation)
    pytorch_training()

    # Run your MicroGrad version
    micrograd_training()
