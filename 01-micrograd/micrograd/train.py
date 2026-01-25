"""
Train a simple neural network using MicroGrad.

This script demonstrates that your autograd implementation can be used
to train a real (tiny) neural network.
"""

import random
from micrograd import Value


class Neuron:
    """A single neuron with weights, bias, and optional ReLU activation."""

    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0.0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]


class Layer:
    """A layer of neurons."""

    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    """A multi-layer perceptron (fully connected neural network)."""

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [
            Layer(sz[i], sz[i + 1], nonlin=(i != len(nouts) - 1))
            for i in range(len(nouts))
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0


def main():
    # Set seed for reproducibility
    random.seed(42)

    # Create a simple dataset
    # Inputs: 2D points, Outputs: +1 or -1
    X = [
        [2.0, 3.0],
        [3.0, -1.0],
        [-1.0, -2.0],
        [-2.0, 1.0],
    ]
    y = [1.0, -1.0, 1.0, -1.0]  # Labels

    # Create a small MLP: 2 inputs -> 8 hidden -> 1 output
    model = MLP(2, [8, 1])
    print(f"Number of parameters: {len(model.parameters())}")

    # Training loop
    learning_rate = 0.05
    n_epochs = 100

    print("\nTraining...")
    print("-" * 40)

    for epoch in range(n_epochs):
        # Forward pass: compute predictions and loss
        predictions = [model(x) for x in X]

        # MSE loss
        loss = sum((pred - yi) ** 2 for pred, yi in zip(predictions, y))

        # Backward pass
        model.zero_grad()
        loss.backward()

        # Update weights (gradient descent)
        for p in model.parameters():
            p.data -= learning_rate * p.grad

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Loss: {loss.data:.6f}")

    print("-" * 40)
    print("\nFinal predictions:")
    for x, yi in zip(X, y):
        pred = model(x)
        print(f"  Input: {x} | Target: {yi:+.0f} | Prediction: {pred.data:+.4f}")


def compare_with_pytorch():
    """
    Compare MicroGrad with PyTorch on the same computation.
    Uncomment this section after completing your implementation.
    """
    import torch

    print("\n" + "=" * 50)
    print("Comparing with PyTorch...")
    print("=" * 50)

    # Simple computation to compare gradients
    print("\nSimple gradient comparison:")

    # MicroGrad
    a = Value(2.0)
    b = Value(3.0)
    c = a * b + a ** 2
    c.backward()
    print(f"MicroGrad: c = {c.data}, da = {a.grad}, db = {b.grad}")

    # PyTorch
    a_pt = torch.tensor(2.0, requires_grad=True)
    b_pt = torch.tensor(3.0, requires_grad=True)
    c_pt = a_pt * b_pt + a_pt ** 2
    c_pt.backward()
    print(f"PyTorch:   c = {c_pt.item()}, da = {a_pt.grad.item()}, db = {b_pt.grad.item()}")

    # Check if they match
    if abs(c.data - c_pt.item()) < 1e-6 and \
       abs(a.grad - a_pt.grad.item()) < 1e-6 and \
       abs(b.grad - b_pt.grad.item()) < 1e-6:
        print("\nGradients match!")
    else:
        print("\nGradients DO NOT match. Check your implementation.")


if __name__ == "__main__":
    main()

    # Uncomment the line below after completing your implementation
    # compare_with_pytorch()
