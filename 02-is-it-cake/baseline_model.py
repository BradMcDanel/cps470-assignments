"""
Baseline model for "Is It Cake?" classifier.
A minimal linear model to verify your pipeline works.
"""
import torch.nn as nn


def create_model() -> nn.Module:
    """
    Tiny linear model. ~24K parameters.
    """
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 64 * 64, 2)
    )
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    import torch
    model = create_model()
    print(f"Parameters: {count_parameters(model):,}")

    x = torch.randn(1, 3, 64, 64)
    y = model(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")
