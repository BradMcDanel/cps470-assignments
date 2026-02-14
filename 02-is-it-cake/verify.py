"""Verify your submission before submitting."""
import torch
from model import create_model

model = create_model()
params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {params:,}")

if params > 1_000_000:
    print("ERROR: Model exceeds 1M parameter limit!")
    exit(1)

model.load_state_dict(torch.load("model.pth", weights_only=True))
model.eval()

x = torch.randn(1, 3, 64, 64)
y = model(x)

if y.shape != (1, 2):
    print(f"ERROR: Expected output shape (1, 2), got {y.shape}")
    exit(1)

print("Submission looks valid.")
