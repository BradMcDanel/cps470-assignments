"""Verify your submission before submitting."""
import torch
from pathlib import Path
from model import create_model

PARAM_LIMIT = 1_000_000


def check_model(checkpoint_path):
    """Check that a checkpoint loads, has <=1M params, and generates text."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Reconstruct model from saved config
    model = create_model(ckpt["vocab_size"], **ckpt["model_kwargs"])
    params = sum(p.numel() for p in model.parameters())
    config = ckpt["model_kwargs"]
    print(f"  Config: {config}")
    print(f"  Parameters: {params:,}")

    if params > PARAM_LIMIT:
        print(f"  ERROR: Model exceeds {PARAM_LIMIT:,} parameter limit!")
        return False

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Check forward pass shape
    vocab_size = ckpt["vocab_size"]
    x = torch.tensor([[0, 1, 2]])  # (1, 3)
    logits, hidden = model(x)
    if logits.shape != (1, 3, vocab_size):
        print(f"  ERROR: Expected logits shape (1, 3, {vocab_size}), got {logits.shape}")
        return False

    # Check that generation works (single step)
    x = torch.tensor([[0]])
    logits, hidden = model(x, hidden)
    probs = torch.softmax(logits[0, -1], dim=0)
    idx = torch.multinomial(probs, 1).item()
    idx_to_char = {int(k): v for k, v in ckpt["idx_to_char"].items()}
    ch = idx_to_char[idx]
    print(f"  Generated character: {ch!r}")
    print(f"  OK")
    return True


def main():
    models_dir = Path("models")
    if not models_dir.exists():
        print("ERROR: No models/ directory found")
        return

    pt_files = list(models_dir.glob("*.pt"))
    if not pt_files:
        print("ERROR: No .pt files found in models/")
        return

    print(f"Found {len(pt_files)} checkpoint(s):\n")
    all_ok = True
    for pt in sorted(pt_files):
        print(f"{pt.name}:")
        if not check_model(pt):
            all_ok = False
        print()

    if all_ok:
        print("All checks passed.")
    else:
        print("Some checks FAILED. Fix the errors above before submitting.")


if __name__ == "__main__":
    main()
