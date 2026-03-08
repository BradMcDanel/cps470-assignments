"""Generate text from a trained character-level language model."""
import argparse
import torch
from model import create_model


def generate(model, char_to_idx, idx_to_char, prompt, length=500,
             temperature=1.0, device="cpu"):
    """Generate text continuation from a prompt.

    Feeds the prompt through the model one character at a time to build up
    hidden state, then samples new characters autoregressively.
    """
    model.eval()
    hidden = None

    # Feed prompt through model to build hidden state
    for ch in prompt:
        idx = char_to_idx.get(ch, 0)
        x = torch.tensor([[idx]], dtype=torch.long, device=device)
        _, hidden = model(x, hidden)

    # Sample new characters starting from the last prompt character
    result = list(prompt)
    idx = char_to_idx.get(prompt[-1], 0)

    with torch.no_grad():
        for _ in range(length):
            x = torch.tensor([[idx]], dtype=torch.long, device=device)
            logits, hidden = model(x, hidden)
            logits = logits[0, -1] / temperature
            probs = torch.softmax(logits, dim=0)
            idx = torch.multinomial(probs, 1).item()
            result.append(idx_to_char[idx])

    return ''.join(result)


def load_checkpoint(path, device="cpu"):
    """Load a checkpoint and return (model, char_to_idx, idx_to_char)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = create_model(ckpt["vocab_size"], **ckpt["model_kwargs"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    char_to_idx = ckpt["char_to_idx"]
    idx_to_char = {int(k): v for k, v in ckpt["idx_to_char"].items()}
    return model, char_to_idx, idx_to_char


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--prompt", type=str, default="T",
                        help="Seed text (single character or multi-word prompt)")
    parser.add_argument("--length", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    model, char_to_idx, idx_to_char = load_checkpoint(args.checkpoint, args.device)

    text = generate(model, char_to_idx, idx_to_char, args.prompt,
                    length=args.length, temperature=args.temperature,
                    device=args.device)
    print(text)


if __name__ == "__main__":
    main()
