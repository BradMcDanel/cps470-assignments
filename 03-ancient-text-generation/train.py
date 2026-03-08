"""Train a character-level language model."""
import argparse
import math
import torch
import torch.nn as nn
from pathlib import Path
from model import create_model


def load_text(path):
    """Load text file and build character vocabulary."""
    text = Path(path).read_text()
    chars = sorted(set(text))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    data = torch.tensor([char_to_idx[c] for c in text], dtype=torch.long)
    return data, char_to_idx, idx_to_char


def make_batches(data, seq_len, batch_size):
    """Reshape data into (num_batches, batch_size, seq_len) input/target pairs."""
    # Trim to fit evenly
    n = (len(data) - 1) // (batch_size * seq_len) * (batch_size * seq_len)
    if n == 0:
        raise ValueError("Text too short for these batch parameters")

    inputs = data[:n].view(batch_size, -1, seq_len)
    targets = data[1:n+1].view(batch_size, -1, seq_len)
    # inputs shape: (batch_size, num_batches, seq_len)
    # Transpose to (num_batches, batch_size, seq_len)
    inputs = inputs.transpose(0, 1).contiguous()
    targets = targets.transpose(0, 1).contiguous()
    return inputs, targets


def train_epoch(model, inputs, targets, optimizer, criterion, device, clip=5.0):
    model.train()
    total_loss = 0.0
    total_chars = 0
    hidden = None

    for i in range(inputs.size(0)):
        x = inputs[i].to(device)   # (batch_size, seq_len)
        y = targets[i].to(device)  # (batch_size, seq_len)

        logits, hidden = model(x, hidden)
        # Detach hidden state to prevent backprop through entire history
        if isinstance(hidden, tuple):
            hidden = tuple(h.detach() for h in hidden)
        else:
            hidden = hidden.detach()

        loss = criterion(logits.view(-1, model.vocab_size), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        total_loss += loss.item() * y.numel()
        total_chars += y.numel()

    return total_loss / total_chars


def evaluate(model, inputs, targets, criterion, device):
    model.eval()
    total_loss = 0.0
    total_chars = 0
    hidden = None

    with torch.no_grad():
        for i in range(inputs.size(0)):
            x = inputs[i].to(device)
            y = targets[i].to(device)

            logits, hidden = model(x, hidden)
            loss = criterion(logits.view(-1, model.vocab_size), y.view(-1))

            total_loss += loss.item() * y.numel()
            total_chars += y.numel()

    return total_loss / total_chars


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to text file")
    parser.add_argument("--cell_type", type=str, default="lstm", choices=["rnn", "lstm", "gru"])
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save", type=str, default=None, help="Path to save checkpoint (.pt)")
    args = parser.parse_args()

    # Load and split data
    data, char_to_idx, idx_to_char = load_text(args.data)
    vocab_size = len(char_to_idx)
    split = int(0.8 * len(data))
    train_data = data[:split]
    val_data = data[split:]

    print(f"Text: {args.data}")
    print(f"Total chars: {len(data):,}  Train: {len(train_data):,}  Val: {len(val_data):,}")
    print(f"Vocab size: {vocab_size}")
    print(f"Vocab: {''.join(sorted(char_to_idx.keys()))!r}")
    print()

    # Build batches
    train_inputs, train_targets = make_batches(train_data, args.seq_len, args.batch_size)
    val_inputs, val_targets = make_batches(val_data, args.seq_len, args.batch_size)
    print(f"Train batches: {train_inputs.shape[0]}  Val batches: {val_inputs.shape[0]}")

    # Create model
    model_kwargs = dict(
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        cell_type=args.cell_type,
        dropout=args.dropout,
    )
    model = create_model(vocab_size, **model_kwargs).to(args.device)
    print(f"Model: {args.cell_type.upper()} | Parameters: {model.count_parameters():,}")
    print()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_inputs, train_targets, optimizer, criterion, args.device)
        val_loss = evaluate(model, val_inputs, val_targets, criterion, args.device)

        train_ppl = math.exp(train_loss)
        val_ppl = math.exp(val_loss)

        print(f"Epoch {epoch:3d} | Train loss {train_loss:.4f} (ppl {train_ppl:.1f}) | "
              f"Val loss {val_loss:.4f} (ppl {val_ppl:.1f})")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if args.save:
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "model_kwargs": model_kwargs,
                    "vocab_size": vocab_size,
                    "char_to_idx": char_to_idx,
                    "idx_to_char": idx_to_char,
                }, args.save)

    print(f"\nBest val loss: {best_val_loss:.4f} (ppl {math.exp(best_val_loss):.1f})")

    if args.save:
        print(f"Checkpoint saved to {args.save}")


if __name__ == "__main__":
    main()
