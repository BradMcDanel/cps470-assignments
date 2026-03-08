"""Character-level language model."""
import torch
import torch.nn as nn


def create_model(vocab_size, embed_dim=64, hidden_dim=128, num_layers=1,
                 cell_type="lstm", dropout=0.0):
    """Create a character-level language model.

    Args:
        vocab_size: Number of unique characters in the text.
        embed_dim: Dimension of character embeddings.
        hidden_dim: Dimension of RNN hidden state.
        num_layers: Number of stacked RNN layers.
        cell_type: One of "rnn", "lstm", "gru".
        dropout: Dropout rate between RNN layers (only used if num_layers > 1).

    Returns:
        An nn.Module whose forward(x, hidden) returns (logits, hidden).
        - x: (batch_size, seq_len) integer tensor of character indices
        - logits: (batch_size, seq_len, vocab_size) raw predictions
        - hidden: updated hidden state (pass back in for next step)

    The model must have a `vocab_size` attribute so the training script
    can reshape logits for the loss function.
    """
    raise NotImplementedError("Build your model here")
