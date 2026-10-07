"""
train_shakespeare.py
TinyShakespeare corpus par GPT train karta hai.
Usage: python train_shakespeare.py
Corpus data/input.txt mein download hota hai (pehli baar), phir 500+ epochs training.
"""
import os
import urllib.request
import torch

from model.gpt import GPT
from generate import Solution as GenerateSolution

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = "data/input.txt"


def download_corpus():
    if not os.path.exists(DATA_PATH):
        print("Downloading TinyShakespeare corpus...")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return f.read()


def main():
    torch.manual_seed(0)

    text = download_corpus()
    print(f"Corpus loaded: {len(text):,} chars")

    # char-level vocab
    chars = sorted(set(text))
    vocab_size = len(chars)
    char_to_int = {c: i for i, c in enumerate(chars)}
    int_to_char = {i: c for i, c in enumerate(chars)}
    data = torch.tensor([char_to_int[c] for c in text], dtype=torch.long)
    print(f"Vocab size: {vocab_size}")

    # hyperparams (phone/CPU friendly)
    context_length = 128
    model_dim = 192
    num_blocks = 4
    num_heads = 4
    batch_size = 32
    epochs = 500
    lr = 3e-4

    model = GPT(vocab_size, context_length, model_dim, num_blocks, num_heads)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    print("\nTraining start...")
    for epoch in range(epochs):
        starts = torch.randint(0, len(data) - context_length - 1, (batch_size,))
        X = torch.stack([data[s:s + context_length] for s in starts])
        Y = torch.stack([data[s + 1:s + 1 + context_length] for s in starts])

        logits = model(X)
        B, T, C = logits.shape
        loss = torch.nn.functional.cross_entropy(logits.reshape(B * T, C), Y.reshape(B * T))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1:4d}/{epochs} | loss: {loss.item():.4f}")

    print(f"\nFinal loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "gpt_shakespeare.pt")
    print("Checkpoint saved: gpt_shakespeare.pt")

    # generation test
    model.eval()
    starter = torch.zeros((1, 1), dtype=torch.long)
    sample = GenerateSolution().generate(model, 500, starter, context_length, int_to_char)
    print("\n=== Generated sample ===")
    print(sample)


if __name__ == "__main__":
    main()
