"""
data.py — character-level TinyShakespeare, same encoding as 01-transformer.

90% train / 10% val split on the raw character stream. Batches are random
windows of length block_size; targets are the same window shifted by one.
"""

from __future__ import annotations

import torch


class CharData:
    def __init__(self, path: str = "input.txt", val_frac: float = 0.1):
        # 1. Raw text -> sorted character vocabulary
        with open(path, "r") as f:
            text = f.read()
        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for i, c in enumerate(self.chars)}

        # 2. Encode everything once, split train/val
        data = torch.tensor(self.encode(text), dtype=torch.long)
        n = int((1 - val_frac) * len(data))
        self.train, self.val = data[:n], data[n:]

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)

    def batch(self, split: str, batch_size: int, block_size: int, device, generator=None):
        d = self.train if split == "train" else self.val
        ix = torch.randint(len(d) - block_size - 1, (batch_size,), generator=generator)
        x = torch.stack([d[i : i + block_size] for i in ix])
        y = torch.stack([d[i + 1 : i + 1 + block_size] for i in ix])
        return x.to(device), y.to(device)

    def fixed_val_batches(self, n_batches: int, batch_size: int, block_size: int, device, seed: int = 1234):
        """Same val windows for every run, so perplexities are directly comparable."""
        g = torch.Generator().manual_seed(seed)
        return [self.batch("val", batch_size, block_size, device, g) for _ in range(n_batches)]


if __name__ == "__main__":
    d = CharData()
    print("vocab_size:", d.vocab_size)
    print("train/val chars:", len(d.train), len(d.val))
    x, y = d.batch("train", 2, 16, "cpu")
    print("x:", repr(d.decode(x[0])))
    print("y:", repr(d.decode(y[0])))
