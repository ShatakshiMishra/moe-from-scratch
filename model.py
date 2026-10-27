"""
model.py — a small decoder-only transformer (GPT) whose FFNs can be MoE layers.

    tokens -> tok_emb + pos_emb -> [Block] x n_layer -> LayerNorm -> lm_head

    Block:  x = x + Attn(LN(x))          (causal self-attention)
            x = x + FFN_or_MoE(LN(x))     (the only part this project changes)

Everything except the FFN is identical between the dense baseline and the MoE
variants, so differences in the ablations come from routing alone.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from moe import FFN, MoELayer


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.0):
        super().__init__()
        assert d_model % n_head == 0
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.n_head = n_head
        self.dropout = dropout

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        # [B, T, C] -> [B, n_head, T, head_dim]
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class Block(nn.Module):
    def __init__(self, cfg: dict, use_moe: bool):
        super().__init__()
        d = cfg["d_model"]
        self.ln1 = nn.LayerNorm(d)
        self.attn = CausalSelfAttention(d, cfg["n_head"], cfg["dropout"])
        self.ln2 = nn.LayerNorm(d)
        if use_moe:
            self.ffn = MoELayer(
                d, cfg["expert_hidden"], cfg["n_experts"], cfg["top_k"], cfg["router"],
                cfg["capacity_factor"], cfg["normalize_gates"], cfg["ec_group"], cfg["dropout"],
            )
        else:
            # Dense baseline: one FFN as wide as the k experts a token activates,
            # so FLOPs per token match the MoE models.
            self.ffn = FFN(d, cfg["dense_hidden"], cfg["dropout"])
        self.is_moe = use_moe

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        h = self.ffn(self.ln2(x))
        aux = {}
        if self.is_moe:
            h, aux = h
        return x + h, aux


class GPT(nn.Module):
    def __init__(self, cfg: dict, vocab_size: int):
        super().__init__()
        self.cfg = cfg
        moe = cfg["router"] != "dense"
        self.tok_emb = nn.Embedding(vocab_size, cfg["d_model"])
        self.pos_emb = nn.Embedding(cfg["block_size"], cfg["d_model"])
        self.blocks = nn.ModuleList(
            Block(cfg, use_moe=moe and (i % cfg["moe_every"] == cfg["moe_every"] - 1))
            for i in range(cfg["n_layer"])
        )
        self.ln_f = nn.LayerNorm(cfg["d_model"])
        self.lm_head = nn.Linear(cfg["d_model"], vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def moe_layers(self):
        return [b.ffn for b in self.blocks if b.is_moe]

    def param_counts(self):
        """(total, active-per-token). Active counts k experts per MoE layer, not E."""
        total = sum(p.numel() for p in self.parameters())
        inactive = 0
        for layer in self.moe_layers():
            inactive += layer.params_per_expert() * (layer.n_experts - self.cfg["top_k"])
        return total, total - inactive

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)

        aux = {"balance": x.new_zeros(()), "z": x.new_zeros(())}
        for block in self.blocks:
            x, a = block(x)
            for key, v in a.items():
                aux[key] = aux[key] + v  # summed over MoE layers (Switch convention)

        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss, aux

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8):
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.cfg["block_size"]:]
            logits, _, _ = self(ctx)
            probs = (logits[:, -1] / temperature).softmax(-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx
