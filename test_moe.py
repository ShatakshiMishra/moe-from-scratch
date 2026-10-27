"""
test_moe.py — check the MoE layer against slow, obviously-correct references.

Run:  ../04-diffusion/.venv/bin/python test_moe.py
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from moe import MoELayer, balance_loss
from model import GPT
from train import load_config

torch.manual_seed(0)
D, H, E, K = 16, 32, 4, 2


def expert(layer, e, x):
    """Run expert e on x the slow way (one row at a time is fine)."""
    h = F.gelu(x @ layer.w1[e] + layer.b1[e, 0])
    return h @ layer.w2[e] + layer.b2[e, 0]


def test_topk_matches_reference():
    layer = MoELayer(D, H, E, K, router="topk")
    x = torch.randn(2, 5, D)
    out, _ = layer(x)
    xf = x.reshape(-1, D)
    probs = layer.router.w(xf).softmax(-1)
    topv, topi = probs.topk(K, -1)
    gates = topv / topv.sum(-1, keepdim=True)
    ref = torch.stack([
        sum(gates[n, j] * expert(layer, topi[n, j], xf[n]) for j in range(K))
        for n in range(xf.shape[0])
    ])
    assert torch.allclose(out.reshape(-1, D), ref, atol=1e-5), "top-k output != reference"
    print("ok  top-k softmax output matches per-token reference")


def test_capacity_drops():
    layer = MoELayer(D, H, E, K, router="topk", capacity_factor=0.5)
    # Make every token prefer expert 0 so capacity must bind.
    with torch.no_grad():
        layer.router.w.weight.zero_()
        layer.router.w.weight[0] = 1.0
    x = torch.randn(1, 40, D).abs()  # positive inputs -> expert 0 wins for every token
    plan = layer.router(x.reshape(-1, D), group_size=40)
    C = math.ceil(0.5 * 40 * K / E)
    kept = plan.gate > 0
    per_expert = torch.bincount(plan.exp[kept], minlength=E)
    assert plan.capacity == C and (per_expert <= C).all(), per_expert
    assert plan.stats["dropped"] > 0
    # Dropped assignments live in the trash slot C and contribute nothing.
    assert (plan.slot[~kept] == C).all()
    print(f"ok  capacity C={C} enforced, {int(plan.stats['dropped'])} assignments dropped")


def test_expert_choice_matches_reference():
    layer = MoELayer(D, H, E, K, router="expert_choice", capacity_factor=1.0, ec_group="batch")
    x = torch.randn(2, 6, D)
    out, _ = layer(x)
    xf = x.reshape(-1, D)
    N = xf.shape[0]
    cap = int(N * K / E)
    probs = layer.router.w(xf).softmax(-1)
    ref = torch.zeros_like(xf)
    for e in range(E):
        g, idx = probs[:, e].topk(cap)  # expert e picks its favourite tokens
        for gi, n in zip(g, idx):
            ref[n] += gi * expert(layer, e, xf[n])
    assert torch.allclose(out.reshape(-1, D), ref, atol=1e-5), "expert-choice output != reference"
    assert set(layer.last_stats["load"].tolist()) == {float(cap)}, "EC load must be perfectly equal"
    print(f"ok  expert-choice output matches reference; every expert processed exactly {cap} tokens")


def test_balance_loss_floor():
    N = 64
    probs = torch.full((N, E), 1.0 / E)
    topi = torch.arange(N * K).view(N, K) % E  # perfectly spread assignments
    assert abs(balance_loss(probs, topi, E).item() - 1.0) < 1e-6
    collapsed = torch.zeros(N, E)
    collapsed[:, 0] = 1.0
    assert balance_loss(collapsed, torch.zeros(N, 1, dtype=torch.long), E).item() == E
    print("ok  balance loss = 1.0 when uniform, = E when fully collapsed")


def test_causality():
    base = load_config("configs/base.json", [])
    base.update(n_layer=2, d_model=32, n_head=2, expert_hidden=32, dense_hidden=64, block_size=32)
    x = torch.randint(65, (4, 32))
    x2 = x.clone()
    x2[:, 16:] = torch.randint(65, (4, 16))
    for router, cf, expect_causal in [("topk", None, True), ("expert_choice", 1.0, False)]:
        torch.manual_seed(0)
        m = GPT({**base, "router": router, "capacity_factor": cf}, 65).eval()
        with torch.no_grad():
            d = (m(x)[0][:, :16] - m(x2)[0][:, :16]).abs().max().item()
        assert (d < 1e-5) == expect_causal, (router, d)
        print(f"ok  {router:13s} past logits change by {d:.2e} when the future changes "
              f"({'causal' if expect_causal else 'LEAKS future'})")


if __name__ == "__main__":
    test_topk_matches_reference()
    test_capacity_drops()
    test_expert_choice_matches_reference()
    test_balance_loss_floor()
    test_causality()
    print("all tests passed")
