"""
moe.py — a sparse Mixture-of-Experts FFN layer, from scratch.

A dense transformer block runs every token through ONE feed-forward network.
An MoE block owns E feed-forward "experts" and a tiny ROUTER (a single linear
layer d_model -> E) that decides which experts see which tokens. Each token
only pays for the few experts it is sent to, so total parameters grow with E
while compute per token stays roughly constant.

Every MoE layer does the same three things:

    1. ROUTE     router logits -> probabilities -> a dispatch plan: for every
                 token->expert assignment, (token, expert, slot, gate).
    2. DISPATCH  scatter tokens into a fixed-size buffer [E, C, d] (expert e's
                 tokens live in row e, `slot` says where), then run all experts
                 at once with one batched matmul.
    3. COMBINE   read each assignment's output back out of the buffer, scale by
                 its gate and scatter-add it onto the token it came from.

WHY A FIXED BUFFER instead of "loop over experts on variable-length slices"?
Real MoE systems (GShard, Switch) do this for hardware reasons: every expert
is padded to the same capacity C so the computation has a static shape. On
Apple MPS it matters a lot too — variable shapes force a kernel recompile on
nearly every step (we measured ~25x slowdown). A consequence worth noticing:
an expert that receives MORE tokens makes the whole buffer bigger, so load
imbalance directly costs throughput, exactly as it does on a real cluster.

Only step 1 differs between strategies, so the routers below all produce the
same `Dispatch` object and `MoELayer` handles steps 2-3 once.

THE ROUTERS
    TokenChoiceRouter  (Shazeer 2017 / GShard / Switch)
        Each TOKEN picks its top-k experts. Gates = softmax probs of the chosen
        experts (renormalized to sum to 1). Optional capacity factor: an expert
        accepts at most ceil(cf * N * k / E) assignments, the rest are DROPPED
        (that token just skips the expert — the residual carries it).
        "top-k softmax" = this router with aux_coef = 0.
        "top-k + aux"   = this router with the Switch load-balancing loss on.

    ExpertChoiceRouter (Zhou et al. 2022, "Mixture-of-Experts with Expert
        Choice Routing")
        Each EXPERT picks its top-C tokens from a group. Load is perfectly
        balanced by construction, but a token may be picked by 0 experts or by
        all E — and, crucially for a decoder, which tokens an expert picks
        depends on every other token in the group, including FUTURE ones.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


class FFN(nn.Module):
    """Standard transformer MLP: d -> hidden -> d. Used as the dense baseline AND as one expert."""

    def __init__(self, d_model: int, hidden: int, dropout: float = 0.0):
        super().__init__()
        self.fc = nn.Linear(d_model, hidden)
        self.proj = nn.Linear(hidden, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))


@dataclass
class Dispatch:
    """The routing plan. One entry per token->expert assignment (A of them).

    tok[a], exp[a], slot[a] : assignment a copies token tok[a] into buffer
                              position [exp[a], slot[a]]
    gate[a]                 : weight its expert output is multiplied by
                              (0 for assignments dropped by capacity — those
                              are parked in a trash slot C and ignored)
    capacity                : C, the per-expert buffer length
    aux                     : differentiable losses {"balance": ..., "z": ...}
    stats                   : detached diagnostics, accumulated by train.py
    """

    tok: torch.Tensor
    exp: torch.Tensor
    slot: torch.Tensor
    gate: torch.Tensor
    capacity: int
    aux: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)


def balance_loss(probs: torch.Tensor, topi: torch.Tensor, n_experts: int) -> torch.Tensor:
    """Switch-Transformer load-balancing loss, generalized to top-k.

        L = E * sum_e  f_e * P_e

    f_e = fraction of the N*k token->expert assignments that went to expert e
          (a hard count — NOT differentiable)
    P_e = mean router probability for expert e over the N tokens (differentiable)

    The gradient flows only through P_e, and it pushes probability DOWN on
    experts that are over-used (large f_e). Under perfectly uniform routing
    f_e = P_e = 1/E, so L = 1.0 — that is the floor to compare against.
    """
    f = F.one_hot(topi.reshape(-1), n_experts).float().mean(0)
    P = probs.mean(0)
    return n_experts * (f * P).sum()


def router_stats(probs, topi, load, n_dropped, n_unrouted):
    """Detached per-forward diagnostics (summed over a logging window by train.py)."""
    E = probs.shape[-1]
    entropy = -(probs * probs.clamp_min(1e-9).log()).sum(-1).mean()
    return {
        "load": load.detach().float(),  # [E] assignments actually processed
        "pref": F.one_hot(topi.reshape(-1), E).sum(0).float(),  # [E] what plain top-k WOULD send
        "dropped": n_dropped.detach().float(),  # assignments dropped by capacity
        "unrouted": n_unrouted.detach().float(),  # tokens that got zero experts
        "entropy": entropy.detach(),  # mean router entropy (nats), summed per forward
        "n_tokens": torch.tensor(float(probs.shape[0]), device=probs.device),
    }


def _round_up(n: int, m: int) -> int:
    return ((n + m - 1) // m) * m


class TokenChoiceRouter(nn.Module):
    def __init__(self, d_model, n_experts, k=2, capacity_factor=None, normalize_gates=True):
        super().__init__()
        self.w = nn.Linear(d_model, n_experts, bias=False)
        self.E, self.k = n_experts, k
        self.cf = capacity_factor
        self.normalize = normalize_gates

    def forward(self, x: torch.Tensor, group_size: int) -> Dispatch:
        N, E, k = x.shape[0], self.E, self.k

        # 1. Router probabilities over ALL experts (fp32 for a stable softmax).
        logits = self.w(x).float()  # [N, E]
        probs = logits.softmax(-1)

        # 2. Each token picks its top-k experts.
        topv, topi = probs.topk(k, dim=-1)  # [N, k] each
        gates = topv / topv.sum(-1, keepdim=True) if self.normalize else topv

        # 3. Flatten to a list of assignments in PRIORITY order: rank-major
        #    (every token's 1st choice, then every token's 2nd choice, ...),
        #    token order within a rank. This is the GShard convention: a
        #    token's top-1 choice is never bumped by someone else's top-2.
        tok = torch.arange(N, device=x.device).repeat(k)  # [k*N]
        exp = topi.t().reshape(-1)  # [k*N]
        gate = gates.t().reshape(-1)  # [k*N]

        # 4. Slot = position in that expert's queue = running count of earlier
        #    assignments to the same expert.
        onehot = F.one_hot(exp, E)  # [k*N, E]
        slot = (onehot.cumsum(0) * onehot).sum(-1) - 1
        load_pre = onehot.sum(0)  # [E]

        # 5. Capacity. With a capacity factor, expert e keeps only its first
        #    C = ceil(cf * N * k / E) assignments; the rest are DROPPED.
        #    Without one, C is however many the busiest expert received
        #    (rounded up to limit the number of distinct shapes).
        if self.cf is not None:
            C = math.ceil(self.cf * N * k / E)
        else:
            C = _round_up(int(load_pre.max().item()), 64)
        keep = slot < C
        slot = torch.where(keep, slot, torch.full_like(slot, C))  # trash slot
        gate = gate * keep

        load = torch.minimum(load_pre, torch.tensor(C, device=x.device))
        n_unrouted = (~keep.view(k, N).any(0)).sum()
        aux = {
            "balance": balance_loss(probs, topi, E),
            "z": torch.logsumexp(logits, -1).pow(2).mean(),  # ST-MoE router z-loss
        }
        stats = router_stats(probs, topi, load, (~keep).sum(), n_unrouted)
        return Dispatch(tok, exp, slot, gate, C, aux, stats)


class ExpertChoiceRouter(nn.Module):
    """Each expert picks its top-C tokens within a group of n tokens.

    C = capacity_factor * n * k / E, so on AVERAGE each token is processed by k
    experts — same compute as token-choice top-k. The group is either one
    sequence ("sequence") or the whole flattened batch ("batch").
    """

    def __init__(self, d_model, n_experts, k=2, capacity_factor=1.0, group="batch"):
        super().__init__()
        self.w = nn.Linear(d_model, n_experts, bias=False)
        self.E, self.k = n_experts, k
        self.cf = capacity_factor if capacity_factor is not None else 1.0
        self.group = group

    def forward(self, x: torch.Tensor, group_size: int) -> Dispatch:
        N, E, k = x.shape[0], self.E, self.k
        n = N if self.group == "batch" else group_size
        G = N // n

        # 1. Same router as token-choice: softmax over experts for every token.
        logits = self.w(x).float()
        probs = logits.softmax(-1)  # [N, E]

        # 2. Transpose the decision: per group, each expert ranks ALL n tokens
        #    by its affinity probs[token, e] and keeps the top `cap`.
        cap = min(n, max(1, int(self.cf * n * k / E)))
        per_expert = probs.view(G, n, E).permute(0, 2, 1)  # [G, E, n]
        gate, idx = per_expert.topk(cap, dim=-1)  # [G, E, cap]
        idx = idx + (torch.arange(G, device=x.device) * n).view(G, 1, 1)  # -> global token ids

        # 3. Expert-major layout [E, G*cap]: expert e's picks fill row e of the buffer.
        C = G * cap
        tok = idx.permute(1, 0, 2).reshape(-1)
        gate = gate.permute(1, 0, 2).reshape(-1)
        exp = torch.arange(E, device=x.device).repeat_interleave(C)
        slot = torch.arange(C, device=x.device).repeat(E)

        # Diagnostics: how many experts did each token get? (0 .. E)
        _, topi = probs.topk(k, dim=-1)
        hits = torch.zeros(N, device=x.device).index_add_(0, tok, torch.ones_like(gate))
        aux = {
            "balance": balance_loss(probs, topi, E),  # logged only; EC needs no balance loss
            "z": torch.logsumexp(logits, -1).pow(2).mean(),
        }
        stats = router_stats(probs, topi, torch.full((E,), C, device=x.device),
                             torch.zeros((), device=x.device), (hits == 0).sum())
        stats["coverage"] = F.one_hot(hits.long(), E + 1).sum(0).float()
        return Dispatch(tok, exp, slot, gate, C, aux, stats)


class MoELayer(nn.Module):
    """Drop-in replacement for FFN: x [B, T, d] -> ([B, T, d], aux losses).

    The E experts are ordinary 2-layer GELU MLPs; their weights are stored
    stacked ([E, d, h] and [E, h, d]) so all experts run in one batched matmul.
    """

    def __init__(self, d_model, hidden, n_experts=8, k=2, router="topk",
                 capacity_factor=None, normalize_gates=True, ec_group="batch", dropout=0.0):
        super().__init__()
        E = n_experts
        self.w1 = nn.Parameter(torch.randn(E, d_model, hidden) * 0.02)
        self.b1 = nn.Parameter(torch.zeros(E, 1, hidden))
        self.w2 = nn.Parameter(torch.randn(E, hidden, d_model) * 0.02)
        self.b2 = nn.Parameter(torch.zeros(E, 1, d_model))
        self.drop = nn.Dropout(dropout)
        self.n_experts = E
        if router == "topk":
            self.router = TokenChoiceRouter(d_model, E, k, capacity_factor, normalize_gates)
        elif router == "expert_choice":
            self.router = ExpertChoiceRouter(d_model, E, k, capacity_factor, ec_group)
        else:
            raise ValueError(f"unknown router {router!r}")
        self.last_stats: dict = {}

    def params_per_expert(self) -> int:
        return sum(p[0].numel() for p in (self.w1, self.b1, self.w2, self.b2))

    def forward(self, x):
        B, T, d = x.shape
        xf = x.reshape(B * T, d)

        # 1. ROUTE
        plan = self.router(xf, group_size=T)
        self.last_stats = plan.stats

        # 2. DISPATCH into [E, C+1, d]; row C is the trash slot for dropped assignments.
        buf = xf.new_zeros(self.n_experts, plan.capacity + 1, d)
        buf[plan.exp, plan.slot] = xf[plan.tok]
        h = F.gelu(torch.baddbmm(self.b1, buf, self.w1))  # [E, C+1, hidden]
        y = self.drop(torch.baddbmm(self.b2, h, self.w2))  # [E, C+1, d]

        # 3. COMBINE: gather each assignment's output, weight it, add onto its token.
        #    A token picked by nobody gets zeros here — the residual stream carries it.
        contrib = y[plan.exp, plan.slot] * plan.gate.unsqueeze(-1).to(x.dtype)
        out = torch.zeros_like(xf).index_add_(0, plan.tok, contrib)
        return out.view(B, T, d), plan.aux
