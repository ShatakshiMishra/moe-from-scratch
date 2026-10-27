"""
train.py — train one GPT (dense or MoE) from a JSON config and log routing metrics.

Run:
    PY=../04-diffusion/.venv/bin/python
    $PY train.py --config configs/topk_aux.json
    $PY train.py --config configs/expert_choice.json --set seed=1 --set steps=500

Config files are flat JSON. A config may name a "base" file; its keys are
loaded first and then overridden. `--set key=value` overrides on top of that
(value parsed as JSON, falling back to a string).

Outputs (runs/<name>/):
    config.json     the fully-resolved config
    metrics.jsonl   one line per eval: losses, throughput, per-layer expert load
    summary.json    final eval: val ppl, causal-consistent ppl, causality probe,
                    inference throughput, parameter counts, a text sample
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F

from data import CharData
from model import GPT


# ----------------------------------------------------------------------------- config

def load_config(path: str, overrides: list[str], _top: bool = True) -> dict:
    with open(path) as f:
        cfg = json.load(f)
    if "base" in cfg:
        base = load_config(os.path.join(os.path.dirname(path), cfg.pop("base")), [], _top=False)
        cfg = {**base, **cfg}
    for item in overrides:
        key, val = item.split("=", 1)
        try:
            cfg[key] = json.loads(val)
        except json.JSONDecodeError:
            cfg[key] = val
    if _top:
        cfg.setdefault("name", os.path.splitext(os.path.basename(path))[0])
    return cfg


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def lr_at(step, cfg):
    """Linear warmup, then cosine decay to 10% of peak."""
    if step < cfg["warmup"]:
        return cfg["lr"] * (step + 1) / cfg["warmup"]
    t = (step - cfg["warmup"]) / max(1, cfg["steps"] - cfg["warmup"])
    return cfg["lr"] * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))


# ----------------------------------------------------------------------------- routing stats

class StatsWindow:
    """Sums each MoE layer's per-forward stats over a logging window (on-device, no syncs)."""

    def __init__(self, model):
        self.model = model
        self.reset()

    def reset(self):
        self.acc = [dict() for _ in self.model.moe_layers()]

    def add(self):
        for acc, layer in zip(self.acc, self.model.moe_layers()):
            for k, v in layer.last_stats.items():
                acc[k] = acc[k] + v if k in acc else v.clone()

    def summarize(self):
        out = []
        for acc in self.acc:
            if not acc:
                continue
            load = acc["load"]
            frac = load / load.sum()
            pref = acc["pref"] / acc["pref"].sum()
            n_tok = acc["n_tokens"]
            row = {
                "load_frac": frac.tolist(),
                # Coefficient of variation of expert load; CV^2 = Var(E * frac).
                # 0 = perfectly balanced, sqrt(E-1) = everything on one expert.
                "load_cv": (frac.std(unbiased=False) / frac.mean()).item(),
                "load_var": (frac * len(frac)).var(unbiased=False).item(),
                "max_load_ratio": (frac.max() * len(frac)).item(),  # 1.0 = balanced
                # Same metrics for where the router WANTS to send tokens (plain top-k
                # on its probs) — the only meaningful imbalance signal for expert-choice.
                "pref_cv": (pref.std(unbiased=False) / pref.mean()).item(),
                "dropped_frac": (acc["dropped"] / (n_tok * self.model.cfg["top_k"])).item(),
                "unrouted_frac": (acc["unrouted"] / n_tok).item(),
                "entropy": (acc["entropy"] / self._n_forwards(acc)).item(),
            }
            if "coverage" in acc:
                cov = acc["coverage"] / acc["coverage"].sum()
                row["coverage"] = cov.tolist()  # P(token processed by j experts), j = 0..E
            out.append(row)
        return out

    def _n_forwards(self, acc):
        # entropy was summed once per forward; recover the count from tokens/forward
        per_fwd = self.model.cfg["batch_size"] * self.model.cfg["block_size"]
        return max(1.0, (acc["n_tokens"] / per_fwd).item())


def mean_over_layers(layers, key):
    vals = [l[key] for l in layers]
    return sum(vals) / len(vals) if vals else 0.0


# ----------------------------------------------------------------------------- evals

@torch.no_grad()
def eval_loss(model, batches):
    model.eval()
    losses = [model(x, y)[1].item() for x, y in batches]
    model.train()
    return sum(losses) / len(losses)


@torch.no_grad()
def causal_eval(model, x, y):
    """Loss when every position is scored the way it would be at generation time.

    For position t we run the model on the prefix x[:t+1] ALONE and read the
    last logit. A strictly causal model gives exactly the teacher-forced loss.
    A model whose routing peeks at future tokens / other sequences does not.
    """
    model.eval()
    tf = model(x, y)[1].item()
    total, n = 0.0, 0
    for s in range(x.shape[0]):
        for t in range(x.shape[1]):
            logits = model(x[s : s + 1, : t + 1])[0]
            total += F.cross_entropy(logits[0, -1], y[s, t]).item()
            n += 1
    model.train()
    return tf, total / n


@torch.no_grad()
def causality_probe(model, x, vocab_size, seed=0):
    """Change the second half of each sequence; do logits in the first half move?"""
    model.eval()
    T = x.shape[1]
    g = torch.Generator().manual_seed(seed)
    x2 = x.clone()
    x2[:, T // 2 :] = torch.randint(vocab_size, x2[:, T // 2 :].shape, generator=g).to(x.device)
    a = model(x)[0][:, : T // 2]
    b = model(x2)[0][:, : T // 2]
    diff = (a - b).abs().amax(-1)  # [B, T/2]
    model.train()
    return {
        "max_abs_logit_change": diff.max().item(),
        "frac_positions_changed": (diff > 1e-4).float().mean().item(),
    }


@torch.no_grad()
def inference_throughput(model, x, iters=20):
    model.eval()
    dev = x.device
    for _ in range(3):
        model(x)
    sync(dev)
    t0 = time.perf_counter()
    for _ in range(iters):
        model(x)
    sync(dev)
    model.train()
    return iters * x.numel() / (time.perf_counter() - t0)


# ----------------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description="Train a dense or MoE char-level GPT")
    p.add_argument("--config", required=True)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = p.parse_args()
    cfg = load_config(args.config, args.set)

    torch.manual_seed(cfg["seed"])
    device = pick_device(cfg["device"])
    run_name = f'{cfg["name"]}_s{cfg["seed"]}'
    outdir = os.path.join(cfg["out_root"], run_name)
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # 1. Data + model
    data = CharData(cfg["data_path"])
    model = GPT(cfg, data.vocab_size).to(device)
    total, active = model.param_counts()
    print(f"[{run_name}] router={cfg['router']} device={device} "
          f"params total={total/1e6:.2f}M active/token={active/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.95),
                            weight_decay=cfg["weight_decay"])
    val_batches = data.fixed_val_batches(cfg["eval_batches"], cfg["batch_size"], cfg["block_size"], device)
    window = StatsWindow(model)
    B, T = cfg["batch_size"], cfg["block_size"]

    # 2. Train
    log = open(os.path.join(outdir, "metrics.jsonl"), "w")
    train_loss_acc, aux_acc, n_acc = 0.0, {"balance": 0.0, "z": 0.0}, 0
    train_tps = []
    sync(device)
    t_window = time.perf_counter()
    for step in range(1, cfg["steps"] + 1):
        for gparam in opt.param_groups:
            gparam["lr"] = lr_at(step, cfg)
        x, y = data.batch("train", B, T, device)
        _, ce, aux = model(x, y)
        loss = ce + cfg["aux_coef"] * aux["balance"] + cfg["z_coef"] * aux["z"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step()

        window.add()
        train_loss_acc += ce.detach()
        aux_acc = {k: aux_acc[k] + aux[k].detach() for k in aux_acc}
        n_acc += 1

        if step % cfg["eval_every"] == 0 or step == cfg["steps"]:
            sync(device)
            tps = n_acc * B * T / (time.perf_counter() - t_window)
            if step > cfg["eval_every"]:  # first window includes warm-up/compile noise
                train_tps.append(tps)
            layers = window.summarize()
            n_moe = max(1, len(model.moe_layers()))
            val = eval_loss(model, val_batches)
            rec = {
                "step": step,
                "train_loss": (train_loss_acc / n_acc).item(),
                "val_loss": val,
                "val_ppl": math.exp(val),
                "balance_loss": (aux_acc["balance"] / n_acc / n_moe).item() if model.moe_layers() else None,
                "z_loss": (aux_acc["z"] / n_acc / n_moe).item() if model.moe_layers() else None,
                "train_tok_per_s": tps,
                "load_cv": mean_over_layers(layers, "load_cv"),
                "load_var": mean_over_layers(layers, "load_var"),
                "pref_cv": mean_over_layers(layers, "pref_cv"),
                "dropped_frac": mean_over_layers(layers, "dropped_frac"),
                "unrouted_frac": mean_over_layers(layers, "unrouted_frac"),
                "entropy": mean_over_layers(layers, "entropy"),
                "layers": layers,
            }
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(f"step {step:5d} | train {rec['train_loss']:.3f} | val {val:.3f} (ppl {rec['val_ppl']:.2f}) "
                  f"| loadCV {rec['load_cv']:.2f} prefCV {rec['pref_cv']:.2f} "
                  f"drop {rec['dropped_frac']:.3f} unrouted {rec['unrouted_frac']:.3f} | {tps/1e3:.1f}k tok/s")
            window.reset()
            train_loss_acc, aux_acc, n_acc = 0.0, {"balance": 0.0, "z": 0.0}, 0
            sync(device)
            t_window = time.perf_counter()
    log.close()

    # 3. Final evaluation
    print("final eval ...")
    final_val = eval_loss(model, data.fixed_val_batches(cfg["final_eval_batches"], B, T, device, seed=99))
    xs, ys = data.fixed_val_batches(1, cfg["causal_eval_seqs"], T, device, seed=7)[0]
    tf_loss, ar_loss = causal_eval(model, xs, ys)
    probe = causality_probe(model, val_batches[0][0][:8], data.vocab_size)
    inf_tps = inference_throughput(model, val_batches[0][0])
    torch.manual_seed(0)
    start = torch.tensor([[data.stoi["\n"]]], device=device)
    sample = data.decode(model.generate(start, 300)[0].tolist())

    summary = {
        "run": run_name,
        "name": cfg["name"],
        "router": cfg["router"],
        "aux_coef": cfg["aux_coef"],
        "capacity_factor": cfg["capacity_factor"],
        "top_k": cfg["top_k"],
        "params_total": total,
        "params_active": active,
        "final_val_loss": final_val,
        "final_val_ppl": math.exp(final_val),
        "teacher_forced_loss_causal_set": tf_loss,
        "autoregressive_loss_causal_set": ar_loss,
        "teacher_forced_ppl_causal_set": math.exp(tf_loss),
        "autoregressive_ppl_causal_set": math.exp(ar_loss),
        "train_tok_per_s": sum(train_tps) / max(1, len(train_tps)),
        "infer_tok_per_s": inf_tps,
        **probe,
        "sample": sample,
    }
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "sample"}, indent=2))
    print("sample:\n" + sample)


if __name__ == "__main__":
    main()
