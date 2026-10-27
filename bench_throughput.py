"""
bench_throughput.py — a fair tokens/sec comparison across routing strategies.

The tok/s logged by train.py is measured while each run trains on its own, so
anything else using the GPU at that moment skews it. Here every strategy is
first trained briefly (so routers have developed their real imbalance), then
timed in INTERLEAVED rounds (A, B, C, ..., A, B, C, ...): background load hits
every strategy equally, and we report the median over rounds.

Run:  ../04-diffusion/.venv/bin/python bench_throughput.py [--warm-steps 600 --rounds 15]
Writes figures/throughput_bench.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import torch

from data import CharData
from model import GPT
from train import load_config, pick_device, sync, lr_at

CONFIGS = ["dense", "topk_softmax", "topk_softmax_cap", "topk_aux", "topk_aux_nocap",
           "expert_choice", "expert_choice_seq"]


def train_step(model, opt, x, y, cfg):
    _, ce, aux = model(x, y)
    loss = ce + cfg["aux_coef"] * aux["balance"] + cfg["z_coef"] * aux["z"]
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--warm-steps", type=int, default=600)
    p.add_argument("--rounds", type=int, default=15)
    p.add_argument("--steps-per-round", type=int, default=10)
    args = p.parse_args()

    data = CharData("input.txt")
    setups = {}
    for name in CONFIGS:
        cfg = load_config(f"configs/{name}.json", [])
        cfg["steps"] = args.warm_steps
        torch.manual_seed(0)
        device = pick_device(cfg["device"])
        model = GPT(cfg, data.vocab_size).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
        for step in range(1, args.warm_steps + 1):
            for g in opt.param_groups:
                g["lr"] = lr_at(step, cfg)
            train_step(model, opt, *data.batch("train", cfg["batch_size"], cfg["block_size"], device), cfg)
        setups[name] = (cfg, model, opt, device)
        print(f"warmed {name}")

    times = {n: {"train": [], "infer": []} for n in CONFIGS}
    for r in range(args.rounds):
        for name, (cfg, model, opt, device) in setups.items():
            B, T = cfg["batch_size"], cfg["block_size"]
            batches = [data.batch("train", B, T, device) for _ in range(args.steps_per_round)]
            sync(device)
            t0 = time.perf_counter()
            for x, y in batches:
                train_step(model, opt, x, y, cfg)
            sync(device)
            times[name]["train"].append(args.steps_per_round * B * T / (time.perf_counter() - t0))
            model.eval()
            with torch.no_grad():
                t0 = time.perf_counter()
                for x, _ in batches:
                    model(x)
                sync(device)
            model.train()
            times[name]["infer"].append(args.steps_per_round * B * T / (time.perf_counter() - t0))
        print(f"round {r + 1}/{args.rounds}")

    result = {n: {"train_tok_per_s": statistics.median(t["train"]),
                  "infer_tok_per_s": statistics.median(t["infer"])} for n, t in times.items()}
    os.makedirs("figures", exist_ok=True)
    with open("figures/throughput_bench.json", "w") as f:
        json.dump(result, f, indent=2)
    for n, v in result.items():
        print(f"{n:20s} train {v['train_tok_per_s']:>10,.0f} tok/s   infer {v['infer_tok_per_s']:>11,.0f} tok/s")


if __name__ == "__main__":
    main()
