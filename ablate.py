"""
ablate.py — run every config in the ablation grid, then make plots.

Run:
    ../04-diffusion/.venv/bin/python ablate.py              # full grid (~45 min on M-series MPS)
    ../04-diffusion/.venv/bin/python ablate.py --steps 300  # quick smoke run
    ../04-diffusion/.venv/bin/python ablate.py --only topk_aux expert_choice

Runs whose runs/<name>_s<seed>/summary.json already exists are skipped, so an
interrupted grid resumes where it stopped (delete the folder to re-run one).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

# (run name, config file, extra --set overrides, seeds)
GRID = [
    ("dense",             "dense.json",             [],                   [0, 1]),
    ("topk_softmax",      "topk_softmax.json",      [],                   [0, 1]),
    ("topk_softmax_cap",  "topk_softmax_cap.json",  [],                   [0, 1]),
    ("topk_aux",          "topk_aux.json",          [],                   [0, 1]),
    ("topk_aux_nocap",    "topk_aux_nocap.json",    [],                   [0, 1]),
    ("expert_choice",     "expert_choice.json",     [],                   [0, 1]),
    ("expert_choice_seq", "expert_choice_seq.json", [],                   [0, 1]),
    # sensitivity of the aux-loss coefficient (capacity 1.25, one seed)
    ("topk_aux_0.001",    "topk_aux.json",          ["aux_coef=0.001"],   [0]),
    ("topk_aux_0.1",      "topk_aux.json",          ["aux_coef=0.1"],     [0]),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--out-root", default="runs")
    p.add_argument("--only", nargs="*", default=None)
    args = p.parse_args()

    for name, cfg_file, sets, seeds in GRID:
        if args.only and name not in args.only:
            continue
        for seed in seeds:
            run_dir = os.path.join(args.out_root, f"{name}_s{seed}")
            if os.path.exists(os.path.join(run_dir, "summary.json")):
                print(f"skip {run_dir} (done)")
                continue
            cmd = [sys.executable, "train.py", "--config", os.path.join("configs", cfg_file),
                   "--set", f"name={name}", "--set", f"seed={seed}", "--set", f"out_root={args.out_root}"]
            for s in sets:
                cmd += ["--set", s]
            if args.steps:
                cmd += ["--set", f"steps={args.steps}", "--set", f"eval_every={max(1, args.steps // 6)}"]
            print("\n$ " + " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)

    subprocess.run([sys.executable, "plots.py", "--runs", args.out_root], check=True)


if __name__ == "__main__":
    main()
