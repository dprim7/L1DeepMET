#!/usr/bin/env python3
"""
Drive the 30-cell event-count sweep: 5 N × 2 variants × 3 seeds.

Each cell launches `train_one.py` as a subprocess so they're isolated
(separate TF processes; failures in one don't poison the others). Up to
`--max-parallel` cells run concurrently via concurrent.futures.

Usage:
  python run_sweep.py \\
      --h5-dir outputs/preprocessed/26May21_142_extended_v0 \\
      --results-dir reports/event_count_justification/results/raw \\
      --max-parallel 8 \\
      --epochs 50
"""
from __future__ import annotations
import argparse
import concurrent.futures
import datetime as _dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

N_TRAIN_GRID = [400, 800, 2000, 4000, 8000]
VARIANTS = ["baseline", "extended"]
SEEDS = [42, 123, 456]

THIS_DIR = Path(__file__).resolve().parent


def cell_id(n: int, variant: str, seed: int) -> str:
    return f"n{n}_{variant}_seed{seed}"


def run_one_cell(args_dict: dict) -> tuple[str, bool, float, str]:
    """Spawn train_one.py for one cell. Returns (cell_id, ok, wall_s, message)."""
    n = args_dict["n"]; variant = args_dict["variant"]; seed = args_dict["seed"]
    cid = cell_id(n, variant, seed)
    out_path = Path(args_dict["results_dir"]) / f"{cid}.json"
    if out_path.exists() and not args_dict["force"]:
        return cid, True, 0.0, "SKIP (already done)"

    cmd = [sys.executable, str(THIS_DIR / "train_one.py"),
           "--h5-dir", str(args_dict["h5_dir"]),
           "--n-train", str(n),
           "--variant", variant,
           "--seed", str(seed),
           "--epochs", str(args_dict["epochs"]),
           "--batch-size", str(args_dict["batch_size"]),
           "--out", str(out_path)]
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(THIS_DIR.parent.parent.parent / "src"))

    t0 = time.time()
    try:
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False, timeout=3600)
        wall = time.time() - t0
        if r.returncode != 0:
            return cid, False, wall, f"exit={r.returncode}; stderr tail: {r.stderr[-500:].strip()}"
        if not out_path.exists():
            return cid, False, wall, "ran but no output JSON"
        return cid, True, wall, "ok"
    except subprocess.TimeoutExpired:
        return cid, False, time.time() - t0, "timeout (1 hr)"
    except Exception as e:
        return cid, False, time.time() - t0, f"{type(e).__name__}: {e}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5-dir", required=True, type=Path)
    p.add_argument("--results-dir", required=True, type=Path)
    p.add_argument("--max-parallel", type=int, default=8)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--force", action="store_true",
                   help="Re-run cells even if their JSON already exists")
    p.add_argument("--cells", nargs="*", default=None,
                   help="Specific cell IDs to run (default: all 30). e.g. n400_extended_seed42")
    args = p.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    all_cells = [{"n": n, "variant": v, "seed": s,
                  "h5_dir": args.h5_dir, "results_dir": args.results_dir,
                  "epochs": args.epochs, "batch_size": args.batch_size,
                  "force": args.force}
                 for n in N_TRAIN_GRID for v in VARIANTS for s in SEEDS]
    if args.cells:
        wanted = set(args.cells)
        all_cells = [c for c in all_cells if cell_id(c["n"], c["variant"], c["seed"]) in wanted]

    print(f"━━━ sweep: {len(all_cells)} cells, max_parallel={args.max_parallel}, epochs={args.epochs}")
    print(f"   results → {args.results_dir}")
    t0 = time.time()
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_parallel) as ex:
        future_to_cid = {ex.submit(run_one_cell, c): cell_id(c["n"], c["variant"], c["seed"])
                         for c in all_cells}
        done = 0
        total = len(future_to_cid)
        for fut in concurrent.futures.as_completed(future_to_cid):
            cid, ok, wall, msg = fut.result()
            done += 1
            mark = "✓" if ok else "✗"
            ts = _dt.datetime.utcnow().strftime("%H:%M:%SZ")
            print(f"  [{done:>2}/{total}] {mark} {cid:35s}  ({wall:>5.0f}s)  {msg}", flush=True)
            results.append({"cell": cid, "ok": ok, "wall_s": wall, "msg": msg})

    wall_total = time.time() - t0
    n_ok = sum(1 for r in results if r["ok"])
    print(f"━━━ sweep done in {wall_total:.0f}s — {n_ok}/{len(results)} ok")
    summary_path = args.results_dir.parent / "sweep_summary.json"
    summary_path.write_text(json.dumps({
        "wall_s": wall_total,
        "n_ok": n_ok, "n_total": len(results),
        "cells": results,
        "finished_at": _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }, indent=2))
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
