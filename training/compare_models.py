# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Score several actuator models on one frozen split with the training.actuator rollout metrics.

    python -m training.compare_models runs/actuator --split test --cases 200 deployed=models/actuator new=runs/x/model

Writes ``compare_<split>_<cases>.json`` into the run directory and prints one row per model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.actuator import evaluate, load_chunks


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run", type=Path, help="prepared run directory (training.prepare --out)")
    parser.add_argument("models", nargs="+", metavar="NAME=MODEL_DIR")
    parser.add_argument("--split", choices=("train", "dev", "test"), default="test")
    parser.add_argument("--cases", type=int, default=200)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--best_out", type=Path, help="write the model directory with the lowest 10 s path error here"
    )
    args = parser.parse_args()
    chunks = load_chunks(args.run, args.split)
    rows = {}
    for item in args.models:
        name, _, path = item.partition("=")
        r = rows[name] = evaluate(Path(path), chunks, args.device, args.cases)
        print(
            f"{name:20s} 1s {r['1']['path_deg']:.2f}  10s {r['10']['path_deg']:.2f}  30s {r['30']['path_deg']:.2f}  "
            f"end30 {r['30']['endpoint_deg']:.2f}  rest10 {r['rest10_deg']:.2f}  stop {r['stop_rate_rad_s']:.4f}",
            flush=True,
        )
    (args.run / f"compare_{args.split}_{args.cases}.json").write_text(json.dumps(rows, indent=2) + "\n")
    if args.best_out:
        paths = dict(item.partition("=")[::2] for item in args.models)
        best = min((name for name in rows if rows[name]["finite"]), key=lambda name: rows[name]["path10_deg"])
        args.best_out.write_text(str(Path(paths[best]).resolve()) + "\n")
        print(f"best: {best}")


if __name__ == "__main__":
    main()
