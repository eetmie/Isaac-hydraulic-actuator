# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Prepare manifest-grouped canonical NPZ data for training.actuator.

Run from the actuator repository with the parent Isaac Lab environment:
uv run --project ../.. python -m training.prepare --out runs/complete_trial
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def prepare(root: Path, out: Path, dev_groups: list[str], test_groups: list[str]) -> dict:
    """Keep collection groups whole and verify each chunk before copying it."""
    manifest_path = root / "dataset.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["format"] != "kaivuri-complete-actuator-v1" or manifest["training_hz"] != 100:
        raise ValueError("Expected the canonical complete actuator dataset at 100 Hz")
    available = {r["collection_group"] for r in manifest["recordings"]}
    dev, test = set(dev_groups), set(test_groups)
    if not dev or dev & test or not (dev | test) <= available or not available - dev - test:
        raise ValueError("Require disjoint known train/dev/test groups and nonempty train/dev")
    if (out / "data").exists() or (out / "split_manifest.json").exists():
        raise FileExistsError("Use a fresh output directory to preserve the frozen split")
    rows = []
    for rec in manifest["recordings"]:
        group = rec["collection_group"]
        split = "dev" if group in dev else "test" if group in test else "train"
        for name in rec["training_paths"]:
            source = (root / name).resolve()
            if not source.is_relative_to(root.resolve()):
                raise ValueError(f"Chunk escapes dataset root: {name}")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            if digest != manifest["output_sha256"][name]:
                raise ValueError(f"Chunk checksum mismatch: {name}")
            with np.load(source, allow_pickle=False) as c:
                n = len(c["t"])
                if n < 300 or not np.allclose(np.diff(c["t"]), 0.01, atol=1e-9, rtol=0):
                    raise ValueError(f"Non-contiguous 100 Hz chunk: {name}")
                for key, width in (("q", 4), ("v", 4), ("target_q", 4), ("target_v", 4), ("u", 3)):
                    if c[key].shape != (n, width) or not np.isfinite(c[key]).all():
                        raise ValueError(f"Invalid {key}: {name}")
                if not np.array_equal(c["q"], c["target_q"]) or not np.array_equal(c["v"], c["target_v"]):
                    raise ValueError(f"Expected unsmoothed canonical targets: {name}")
            rows.append(
                dict(
                    source=name,
                    split=split,
                    recording=rec["id"],
                    collection_group=group,
                    generation=rec["generation"],
                    rows=n,
                    sha256=digest,
                )
            )
    # Validate the entire input before creating the prepared corpus.
    for row in rows:
        destination = out / "data" / row["split"] / Path(row["source"]).name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / row["source"], destination)
    report = dict(
        dataset_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        dataset=str(root.resolve()),
        sensor_contract=manifest["sensor_contract"],
        split_unit="collection_group",
        target="canonical same-packet gyro; no smoothing",
        groups={"train": sorted(available - dev - test), "dev": sorted(dev), "test": sorted(test)},
        independent_test_available=bool(test),
        note="Development selects checkpoints. Existing baseline may have seen these historical recordings.",
        chunks=rows,
    )
    (out / "split_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "dataset/complete_dataset")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dev-groups", nargs="+", default=["v2_20260909"])
    parser.add_argument("--test-groups", nargs="*", default=[])
    args = parser.parse_args()
    report = prepare(args.dataset, args.out, args.dev_groups, args.test_groups)
    print(
        json.dumps(
            {
                "groups": report["groups"],
                "rows": {
                    split: sum(c["rows"] for c in report["chunks"] if c["split"] == split)
                    for split in ("train", "dev", "test")
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
