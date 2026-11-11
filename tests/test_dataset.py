# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""The canonical adapter must preserve collection boundaries and reject corrupt data."""

import hashlib
import json

import numpy as np
import pytest

from training.prepare import prepare


def corpus(tmp_path):
    root = tmp_path / "corpus"
    root.mkdir()
    recordings, hashes = [], {}
    for i, group in enumerate(("first", "first", "second", "third")):
        name = f"chunk_{i}.npz"
        q = np.zeros((300, 4), dtype=np.float32)
        np.savez(root / name, q=q, v=q, target_q=q, target_v=q, u=q[:, :3], t=np.arange(300) * 0.01)
        hashes[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
        recordings.append(
            dict(
                id=f"recording_{i}",
                generation="v1" if i < 2 else "v2",
                collection_group=group,
                training_paths=[name],
            )
        )
    manifest = dict(
        format="kaivuri-complete-actuator-v1",
        training_hz=100,
        sensor_contract={},
        recordings=recordings,
        output_sha256=hashes,
    )
    (root / "dataset.json").write_text(json.dumps(manifest))
    return root


def test_collection_groups_and_canonical_arrays_stay_intact(tmp_path):
    root = corpus(tmp_path)
    out = tmp_path / "run"
    report = prepare(root, out, ["second"], ["third"])
    assert report["groups"] == {"train": ["first"], "dev": ["second"], "test": ["third"]}
    for row in report["chunks"]:
        assert row["split"] == {"first": "train", "second": "dev", "third": "test"}[row["collection_group"]]
        assert (out / "data" / row["split"] / row["source"]).read_bytes() == (
            root / row["source"]
        ).read_bytes()
    with pytest.raises(FileExistsError):
        prepare(root, out, ["second"], ["third"])


@pytest.mark.parametrize(
    "dev,test", [(["unknown"], []), (["second"], ["second"]), (["first", "second", "third"], []), ([], [])]
)
def test_invalid_group_assignments_are_rejected(tmp_path, dev, test):
    with pytest.raises(ValueError):
        prepare(corpus(tmp_path), tmp_path / "run", dev, test)


def test_checksum_failure_creates_no_prepared_data(tmp_path):
    root = corpus(tmp_path)
    with (root / "chunk_3.npz").open("ab") as stream:
        stream.write(b"changed")
    out = tmp_path / "run"
    with pytest.raises(ValueError, match="checksum"):
        prepare(root, out, ["second"], [])
    assert not (out / "data").exists()


def test_trainer_rejects_a_prepared_chunk_changed_after_split_freezing(tmp_path):
    from training.actuator import load_chunks

    root = corpus(tmp_path)
    out = tmp_path / "run"
    prepare(root, out, ["second"], ["third"])
    assert len(load_chunks(out, "train")) == 2
    with (out / "data/train/chunk_1.npz").open("ab") as stream:
        stream.write(b"changed after preparation")
    with pytest.raises(ValueError, match="checksum"):
        load_chunks(out, "train")
