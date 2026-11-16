"""Package a controller and its exact actuator with portable paths, without modifying the source run."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml

from hydraulic_controller.core import model_fingerprint, portable_path, resolve_path, sha256
from hydraulic_controller.policy import ControllerPolicy


def package(checkpoint: Path, out: Path, model_out: Path, *, source_run: str, selection: str) -> Path:
    """Copy a verified checkpoint and plant to fresh destinations; retain original contract and provenance."""
    policy = ControllerPolicy(checkpoint)
    if out.exists() or model_out.exists():
        raise FileExistsError("Use fresh policy and actuator destinations")
    environment = yaml.safe_load((policy.checkpoint.parent / "params/env.yaml").read_text())
    # A relative path in a frozen run is resolved by that run's source tree. Its saved environment records
    # absolute dependency paths, so use those to verify and relocate any secondary training plants.
    extra = []
    for original, target in zip(
        environment.get("extra_model_paths", []), policy.contract.get("extra_model_paths", []), strict=True
    ):
        destination = resolve_path(target)
        if model_fingerprint(resolve_path(original)) != model_fingerprint(destination):
            raise ValueError(f"Secondary training plant differs from the packaged dependency: {destination}")
        extra.append(portable_path(destination))

    model_out.mkdir(parents=True)
    for name in policy.contract["model_files"]:
        shutil.copy2(policy.model_path / name, model_out / name)
    out.mkdir(parents=True)
    shutil.copy2(policy.checkpoint, out / policy.checkpoint.name)
    shutil.copytree(policy.checkpoint.parent / "params", out / "params")
    shutil.copy2(
        policy.checkpoint.parent / "controller_contract.json", out / "controller_contract.original.json"
    )
    contract = dict(
        policy.contract, model_path=portable_path(model_out), asset_path=portable_path(policy.asset_path)
    )
    if extra:
        contract["extra_model_paths"] = extra
    (out / "controller_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    environment.update(
        model_path=contract["model_path"], asset_path=contract["asset_path"], extra_model_paths=extra
    )
    (out / "params/env.yaml").write_text(yaml.safe_dump(environment, sort_keys=False))
    provenance = dict(
        source_run=source_run,
        source_checkpoint=str(policy.checkpoint),
        checkpoint_sha256=sha256(policy.checkpoint),
        selection=selection,
        tracked_point=policy.tracked_point,
        model_files=policy.contract["model_files"],
        asset_sha256=policy.contract["asset_sha256"],
        hardware_validated=False,
    )
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    result = out / policy.checkpoint.name
    ControllerPolicy(result)  # verify the relocated package before reporting success
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model_out", type=Path, required=True)
    parser.add_argument("--source_run", required=True)
    parser.add_argument("--selection", required=True)
    args = parser.parse_args()
    print(package(**vars(args)))


if __name__ == "__main__":
    main()
