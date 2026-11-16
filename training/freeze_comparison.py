"""Create an isolated code/model/profile snapshot for unattended comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import yaml

from training.overnight import ROOT, write_json


def stage_policy(checkpoint, destination, model, asset):
    """Relocate a policy's files; preserve its trained model/asset checksums."""
    source = checkpoint.parent
    destination.mkdir(parents=True)
    shutil.copy2(checkpoint, destination / checkpoint.name)
    shutil.copytree(source / "params", destination / "params")
    contract = json.loads((source / "controller_contract.json").read_text())
    if hashlib.sha256(asset.read_bytes()).hexdigest() != contract["asset_sha256"]:
        raise ValueError("Snapshot asset does not match the policy's training contract")
    for name, digest in contract["model_files"].items():
        if hashlib.sha256((model / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Snapshot model mismatch: {name}")
    contract.update(model_path=str(model), asset_path=str(asset))
    write_json(destination / "controller_contract.json", contract)
    env = yaml.safe_load((destination / "params/env.yaml").read_text())
    env.update(model_path=str(model), asset_path=str(asset))
    (destination / "params/env.yaml").write_text(yaml.safe_dump(env, sort_keys=False))
    result = (
        json.loads((source / "training_result.json").read_text())
        if (source / "training_result.json").exists()
        else {}
    )
    result["checkpoint"] = str(destination / checkpoint.name)
    write_json(destination / "training_result.json", result)
    return destination / checkpoint.name


def freeze(base, out):
    if out.exists():
        raise FileExistsError("Use a fresh snapshot directory")
    out.mkdir(parents=True)
    code = out / "Isaac-hydraulic-actuator"
    subprocess.run(
        ["git", "clone", "--quiet", "--local", "--no-hardlinks", "--branch", "dev", str(ROOT), str(code)],
        check=True,
    )
    names = subprocess.check_output(
        ["rg", "--files", "-g", "*.py", "-g", "*.toml", "-g", "*.md", "-g", "*.yaml", "-g", "*.ps1"],
        cwd=ROOT,
        text=True,
    ).splitlines()
    hashes = {}
    for name in names:
        source = ROOT / name
        target = code / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()

    original_plan = json.loads((base / "plan.json").read_text())
    robot_source = Path(original_plan["robot_repo"])
    robot = out / "kaivuriprokkis"
    for directory in ("learned_control", "modules"):
        for source in (robot_source / directory).rglob("*.py"):
            if any(part in {"__pycache__", ".venv"} for part in source.parts):
                continue
            target = robot / source.relative_to(robot_source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    runtime_provenance = dict(
        source_repository=str(robot_source),
        active_branch=subprocess.check_output(
            ["git", "-C", str(robot_source), "rev-parse", "HEAD"], text=True
        ).strip(),
    )
    if not (robot / "learned_control").is_dir():
        revision = "origin/python"
        files = subprocess.check_output(
            [
                "git",
                "-C",
                str(robot_source),
                "ls-tree",
                "-r",
                "--name-only",
                revision,
                "learned_control",
                "modules",
            ],
            text=True,
        ).splitlines()
        for name in files:
            if not name.endswith(".py"):
                continue
            target = robot / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(
                subprocess.check_output(["git", "-C", str(robot_source), "show", f"{revision}:{name}"])
            )
        runtime_provenance.update(
            runtime="Legacy Python parity reference, not the current native runtime",
            revision=subprocess.check_output(
                ["git", "-C", str(robot_source), "rev-parse", revision], text=True
            ).strip(),
        )
    write_json(robot / "runtime_provenance.json", runtime_provenance)
    profile = robot / "configuration_files/profiles/jetson"
    profile.mkdir(parents=True)
    for name in ("control_config.yaml", "servo_config.yaml", "profile.yaml"):
        shutil.copy2(robot_source / "configuration_files/profiles/jetson" / name, profile / name)

    frozen = out / "base"
    frozen.mkdir()
    model = frozen / "actuator"
    shutil.copytree(Path((base / "best_actuator.txt").read_text().strip()), model)
    (frozen / "best_actuator.txt").write_text(str(model) + "\n")
    plan = dict(original_plan, robot_repo=str(robot), repository=str(code), original_base=str(base))
    write_json(frozen / "plan.json", plan)

    reference_item = json.loads((base / "controller_checkpoints.json").read_text())[0]
    _, _, reference_path = reference_item.partition("=")
    source_ref = Path(reference_path)
    source_contract = json.loads((source_ref.parent / "controller_contract.json").read_text())
    asset = Path(source_contract["asset_path"])
    if not asset.is_absolute():
        asset = ROOT / asset
    ref_asset = frozen / "historical_asset.usd"
    shutil.copy2(asset, ref_asset)
    reference = stage_policy(source_ref, frozen / "reference_controller", code / "models/actuator", ref_asset)

    previous = base / "controllers/lookahead_accel_lr3e-4"
    checkpoint = Path(json.loads((previous / "training_result.json").read_text())["checkpoint"])
    previous_dir = frozen / "controllers/lookahead_accel_lr3e-4"
    previous_checkpoint = stage_policy(
        checkpoint, previous_dir, model, code / "assets/excavator_bucket_rocking.usd"
    )
    env_path = previous_dir / "params/env.yaml"
    env = yaml.safe_load(env_path.read_text())
    env["extra_model_paths"] = [str(code / "models/actuator")]
    env_path.write_text(yaml.safe_dump(env, sort_keys=False))
    write_json(
        frozen / "controller_checkpoints.json",
        [f"deployed={reference}", f"previous_tip_long={previous_checkpoint}"],
    )
    write_json(
        out / "snapshot.json",
        dict(original_repository=str(ROOT), original_run=str(base), source_sha256=hashes),
    )
    print(json.dumps(dict(code=str(code), base=str(frozen), robot=str(robot)), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    freeze(args.base.resolve(), args.out.resolve())


if __name__ == "__main__":
    main()
