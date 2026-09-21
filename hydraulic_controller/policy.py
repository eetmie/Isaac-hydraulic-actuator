"""Load standard RSL-RL checkpoints for simulator-independent inference."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import torch

from .core import CONTRACT_VERSION, ControllerSettings, model_fingerprint, resolve_path, sha256


class ControllerPolicy:
    """Deterministic actor with checkpoint normalization and verified plant metadata."""

    def __init__(self, checkpoint: str | Path, device: str = "cpu"):
        import yaml
        from rsl_rl.models import MLPModel
        from tensordict import TensorDict

        self.checkpoint = Path(checkpoint).resolve()
        directory = self.checkpoint.parent
        self.contract = json.loads((directory / "controller_contract.json").read_text())
        if self.contract["version"] != CONTRACT_VERSION or self.contract["action_transform"] != "tanh":
            raise ValueError("Unsupported hydraulic controller checkpoint contract")
        self.settings = ControllerSettings(**self.contract["settings"])
        self.model_path = resolve_path(self.contract["model_path"])
        self.asset_path = resolve_path(self.contract["asset_path"])
        if model_fingerprint(self.model_path) != self.contract["model_files"]:
            raise ValueError("Hydraulic model changed since training; use the checkpoint's original model")
        if sha256(self.asset_path) != self.contract["asset_sha256"]:
            raise ValueError("Robot asset changed since training; kinematics must match the checkpoint")
        agent = yaml.safe_load((directory / "params/agent.yaml").read_text())
        actor = copy.deepcopy(agent["actor"])
        kwargs = {
            key: actor[key] for key in ("hidden_dims", "activation", "obs_normalization", "distribution_cfg")
        }
        self.device = device
        width = self.contract["observation_dim"]
        dummy = TensorDict({"policy": torch.zeros(1, width, device=device)}, batch_size=[1])
        self.actor = MLPModel(dummy, {"actor": ["policy"]}, "actor", 3, **kwargs).to(device)
        state = torch.load(self.checkpoint, map_location=device, weights_only=True)
        self.actor.load_state_dict(state["actor_state_dict"], strict=True)
        self.actor.eval().requires_grad_(False)
        self.inference = self.actor.as_jit().to(device).eval()

    def governor(self, kinematics, speed_margin: float | None = None):
        """Command governor, configured as during training unless ``speed_margin`` overrides it.

        Args:
            kinematics: Kinematics used for the workspace projection.
            speed_margin: ``None`` keeps the training configuration, ``0`` disables joint-speed scaling, and a
                positive value scales commands to that fraction of the full-valve joint speeds. Checkpoints trained
                without the governor have no stored limits; they are then measured on the benchmark poses.
        """
        from .kinematics import CommandGovernor

        limits = self.contract.get("joint_speed_limits_rad_s")
        margin = self.contract.get("governor_joint_speed_margin", 0.8)
        if speed_margin is not None:
            margin = speed_margin
            if speed_margin <= 0:
                limits = None
            elif limits is None:
                from .benchmark import benchmark_poses
                from .core import measure_joint_speed_limits

                limits = measure_joint_speed_limits(
                    self.model_path, kinematics, benchmark_poses(kinematics, 8), self.settings
                ).tolist()
        return CommandGovernor(
            kinematics,
            self.settings,
            None if limits is None else torch.tensor(limits, device=self.device),
            margin,
        )

    @torch.no_grad()
    def __call__(self, observations: torch.Tensor) -> torch.Tensor:
        """Return deterministic raw actions; the shared plant applies tanh once."""
        return self.inference(observations)

    def export(self, directory: str | Path | None = None) -> Path:
        """Export actor and normalization; output actions require the documented tanh."""
        directory = Path(directory) if directory else self.checkpoint.parent / "exported"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (self.checkpoint.stem + "_policy.pt")
        torch.jit.script(copy.deepcopy(self.inference).cpu()).save(str(path))
        (path.with_suffix(".json")).write_text(json.dumps(self.contract, indent=2) + "\n")
        return path
