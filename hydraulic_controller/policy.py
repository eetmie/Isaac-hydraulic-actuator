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


class PolicyController:
    """The trained valve policy as a ``closed_loop`` controller, run the way ``run_robot_controller.py`` runs it.

    Every 100 Hz tick pushes the measurement into a ``MeasuredHistory``. Every ``decimation`` ticks the reference
    becomes a twist request ``v_ref + kp (p_ref - p)`` with the bucket angle held by ``kp_angle``, as in
    ``benchmark.run_trajectories``. The training-time governor admits it, and the actor's tanh output is held
    until the next policy tick. Tip tasks only.
    """

    def __init__(self, policy: ControllerPolicy, kp: float = 3.0, kp_angle: float = 3.0):
        self.policy, self.kp, self.kp_angle = policy, kp, kp_angle

    def reset(self, task) -> None:
        from .core import HydraulicModelBatch
        from .observations import MeasuredHistory

        if not bool(task.is_tip.all()):
            raise ValueError("the policy tracks tip motion; give it tip-line tasks only")
        device = task.q0.device
        source = HydraulicModelBatch(self.policy.model_path, 1, str(device)).source
        self.history = MeasuredHistory(
            task.count, str(device), len(source._qdot_buf), len(source._u_buf), source.u_stride
        )
        self.history.reset(torch.arange(task.count, device=device), task.q0)
        self.governor = self.policy.governor(task.kin)
        self.task, self.v_ref = task, task.tip_velocity()
        self.u = torch.zeros(task.count, 3, device=device)

    @torch.no_grad()
    def __call__(self, k: int, q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        from .core import wrap_angle

        self.history.push(q, v, self.u)
        if k % self.policy.settings.decimation == 0:
            task = self.task
            pose, _ = task.kin.pose_jacobian(q)
            reference = task.tip_ref[k]
            requested = torch.zeros_like(self.u)
            requested[:, :2] = self.v_ref[k, :, :2] + self.kp * (reference[:, :2] - pose[:, :2])
            requested[:, 2] = self.kp_angle * wrap_angle(reference[:, 2] - pose[:, 2])
            command, _ = self.governor(q, v, requested)
            self.u = torch.tanh(self.policy(self.history.observe(task.kin, command)))
        return self.u
