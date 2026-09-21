"""Shared, simulator-independent hydraulic recurrence and policy observations."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from actuators import HydraulicActuatorNet

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models/arm_v5_steady"
DEFAULT_ASSET = ROOT / "assets/excavator_bucket_rocking.usd"
JOINT_NAMES = ["revolute_lift", "revolute_tilt", "revolute_tool", "revolute_carriage_pitch"]
HOME = [-0.5498, 1.2549, -0.7540, 0.0]
CONTRACT_VERSION = 2
MAX_ACTION_DELAY_STEPS = 3
OBSERVATION_FIELDS = [
    "q",
    "current_and_past_v_every_20ms",
    "past_commanded_u",
    "tip_xz",
    "sin_pitch",
    "cos_pitch",
    "arm_tip_twist",
    "admitted_command",
    "command_minus_arm_tip_twist",
]


@dataclass
class ControllerSettings:
    """Shared timing and command bounds, in seconds, meters and radians."""

    dt: float = 0.01
    decimation: int = 5
    speed_max: float = 0.12
    pitch_rate_max: float = 0.3
    joint_margin: float = 0.06
    lookahead: float = 0.8
    velocity_limit: float = 2.0

    def __post_init__(self):
        if self.dt != 0.01:
            raise ValueError("This controller contract requires 100 Hz hydraulics")
        if isinstance(self.decimation, bool) or not isinstance(self.decimation, int) or self.decimation < 1:
            raise ValueError("Policy decimation must be a positive integer")
        if any(not math.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError("Controller settings must be finite and positive")

    @property
    def policy_dt(self) -> float:
        """Policy period [s]."""
        return self.dt * self.decimation

    @property
    def policy_hz(self) -> float:
        """Policy update rate [Hz]."""
        return 1.0 / self.policy_dt


def sha256(path: Path) -> str:
    """Fingerprint a model or geometry file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def portable_path(path: Path) -> str:
    """Store a path inside this repository relative to its root, so checkpoints survive a move or clone."""
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def resolve_path(value: str | Path) -> Path:
    """Resolve a stored contract path; relative paths are taken from this repository's root."""
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def model_fingerprint(model: Path) -> dict[str, str]:
    """Fingerprint all files which determine hydraulic inference."""
    return {
        name: sha256(Path(model) / name)
        for name in (
            "model_meta.json",
            "mlp_state_dict.pt",
            "x_mean.npy",
            "x_std.npy",
            "y_mean.npy",
            "y_std.npy",
        )
    }


class HydraulicModelBatch:
    """Frozen actuator with independent histories for a batch of environments."""

    def __init__(self, model: str | Path, count: int, device: str, dt: float = 0.01):
        self.path = Path(model).resolve()
        self.meta = json.loads((self.path / "model_meta.json").read_text())
        if self.meta.get("simulation_joint_names") != JOINT_NAMES:
            raise ValueError("Controller requires boom/arm/bucket/carriage-pitch motion channels")
        source = HydraulicActuatorNet(self.path, device=device, sim_dt=dt)
        if source.num_commands != 3:
            raise ValueError("Controller requires three hydraulic valve channels")
        self.source = source
        self.net = source._model.requires_grad_(False).eval()
        self.q_history = torch.zeros(count, source.hist_q, 4, device=device)
        self.v_history = torch.zeros(count, len(source._qdot_buf), 4, device=device)
        self.u_history = torch.zeros(count, len(source._u_buf), 3, device=device)
        self.primed = torch.zeros(count, dtype=torch.bool, device=device)

    def reset(self, ids: torch.Tensor) -> None:
        """Clear only the specified environment histories."""
        self.q_history[ids] = 0
        self.v_history[ids] = 0
        self.u_history[ids] = 0
        self.primed[ids] = False

    @torch.no_grad()
    def predict(self, q: torch.Tensor, v: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Predict next velocity [rad/s] from q [rad], v [rad/s], valves [-1, 1]."""
        for history, current in ((self.q_history, q), (self.v_history, v), (self.u_history, u)):
            history[:, 1:] = history[:, :-1].clone()
            history[:, 0] = current
        self.q_history[~self.primed] = q[~self.primed, None, :]
        self.primed[:] = True
        src = self.source
        features = []
        if src._has_q:
            features.append(self.q_history.flatten(1))
        if src.hist_qdot:
            features.append(self.v_history[:, :: src.qdot_stride].flatten(1))
        features.append(self.u_history[:, :: src.u_stride].flatten(1))
        x = torch.cat(features, dim=1)
        prediction = self.net((x - src._x_mean) / src._x_std) * src._y_std + src._y_mean
        return v + prediction if src.target_mode == "delta_velocity" else prediction


class HydraulicPlant:
    """Authoritative learned state, shared by training, benchmark and Isaac playback.

    The policy commands ``u_cmd``. The actuator model receives ``u``, which may differ by a hidden per-environment
    valve gain, offset and transport delay (all identity by default). Observations only expose ``u_cmd``, as on
    the real machine where the controller knows what it sent, not what the spool did.

    ``speed_scale`` multiplies the model's joint velocities before integration, while the network keeps its own
    unscaled history. It emulates the measured model-vs-machine speed spread without leaving the model's data range.
    """

    def __init__(
        self,
        model: str | Path,
        kinematics,
        count: int,
        device: str,
        settings: ControllerSettings | None = None,
    ):
        self.settings = settings or ControllerSettings()
        self.device = device
        self.count = count
        self.kinematics = kinematics
        self.model = HydraulicModelBatch(model, count, device, self.settings.dt)
        self.q = torch.tensor(HOME, device=device).repeat(count, 1)
        self.v = torch.zeros_like(self.q)
        self._v_model = torch.zeros_like(self.q)
        self.speed_scale = torch.ones_like(self.q)
        self.u = torch.zeros(count, 3, device=device)
        self.u_cmd = torch.zeros_like(self.u)
        self.u_cmd_history = torch.zeros_like(self.model.u_history)
        self.valve_gain = torch.ones_like(self.u)
        self.valve_offset = torch.zeros_like(self.u)
        self.action_delay = torch.zeros(count, dtype=torch.long, device=device)
        self._queue = torch.zeros(count, MAX_ACTION_DELAY_STEPS + 1, 3, device=device)
        self._rows = torch.arange(count, device=device)
        self.latched = torch.zeros(count, 3, dtype=torch.int8, device=device)
        self.invalid = torch.zeros(count, dtype=torch.bool, device=device)
        self.limit_hit = self.invalid.clone()
        self.velocity_clipped = self.invalid.clone()
        self.steps = 0

    def reset(self, ids: torch.Tensor, q: torch.Tensor | None = None) -> None:
        """Reset selected environments to q [rad] at rest with neutral valve histories."""
        self.q[ids] = torch.tensor(HOME, device=self.device) if q is None else q
        self.v[ids] = 0
        self._v_model[ids] = 0
        self.u[ids] = 0
        self.u_cmd[ids] = 0
        self.u_cmd_history[ids] = 0
        self._queue[ids] = 0
        self.latched[ids] = 0
        self.invalid[ids] = False
        self.limit_hit[ids] = False
        self.velocity_clipped[ids] = False
        self.model.reset(ids)

    def begin_action(self, actions: torch.Tensor) -> None:
        """Map unconstrained policy actions to normalized valve commands with tanh."""
        self.invalid |= ~torch.isfinite(actions).all(dim=1)
        self.u_cmd.copy_(torch.tanh(torch.nan_to_num(actions)))
        self.limit_hit.zero_()
        self.velocity_clipped.zero_()

    @torch.no_grad()
    def step(self) -> None:
        """Advance exactly one hydraulic timestep [0.01 s]."""
        cfg = self.settings
        self._queue[:, 1:] = self._queue[:, :-1].clone()
        self._queue[:, 0] = self.u_cmd
        delayed = self._queue[self._rows, self.action_delay]
        self.u.copy_((self.valve_gain * delayed + self.valve_offset).clamp(-1.0, 1.0))
        self.u_cmd_history[:, 1:] = self.u_cmd_history[:, :-1].clone()
        self.u_cmd_history[:, 0] = self.u_cmd
        model_velocity = self.model.predict(self.q, self._v_model, self.u)
        finite = torch.isfinite(model_velocity).all(dim=1)
        self.invalid |= ~finite
        model_velocity = torch.where(finite[:, None], model_velocity, 0.0)
        self.velocity_clipped |= (model_velocity.abs() > cfg.velocity_limit).any(dim=1)
        model_velocity = model_velocity.clamp(-cfg.velocity_limit, cfg.velocity_limit)
        limits = self.kinematics.limits
        # Vectorized equivalent of the demo's EndStopGuard, including arrival/reversal semantics.
        pushing = torch.where(self.u > 0.05, 1, torch.where(self.u < -0.05, -1, 0))
        self.latched[self.latched != pushing] = 0
        lower = (self.q[:, :3] <= limits[:3, 0] + 1e-6) & (pushing == -1)
        upper = (self.q[:, :3] >= limits[:3, 1] - 1e-6) & (pushing == 1)
        self.latched[lower] = -1
        self.latched[upper] = 1
        model_velocity[:, :3] = torch.where(self.latched != 0, 0.0, model_velocity[:, :3])
        velocity = model_velocity * self.speed_scale
        proposed = self.q + cfg.dt * velocity
        self.latched[(proposed[:, :3] < limits[:3, 0]) & (pushing == -1)] = -1
        self.latched[(proposed[:, :3] > limits[:3, 1]) & (pushing == 1)] = 1
        clamped = proposed.clamp(limits[:, 0], limits[:, 1])
        hit = clamped != proposed
        self.limit_hit |= hit.any(dim=1)
        self.v.copy_(torch.where(hit, 0.0, velocity))
        self._v_model.copy_(torch.where(hit, 0.0, model_velocity))
        self.q.copy_(clamped)
        self.steps += 1

    def tip_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return tip pose [m, m, rad] and the tip twist [m/s, m/s, rad/s] driven by the three arm joints.

        Carriage pitch is excluded: rocking cannot be controlled through the valves, so it is not tracked.
        """
        pose, jacobian = self.kinematics.pose_jacobian(self.q)
        twist = torch.einsum("nij,nj->ni", jacobian[:, :, :3], self.v[:, :3])
        return pose, twist

    def observe(self, command: torch.Tensor) -> torch.Tensor:
        """Build causal policy observations; command is [m/s, m/s, rad/s]."""
        from .observations import assemble_observation

        # Model history ends at the previous step's input velocity. Prepend the current velocity.
        history_v = torch.cat(
            (self.v[:, None], self.model.v_history[:, :-1] * self.speed_scale[:, None]), dim=1
        )
        return assemble_observation(
            self.q, history_v, self.u_cmd_history, self.model.source.u_stride, self.kinematics, command
        )

    def contract(self) -> dict:
        """Describe all runtime assumptions needed to interpret a checkpoint."""
        return {
            "version": CONTRACT_VERSION,
            "settings": asdict(self.settings),
            "policy_hz": self.settings.policy_hz,
            "model_path": portable_path(self.model.path),
            "model_files": model_fingerprint(self.model.path),
            "asset_path": portable_path(self.kinematics.path),
            "asset_sha256": sha256(self.kinematics.path),
            "joint_names": JOINT_NAMES,
            "tip_path": "/excavator/bucket/ee_tip",
            "command_frame": "lower_carriage_XZ_pitch_about_positive_Y",
            "tracked_twist": "arm_joints_only",
            "action_transform": "tanh",
            "observation_dim": self.observe(self.u * 0).shape[1],
            "observation_fields": OBSERVATION_FIELDS,
        }


@torch.no_grad()
def measure_joint_speed_limits(
    model: str | Path, kinematics, poses: torch.Tensor, settings: ControllerSettings | None = None
) -> torch.Tensor:
    """Median steady joint speed [rad/s] at full valve opening, shape [3, 2] = (negative, positive) per joint.

    Measured on the nominal actuator model from ``poses`` [rad], so it reflects hydraulic flow saturation (e.g. a
    slow boom-down) rather than valve saturation.
    """
    count, device = len(poses), poses.device
    limits = torch.zeros(3, 2, device=device)
    plant = HydraulicPlant(model, kinematics, count, str(device), settings)
    ids = torch.arange(count, device=device)
    for joint in range(3):
        for column, sign in enumerate((-1.0, 1.0)):
            plant.reset(ids, poses)
            action = torch.zeros(count, 3, device=device)
            action[:, joint] = sign * 5.0
            plant.begin_action(action)
            speeds = []
            for step in range(80):
                plant.step()
                if step >= 40:
                    speeds.append(plant.v[:, joint].abs())
            limits[joint, column] = torch.stack(speeds).mean(0).median()
    return limits


def wrap_angle(value: torch.Tensor) -> torch.Tensor:
    """Wrap angles [rad] to [-pi, pi]."""
    return torch.atan2(torch.sin(value), torch.cos(value))
