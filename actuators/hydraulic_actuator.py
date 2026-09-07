"""
Hydraulic actuator net.

Wraps a trained MLP forward model.
Input  : current joint positions [rad], velocity history [rad/s], and
         normalized valve command history in [-1, 1].
Output : predicted next joint velocity [rad/s]. The network predicts a velocity
         delta; the absolute next velocity is reconstructed here.

The joint count, the command count and the architecture all come from the
model directory's ``model_meta.json`` -- nothing here is fixed to one machine.
One instance drives whatever joint group its model was trained on: a three-joint
excavator arm and a one-joint cabin slew are two instances of this class, not
two classes.

Whether a model exists for a given joint group at all is the caller's problem.
This class either loads one or raises; it has no degraded mode. Keeping the
fallback decision in the sim script is what stops "no model yet" leaking into
every layer.

Joint order is whatever ``qdot_cols`` says in the metadata, and the caller must
feed channels in that order -- see ``joint_names`` / ``command_names``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

# Must mirror training/dataset.py ACTIVATIONS. Both entries are parameterless, so
# the nn.Sequential numbering -- and hence the state_dict keys loaded below with
# strict=True -- is the same whichever is used.
ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh}


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: Sequence[int], activation: str = "relu"):
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(
                f"Unknown activation {activation!r}; expected one of {sorted(ACTIVATIONS)}"
            )
        make_activation = ACTIVATIONS[activation]
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), make_activation()]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Actuator net
# ---------------------------------------------------------------------------


class HydraulicActuatorNet:
    """
    Stateful (history-aware) wrapper around the trained actuator MLP.

    Call reset() at the start of every episode, then predict_velocity() once
    per simulation timestep.

    Parameters
    ----------
    model_dir : path to the directory containing
        model_meta.json, mlp_state_dict.pt, x_mean/std.npy, y_mean/std.npy
    device : torch device string, e.g. "cpu" or "cuda:0"
    """

    def __init__(
        self,
        model_dir: str | Path,
        device: str = "cpu",
        sim_dt: float | None = None,
        weights: str | Path | None = None,
    ) -> None:
        """
        sim_dt : the interval at which predict_velocity() will actually be called.
                 Pass it and a mismatch against the model's training dt is reported
                 loudly. The history buffers assume taps are dt apart, so calling at a
                 different rate silently stretches every horizon by that ratio -- a 1.0 s
                 command history becomes 2.0 s if the sim steps at 0.02 s.
        """
        model_dir = Path(model_dir)

        with open(model_dir / "model_meta.json", encoding="utf-8") as f:
            meta = json.load(f)

        self.dt: float       = meta["dt"]
        self.hist_q: int     = meta["hist_q"]
        self.qdot_stride: int = meta["qdot_stride"]
        self.u_stride: int    = meta["u_stride"]
        self.hist_qdot: int  = meta["hist_qdot"]
        self.hist_u: int     = meta["hist_u"]
        self._has_q: bool    = meta["include_q"]
        self.target_mode: str = meta["target_mode"]
        self.device          = device

        # Widths come from the metadata's column lists, so one class serves any
        # joint group. Commands are counted separately from joints: a three-joint
        # arm driven by four valve channels is a valid model.
        self.joint_names: list[str] = list(meta["qdot_cols"])
        self.position_names: list[str] = list(meta["q_cols"])
        self.command_names: list[str] = list(meta["u_cols"])
        self.num_joints: int = len(self.joint_names)
        self.num_commands: int = len(self.command_names)

        in_dim: int      = meta["model"]["in_dim"]
        out_dim: int     = meta["model"]["out_dim"]
        hidden: list[int] = meta["model"]["hidden"]
        activation: str  = meta["model"]["activation"]
        if self.target_mode != "delta_velocity":
            raise ValueError(
                f"Model target must be 'delta_velocity', got {self.target_mode!r}"
            )
        if len(self.position_names) != self.num_joints:
            raise ValueError(
                f"Model metadata lists {len(self.position_names)} position columns but "
                f"{self.num_joints} velocity columns; position is integrated from velocity"
            )
        if out_dim != self.num_joints:
            raise ValueError(
                f"Model output width {out_dim} does not match its {self.num_joints} "
                "velocity columns"
            )

        self._x_mean = torch.from_numpy(np.load(model_dir / "x_mean.npy")).float().to(device)
        self._x_std  = torch.from_numpy(np.load(model_dir / "x_std.npy")).float().to(device)
        self._y_mean = torch.from_numpy(np.load(model_dir / "y_mean.npy")).float().to(device)
        self._y_std  = torch.from_numpy(np.load(model_dir / "y_std.npy")).float().to(device)

        if self._x_mean.numel() != in_dim:
            raise ValueError(
                f"Normalizer input dim mismatch: model expects {in_dim}, "
                f"x_mean has {self._x_mean.numel()}"
            )

        self._model = MLP(in_dim, out_dim, hidden, activation).to(device)
        weights_path = Path(weights) if weights is not None else model_dir / "mlp_state_dict.pt"
        state = torch.load(weights_path, map_location=device, weights_only=True)
        self._model.load_state_dict(state)
        self._model.eval()

        # History ring-buffers (newest at index 0)
        # Buffers hold raw samples; every stride-th one becomes a network input tap.
        self._q_buf = np.zeros((self.hist_q, self.num_joints), dtype=np.float32)
        self._qdot_buf = np.zeros(
            ((self.hist_qdot - 1) * self.qdot_stride + 1, self.num_joints), dtype=np.float32
        )
        self._u_buf = np.zeros(
            ((self.hist_u - 1) * self.u_stride + 1, self.num_commands), dtype=np.float32
        )
        # Zero is not a valid pose, so the position buffer is seeded from the first
        # observation rather than from reset(). Training pads history by repeating the
        # first row (see make_history_matrix), so this matches what the net saw.
        self._q_primed = False

        if sim_dt is not None and abs(float(sim_dt) - self.dt) > 1e-4:
            ratio = float(sim_dt) / self.dt
            qdot_tap = self.dt * self.qdot_stride
            u_tap = self.dt * self.u_stride
            raise ValueError(
                f"HydraulicActuatorNet timestep mismatch: model dt={self.dt}s, "
                f"controller dt={float(sim_dt)}s ({ratio:.2g}x). "
                f"The model's velocity and command histories would be stretched to "
                f"{qdot_tap*ratio*1000:.0f}/{u_tap*ratio*1000:.0f} ms taps. "
                "Match the simulator control period to the model metadata."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _as_channels(self, value, expected: int, name: str) -> np.ndarray:
        """Coerce one argument to float32 and check its width.

        Without this a wrongly sized argument broadcasts into the ring buffer and
        the model quietly predicts nonsense; the joint count is exactly the thing
        that varies between an arm model and a slew model.
        """
        array = np.asarray(value, dtype=np.float32).reshape(-1)
        if array.shape[0] != expected:
            raise ValueError(
                f"{name} has {array.shape[0]} channels, expected {expected} "
                f"({', '.join(self.joint_names if expected == self.num_joints else self.command_names)})"
            )
        return array

    def reset(self) -> None:
        """Clear history buffers. Call at the start of every episode."""
        self._qdot_buf[:] = 0.0
        self._u_buf[:] = 0.0
        self._q_buf[:] = 0.0
        self._q_primed = False

    def predict_velocity(
        self,
        q_rad: np.ndarray,
        qdot_rad_s: np.ndarray,
        u: np.ndarray,
    ) -> np.ndarray:
        """
        Run one forward pass and return predicted joint velocity [rad/s].

        Updates history buffers with the current observed state.

        Parameters
        ----------
        q_rad      : [num_joints] current joint positions in radians
        qdot_rad_s : [num_joints] current joint velocities in rad/s
        u          : [num_commands] normalized valve commands in [-1, 1]

        Channel order is the metadata's, exposed as :attr:`joint_names` and
        :attr:`command_names`.

        Returns
        -------
        qdot_pred_rad_s : [num_joints] predicted joint velocities in rad/s
        """
        q    = self._as_channels(q_rad,      self.num_joints,   "q_rad")
        qdot = self._as_channels(qdot_rad_s, self.num_joints,   "qdot_rad_s")
        u    = self._as_channels(u,          self.num_commands, "u")

        # Push newest readings into ring buffers
        if not self._q_primed:
            self._q_buf[:] = q            # repeat first observation, as training pads
            self._q_primed = True
        else:
            self._q_buf = np.roll(self._q_buf, shift=1, axis=0)
            self._q_buf[0] = q
        self._qdot_buf = np.roll(self._qdot_buf, shift=1, axis=0)
        self._qdot_buf[0] = qdot
        self._u_buf = np.roll(self._u_buf, shift=1, axis=0)
        self._u_buf[0] = u

        # Build feature vector — must match build_supervised_xy ordering in train.py
        feats: list[np.ndarray] = []
        if self._has_q:
            feats.append(self._q_buf.flatten())                     # [hist_q * 3]
        if self.hist_qdot > 0:
            feats.append(self._qdot_buf[::self.qdot_stride].flatten())
        feats.append(self._u_buf[::self.u_stride].flatten())

        x = np.concatenate(feats)
        if x.shape[0] != self._x_mean.numel():
            raise ValueError(f"Input dim mismatch: built {x.shape[0]}, model expects {self._x_mean.numel()}")

        with torch.no_grad():
            x_t    = torch.from_numpy(x).unsqueeze(0).to(self.device)
            x_norm = (x_t - self._x_mean) / self._x_std
            y_norm = self._model(x_norm)
            model_output = (y_norm * self._y_std + self._y_mean).squeeze(0).cpu().numpy()

        return qdot + model_output

    def predict(
        self,
        q_rad: np.ndarray,
        qdot_rad_s: np.ndarray,
        u: np.ndarray,
    ) -> np.ndarray:
        """
        Predict next joint positions (rad) via Euler integration of predict_velocity().
        """
        q = np.asarray(q_rad, dtype=np.float32)
        qdot_pred = self.predict_velocity(q_rad, qdot_rad_s, u)
        return q + self.dt * qdot_pred

    def predict_torch(
        self,
        q_rad: torch.Tensor,
        qdot_rad_s: torch.Tensor,
        u: torch.Tensor,
    ) -> torch.Tensor:
        """
        Same as predict() but accepts and returns torch Tensors ([num_joints]).
        """
        q_next = self.predict(
            q_rad.cpu().numpy(),
            qdot_rad_s.cpu().numpy(),
            u.cpu().numpy(),
        )
        return torch.from_numpy(q_next).to(q_rad.device)
