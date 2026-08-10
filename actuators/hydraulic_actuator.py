"""
Hydraulic actuator net.

Wraps the trained MLP forward model (boom / arm / bucket).
Input  : current joint positions (rad), velocity history (rad/s), and
         normalized valve command history [-1, 1].
Output : predicted next joint velocity (rad/s). The network predicts a velocity
         delta; the absolute next velocity is reconstructed here.

The model is trained in rad / rad/s units.
Cabin / slew is NOT covered by this model and must be controlled separately.

Joint order: [boom/lift, arm/tilt, bucket/tool]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: List[int]):
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
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

    N_JOINTS = 3  # boom, arm, bucket

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

        in_dim: int      = meta["model"]["in_dim"]
        out_dim: int     = meta["model"]["out_dim"]
        hidden: List[int] = meta["model"]["hidden"]
        if meta["model"]["activation"] != "relu" or self.target_mode != "delta_velocity":
            raise ValueError("Model must use ReLU and the delta_velocity target")

        self._x_mean = torch.from_numpy(np.load(model_dir / "x_mean.npy")).float().to(device)
        self._x_std  = torch.from_numpy(np.load(model_dir / "x_std.npy")).float().to(device)
        self._y_mean = torch.from_numpy(np.load(model_dir / "y_mean.npy")).float().to(device)
        self._y_std  = torch.from_numpy(np.load(model_dir / "y_std.npy")).float().to(device)

        if self._x_mean.numel() != in_dim:
            raise ValueError(
                f"Normalizer input dim mismatch: model expects {in_dim}, "
                f"x_mean has {self._x_mean.numel()}"
            )

        self._model = MLP(in_dim, out_dim, hidden).to(device)
        weights_path = Path(weights) if weights is not None else model_dir / "mlp_state_dict.pt"
        state = torch.load(weights_path, map_location=device, weights_only=True)
        self._model.load_state_dict(state)
        self._model.eval()

        # History ring-buffers (newest at index 0)
        # Buffers hold raw samples; every stride-th one becomes a network input tap.
        self._q_buf = np.zeros((self.hist_q, self.N_JOINTS), dtype=np.float32)
        self._qdot_buf = np.zeros(
            ((self.hist_qdot - 1) * self.qdot_stride + 1, self.N_JOINTS), dtype=np.float32
        )
        self._u_buf = np.zeros(
            ((self.hist_u - 1) * self.u_stride + 1, self.N_JOINTS), dtype=np.float32
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
        Run one forward pass and return predicted joint velocity in rad/s [3].

        Updates history buffers with the current observed state.

        Parameters
        ----------
        q_rad      : [3] current joint positions in radians
        qdot_rad_s : [3] current joint velocities in rad/s
        u          : [3] normalized valve commands in [-1, 1]
                         order: [lift/boom, tilt/arm, tool/bucket]

        Returns
        -------
        qdot_pred_rad_s : [3] predicted joint velocities in rad/s
        """
        q    = np.asarray(q_rad,      dtype=np.float32)
        qdot = np.asarray(qdot_rad_s, dtype=np.float32)
        u    = np.asarray(u,          dtype=np.float32)

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
        feats: List[np.ndarray] = []
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
        Same as predict() but accepts and returns torch Tensors (shape [3]).
        """
        q_next = self.predict(
            q_rad.cpu().numpy(),
            qdot_rad_s.cpu().numpy(),
            u.cpu().numpy(),
        )
        return torch.from_numpy(q_next).to(q_rad.device)
