"""
Hydraulic actuator net.

Wraps the trained MLP forward model (boom / arm / bucket).
Input  : current joint positions (rad), velocity history (rad/s), and
         normalized valve command history [-1, 1].
Output : predicted joint velocity (rad/s). The sim-side actuator integrates
         this into position targets via Euler integration.

The model is trained in rad / rad/s units — no unit conversion is performed here.
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

# ---------------------------------------------------------------------------
# MLP — identical architecture to train.py
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: List[int]):
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.Tanh()]
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

    def __init__(self, model_dir: str | Path, device: str = "cpu") -> None:
        model_dir = Path(model_dir)

        with open(model_dir / "model_meta.json", encoding="utf-8") as f:
            meta = json.load(f)

        self.dt: float       = meta["dt"]
        self.hist_qdot: int  = meta["hist_qdot"]
        self.hist_u: int     = meta["hist_u"]
        self._has_q: bool    = meta["include_q"]
        self.device          = device

        in_dim: int      = meta["model"]["in_dim"]
        out_dim: int     = meta["model"]["out_dim"]
        hidden: List[int] = meta["model"]["hidden"]

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
        state = torch.load(model_dir / "mlp_state_dict.pt", map_location=device, weights_only=True)
        self._model.load_state_dict(state)
        self._model.eval()

        # History ring-buffers (newest at index 0)
        self._qdot_buf = np.zeros((self.hist_qdot, self.N_JOINTS), dtype=np.float32)
        self._u_buf    = np.zeros((self.hist_u,    self.N_JOINTS), dtype=np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear history buffers. Call at the start of every episode."""
        self._qdot_buf[:] = 0.0
        self._u_buf[:] = 0.0

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
        self._qdot_buf = np.roll(self._qdot_buf, shift=1, axis=0)
        self._qdot_buf[0] = qdot
        self._u_buf = np.roll(self._u_buf, shift=1, axis=0)
        self._u_buf[0] = u

        # Build feature vector — must match build_supervised_xy ordering in train.py
        feats: List[np.ndarray] = []
        if self._has_q:
            feats.append(q)
        feats.append(self._qdot_buf.flatten())  # [hist_qdot * 3]: t, t-1, ...
        feats.append(self._u_buf.flatten())     # [hist_u    * 3]: t, t-1, ...

        x = np.concatenate(feats)
        if x.shape[0] != self._x_mean.numel():
            raise ValueError(f"Input dim mismatch: built {x.shape[0]}, model expects {self._x_mean.numel()}")

        with torch.no_grad():
            x_t    = torch.from_numpy(x).unsqueeze(0).to(self.device)
            x_norm = (x_t - self._x_mean) / self._x_std
            y_norm = self._model(x_norm)
            qdot_pred = (y_norm * self._y_std + self._y_mean).squeeze(0).cpu().numpy()

        return qdot_pred

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
