"""Learned-hydraulics tip-velocity control; importing this package does not start Isaac.

Modules:

- ``core``: batched actuator-model plant, observations and the checkpoint contract.
- ``kinematics``: USD-derived planar kinematics, collision checks and the command governor.
- ``env``: RSL-RL vectorized training environment (no simulator).
- ``policy``: checkpoint loading for inference.
- ``benchmark``: held-command and trajectory evaluation.
- ``trajectory``, ``draw_ui``, ``gamepad``, ``scene``: interactive playback in Isaac Sim.
"""
