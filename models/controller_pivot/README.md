# Default pivot controller

`model_3499.pt` is `pivot_control_seed42` from `weekend_20261113/study`.
It was selected as a fixed representative seed of the pivot method, without
ranking the three seeds on evaluation scores. Checkpoint weights, normalization
and optimizer state are copied unchanged. `provenance.json` records its hash,
source and primary plant fingerprint. `controller_contract.original.json`
retains the source contract; the active contract relocates runtime dependencies
to repository-relative paths.

The policy tracks bucket-pivot translation and absolute bucket angle internally.
External requests remain at the cutting edge and are converted before the
governor. Scores and displayed paths also remain at the cutting edge.

Training used 3,500 PPO updates, seed 42, 4,096 environments, fixed learning rate
3e-4, angular reward 0.5, no linear angular-error penalty, four-step lookahead,
acceleration reward 0.4, noisy gyro observations and 30% moving resets. It used
both `models/actuator_pivot` and the historical `models/actuator` as plants.
The saved YAML records the complete recipe. Playback without future references
supplies zero lookahead, meaning the current request is held.

This package is not hardware validated. `export_robot_bundle.py native` exports it for the
robot's native stack (`ppo_live.py`); the legacy Python-stack exporter rejects it. The
historical tip-policy package remains in `models/controller`.
