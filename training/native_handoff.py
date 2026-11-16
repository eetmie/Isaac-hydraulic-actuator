"""Pin the current native runtime and document the PPO integration contract, without hardware access."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import yaml

from hydraulic_controller.sensors import SENSOR_CONTRACT, policy_joint_offset
from training.overnight import write_json

FILES = (
    "native_stack/README.md",
    "native_stack/include/kaivuri/core.hpp",
    "native_stack/include/kaivuri/pi05_policy.hpp",
    "native_stack/src/core.cpp",
    "native_stack/src/control_main.cpp",
    "native_stack/tools/policy_runtime/bundle.py",
    "native_stack/notes/PROTOCOL.md",
    "configuration_files/profiles/jetson/control_config.yaml",
    "configuration_files/profiles/jetson/servo_config.yaml",
    "configuration_files/profiles/jetson/profile.yaml",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot_repo", required=True, type=Path)
    parser.add_argument("--revision", default="origin/main")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if (args.out / "contract.json").exists():
        raise FileExistsError("Use a fresh directory for a completed native audit")
    args.out.mkdir(parents=True, exist_ok=True)

    def git(*parts):
        return subprocess.check_output(["git", "-C", str(args.robot_repo), *parts])

    commit = git("rev-parse", args.revision).decode().strip()
    hashes = {}
    for name in FILES:
        value = git("show", f"{commit}:{name}")
        target = args.out / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(value)
        hashes[name] = hashlib.sha256(value).hexdigest()
    config = yaml.safe_load((args.out / "source" / FILES[-3]).read_text())
    offsets = policy_joint_offset(config["imu"])
    report = dict(
        target="native_stack",
        source_repository=str(args.robot_repo.resolve()),
        source_revision=commit,
        checked_out_revision=git("rev-parse", "HEAD").decode().strip(),
        source_sha256=hashes,
        native_ppo_loader_verified=False,
        hardware_validated=False,
        policy_q_order=["boom", "arm", "bucket", "carriage_pitch"],
        native_angles_order=["slew", "boom", "arm", "bucket"],
        native_commands_order=["slew", "boom", "arm", "bucket", "trackL", "trackR"],
        policy_to_native_action_indices=[1, 2, 3],
        actor_output_transform="tanh exactly once before normalized native commands; reference_io action is raw actor output",
        policy_joint_offset_rad=offsets.tolist(),
        sensor_contract=SENSOR_CONTRACT,
        sample_dt_s=0.01,
        policy_dt_s=0.05,
        joint_rate_formula="deg2rad([gyro_boom_y-gyro_base_y, gyro_arm_y-gyro_boom_y, gyro_bucket_y-gyro_arm_y, gyro_base_y])",
        reference_fixtures="study/exports/deploy_tip_seed*/bundle/reference_io.npz (produced after training)",
    )
    write_json(args.out / "contract.json", report)
    text = f"""# Native-stack handoff for Monday

Target is the current **native_stack**, pinned at `{commit}` from `{args.revision}`.
The original robot checkout is unchanged. Audited source files and hashes are retained here.

The current GR00T/OpenPI loaders do not implement this PPO observation/history/governor contract.
No native PPO loader, native replay, Jetson latency, or hardware operation has been verified.
The weekend exports use the old Python runtime only as an independent reference replay.

## State and command contract

- Policy angles: `[boom, arm, bucket, carriage_pitch]` in radians. Native `angles_rad` is
  `[slew, boom, arm, bucket]`; slew is **not** the fourth policy angle.
- Native status also exposes role-ordered raw/corrected quaternions and raw gyro vectors
  `[base, boom, arm, bucket]` in degrees/s. Source: `control_main.cpp` status assembly.
  Use same-packet role data and the training `sensors.py` contract, not finite-difference joint velocity.
- For the aligned-Y planar case, rates are
  `deg2rad([boomY-baseY, armY-boomY, bucketY-armY, baseY])`.
- Training-calibration angle offset relative to corrected native joint angles is
  `{json.dumps(offsets.tolist())}` rad. Derive the fourth angle from cabin pitch and
  subtract the sensor contract's carriage zero as well. Raw role quaternions can instead
  go through `imu_positions(..., corrected=False)`. These are planar calibration mappings,
  not a confirmation of operation with roll/slew motion.
- Sample histories at **100 Hz**; run the policy at **20 Hz** and hold its action between updates.
  Histories contain newest sample first: 41 velocity samples (4 channels, select every second)
  and 61 preceding applied-command samples (3 channels, select every third). The resulting
  fixed-tip observation has 164 values. Read actual sizes/strides from each bundle contract.
- Actor outputs raw actions; apply **tanh exactly once** to obtain normalized
  `[boom, arm, bucket]` in `[-1, 1]`. `reference_io.npz` stores raw actor outputs.
  Native command vector is
  `[0, boom, arm, bucket, 0, 0]`; native valve/PWM calibration and control ownership remain in
  the daemon. Record commands actually applied after any gates; do not record PWM microseconds
  or proposed commands as the applied normalized-command history.
- Operator target stays EE pose/twist. Convert native physical EE coordinates to the training
  planar frame; apply physical/training Jacobian conversion where profiles differ, then the
  command governor. Pivot policies additionally convert EE twist internally. Conditioned
  policies additionally require the trained preference scalar; existing exporter rejects both.

## Work before a robot candidate

1. Implement a dedicated native PPO loader/inference adapter, separate from the VLA contracts.
   Port the observation assembly, histories, geometry, governor and timing contract together.
2. Replay each bundle's `reference_io.npz`: FK, Jacobian, admitted twist, full observation and
   actor output. Also replay timestamped sequences covering history rollover and actual commands.
   Python parity alone does not establish native parity.
3. Reconcile active physical profile, training USD, tool pitch convention, frame origins and
   IMU calibration. Weekend exports generate an isolated USD profile for reference replay;
   they do not replace the actual robot profile or prove a match to it.
4. Measure native/Jetson end-to-end latency at 100/20 Hz, packet age and missed-tick behavior.
   Integrate through the existing lease, deadman, stale-state and neutral-on-stop behavior.
5. Choose a candidate from position **and** orientation results, plus valve travel, worst-case
   rotation and recorded failures. Begin with the fixed-tip/no-lookahead family if it qualifies;
   slider/pivot support is additional integration work. Hardware evaluation is a separate step.

The three deployment-oriented policies are included to leave a simpler candidate available;
none is automatically promoted by the training queue.
"""
    (args.out / "README.md").write_text(text, encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
