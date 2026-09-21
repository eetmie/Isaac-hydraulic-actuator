"""Train the hydraulic tip-velocity controller with RSL-RL PPO on the learned actuator plant.

No simulator is launched: the frozen actuator MLP is the plant, and thousands of environments run as batched
Torch on the GPU. From the Isaac Lab root::

    isaaclab.bat -p scripts/Isaac-hydraulic-actuator/train_controller.py --run_name baseline

Checkpoints land in ``logs/rsl_rl/hydraulic_controller/<time>_<run_name>/`` next to ``controller_contract.json``,
which ``run_controller.py`` uses to rebuild the exact plant, kinematics and observation layout.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def agent_config(args, policy_hz: int) -> dict:
    """RSL-RL runner configuration; discounting is defined in seconds so other policy rates stay comparable."""
    steps = max(1, round(args.rollout_s * policy_hz))
    time_scale = 20.0 / policy_hz
    return {
        "class_name": "OnPolicyRunner",
        "seed": args.seed,
        "device": args.device,
        "num_steps_per_env": steps,
        "max_iterations": args.max_iterations,
        "save_interval": args.save_interval,
        "experiment_name": "hydraulic_controller",
        "run_name": args.run_name,
        "logger": "tensorboard",
        "check_for_nan": True,
        "multi_gpu": None,
        "obs_groups": {"actor": ["policy"], "critic": ["critic"]},
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [128, 128],
            "activation": "tanh",
            "obs_normalization": True,
            "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 0.5, "std_type": "scalar"},
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [256, 256],
            "activation": "tanh",
            "obs_normalization": True,
            "distribution_cfg": None,
        },
        "algorithm": {
            "class_name": "PPO",
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 1.0e-3,
            "schedule": "adaptive",
            "gamma": 0.99**time_scale,
            "lam": 0.95**time_scale,
            "entropy_coef": args.entropy_coef,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
            "optimizer": "adam",
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "normalize_advantage_per_mini_batch": False,
            "share_cnn_encoders": False,
            "rnd_cfg": None,
            "symmetry_cfg": None,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run_name", default="controller")
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=1500)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--policy_hz", type=int, default=20, help="Policy rate; must divide 100 Hz.")
    parser.add_argument("--rollout_s", type=float, default=1.2, help="PPO rollout length per env [s].")
    parser.add_argument("--curriculum_iterations", type=int, default=300)
    parser.add_argument("--entropy_coef", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no_randomize", action="store_true", help="Disable plant perturbations and noise.")
    parser.add_argument("--resume", help="Checkpoint to continue from (weights and optimizer).")
    parser.add_argument(
        "--set", nargs="*", default=[], metavar="KEY=VALUE", help="Override HydraulicControlEnvCfg fields."
    )
    args = parser.parse_args()

    import torch
    import yaml
    from rsl_rl.runners import OnPolicyRunner

    from hydraulic_controller.env import HydraulicControlEnv, HydraulicControlEnvCfg

    agent = agent_config(args, args.policy_hz)
    cfg = HydraulicControlEnvCfg(
        num_envs=args.num_envs,
        device=args.device,
        seed=args.seed,
        policy_hz=args.policy_hz,
        randomize=not args.no_randomize,
        curriculum_steps=args.curriculum_iterations * agent["num_steps_per_env"],
    )
    for item in args.set:
        key, _, value = item.partition("=")
        if not hasattr(cfg, key):
            parser.error(f"Unknown environment field: {key}")
        setattr(cfg, key, type(getattr(cfg, key))(yaml.safe_load(value)))

    torch.backends.cuda.matmul.allow_tf32 = True
    env = HydraulicControlEnv(cfg)
    log_dir = (
        ROOT / "logs/rsl_rl/hydraulic_controller" / f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{args.run_name}"
    )
    (log_dir / "params").mkdir(parents=True, exist_ok=True)
    (log_dir / "params/agent.yaml").write_text(yaml.safe_dump(agent, sort_keys=False))
    (log_dir / "params/env.yaml").write_text(
        yaml.safe_dump(json.loads(json.dumps(asdict(cfg))), sort_keys=False)
    )
    (log_dir / "controller_contract.json").write_text(json.dumps(env.contract(), indent=2) + "\n")
    print(f"[INFO] Logging to {log_dir}")
    print(
        f"[INFO] {cfg.num_envs} envs at {cfg.policy_hz} Hz, {agent['num_steps_per_env']} steps/env per iteration, "
        f"observation {env.get_observations()['policy'].shape[1]}, randomize={cfg.randomize}"
    )

    runner = OnPolicyRunner(env, copy.deepcopy(agent), str(log_dir), device=args.device)
    runner.logger.git_status_repos.append(__file__)
    if args.resume:
        runner.load(args.resume)
    runner.learn(args.max_iterations, init_at_random_ep_len=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
