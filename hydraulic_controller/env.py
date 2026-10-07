"""Vectorized PPO environment: learned hydraulics, planar tip-velocity tracking, no physics engine.

The frozen actuator network *is* the plant, so training needs neither PhysX nor rendering. Isaac Sim is only used
to visualize a trained policy (see ``run_controller.py``). The task follows Egli & Hutter (RA-L 2022): the policy
receives a desired tip twist and outputs pilot valve commands directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from .core import (
    DEFAULT_ASSET,
    DEFAULT_MODEL,
    HOME,
    MAX_ACTION_DELAY_STEPS,
    ControllerSettings,
    HydraulicPlant,
    measure_joint_speed_limits,
    resolve_path,
)
from .kinematics import CommandGovernor, ExcavatorKinematics


@dataclass
class HydraulicControlEnvCfg:
    """Task, reward and randomization settings. Units: m, rad, s unless stated."""

    model_path: str = str(DEFAULT_MODEL)
    # Further actuator networks; environment i runs model i % (1 + len(extra)), so no policy fits a single twin.
    extra_model_paths: tuple[str, ...] = ()
    asset_path: str = str(DEFAULT_ASSET)
    num_envs: int = 4096
    device: str = "cuda:0"
    seed: int = 42
    policy_hz: int = 20
    episode_length_s: float = 6.0

    # Commands: a target twist is held for a random time; the requested twist slews toward it.
    speed_max: float = 0.12
    speed_min: float = 0.01
    pitch_rate_max: float = 0.3
    zero_speed_prob: float = 0.1
    zero_pitch_rate_prob: float = 0.4
    command_hold_s: tuple[float, float] = (1.0, 3.0)
    # "uniform": speed_min..speed_max regardless of pose. "achievable": a fraction of what the plant can reach at
    # the current pose in that direction (speed_limits.SpeedTable, bucket angle held), mostly well inside the limit
    # and with over_limit_prob just beyond it, clipped to [achievable_speed_min, speed_max].
    command_speed_mode: str = "uniform"
    uniform_mix_prob: float = (
        0.3  # in "achievable" mode, this share still draws speed_min..speed_max, pose-blind
    )
    achievable_fraction: tuple[float, float] = (0.05, 0.9)
    over_limit_prob: float = 0.1
    over_limit_fraction: tuple[float, float] = (1.0, 1.3)
    achievable_speed_min: float = 0.002
    command_step_prob: float = 0.1
    linear_accel: float = 0.25
    angular_accel: float = 1.0
    reset_joint_margin: float = 0.1
    reset_pitch_std: float = 0.002
    governor_joint_speed_margin: float = 0.8

    # Reward. Tracking terms are per 0.05 s of simulated time; action-rate terms are per unit valve travel.
    curriculum_steps: int = 300 * 24
    track_sigma: tuple[float, float] = (0.08, 0.03)
    track_fine_sigma: float = 0.01
    track_fine_weight: float = 0.5
    angular_weight: float = 0.5
    angular_sigma: tuple[float, float] = (0.3, 0.1)
    action_rate_l1: float = 0.5
    action_rate_l2: float = 1.0
    action_rate_scale: tuple[float, float] = (0.2, 1.0)
    effort: float = 0.02
    # Linear error penalties (per speed_max / pitch_rate_max of error). The Gaussian tracking terms fade beyond ~2
    # sigma; these keep a gradient when a command is out of reach, so the compromise there is learned, not random.
    linear_error_weight: float = 0.0
    angular_error_weight: float = 0.0
    termination_penalty: float = 10.0

    # Tracked point: "tip" (bucket tip) or "pivot" (bucket joint). With "pivot" the bucket joint only turns the
    # bucket; tip requests are converted outside the policy (``policy.PolicyController``).
    tracked_point: str = "tip"
    # Episodes that start mid-motion: this share of resets continues a state from a pool driven for
    # motion_start_s by a joint-space P controller toward random poses, so histories and the arm are already
    # moving; the request starts at that motion and slews to the new target. The pool is redriven every
    # motion_pool_refresh_steps policy steps.
    motion_start_prob: float = 0.0
    motion_start_s: float = 1.2
    motion_pool_size: int = 4096
    motion_pool_refresh_steps: int = 120

    # Hidden per-episode plant perturbations and observation noise (sim-to-real robustness).
    randomize: bool = True
    valve_gain_range: tuple[float, float] = (0.9, 1.1)
    valve_offset_max: float = 0.02
    joint_speed_scale_range: tuple[float, float] = (1.0, 1.0)
    action_delay_max_steps: int = 2
    noise_q: float = 0.003
    noise_v: float = 0.02
    noise_pose: float = 0.002
    noise_twist: float = 0.005
    gyro_observations: bool = False
    gyro_noise_std: float = 0.005
    gyro_bias_max: float = 0.003
    sensor_delay_max_steps: int = 2

    @property
    def settings(self) -> ControllerSettings:
        if 100 % self.policy_hz or not 5 <= self.policy_hz <= 100:
            raise ValueError("policy_hz must divide the 100 Hz hydraulic rate")
        return ControllerSettings(
            decimation=100 // self.policy_hz, speed_max=self.speed_max, pitch_rate_max=self.pitch_rate_max
        )


class HydraulicControlEnv(VecEnv):
    """RSL-RL vectorized environment around :class:`HydraulicPlant`."""

    def __init__(self, cfg: HydraulicControlEnvCfg):
        if cfg.action_delay_max_steps > MAX_ACTION_DELAY_STEPS:
            raise ValueError(f"action_delay_max_steps must be <= {MAX_ACTION_DELAY_STEPS}")
        self.cfg = cfg
        self.device = cfg.device
        self.num_envs = cfg.num_envs
        self.num_actions = 3
        self.settings = cfg.settings
        self.decimation = self.settings.decimation
        self.policy_dt = self.settings.policy_dt
        self.time_scale = 0.05 / self.policy_dt
        self.max_episode_length = round(cfg.episode_length_s / self.policy_dt)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.common_step_counter = 0
        torch.manual_seed(cfg.seed)

        self.kin = ExcavatorKinematics(cfg.asset_path, self.device)
        self.kin.build_collision_grid()
        extra = tuple(resolve_path(path) for path in cfg.extra_model_paths)
        self.plant = HydraulicPlant(
            cfg.model_path, self.kin, self.num_envs, self.device, self.settings, extra, cfg.tracked_point
        )
        self.speed_table = None
        if cfg.command_speed_mode == "achievable":
            from .core import ROOT
            from .speed_limits import SpeedTable

            self.speed_table = SpeedTable.load_or_build(cfg.model_path, self.kin, ROOT / "runs/speed_tables")
        elif cfg.command_speed_mode != "uniform":
            raise ValueError(f"unknown command_speed_mode {cfg.command_speed_mode!r}")
        from .observations import SensorObservation, SensorSettings

        self.sensors = None
        if cfg.gyro_observations:
            settings = SensorSettings(
                cfg.gyro_noise_std, cfg.gyro_bias_max, cfg.noise_q, cfg.sensor_delay_max_steps
            )
            if not cfg.randomize:
                settings = SensorSettings(0.0, 0.0, 0.0, 0)
            self.sensors = SensorObservation(self.plant, settings)
        self.joint_speed_limits = None
        if cfg.governor_joint_speed_margin > 0:
            self.joint_speed_limits = measure_joint_speed_limits(
                cfg.model_path, self.kin, self._sample_poses(64), self.settings
            )
        self.governor = CommandGovernor(
            self.kin,
            self.settings,
            self.joint_speed_limits,
            cfg.governor_joint_speed_margin,
            cfg.tracked_point,
        )
        self.motion_pool = None
        if cfg.motion_start_prob > 0:
            self.motion_pool = MotionStartPool(self, extra)

        n = self.num_envs
        self.target = torch.zeros(n, 3, device=self.device)
        self.requested = torch.zeros_like(self.target)
        self.command = torch.zeros_like(self.target)
        self.intervention = torch.zeros(n, device=self.device)
        self.hold_timer = torch.zeros(n, device=self.device)
        self._noise = self._noise_scale() if cfg.randomize else None
        self._reset_idx(torch.arange(n, device=self.device))
        self._update_command(resample=False)
        self._obs = self._compute_observations()

    # ---- VecEnv -------------------------------------------------------------------------------------------------

    def get_observations(self) -> TensorDict:
        return self._obs

    def contract(self) -> dict:
        """Plant contract plus the governor configuration needed to reproduce admitted commands."""
        contract = self.plant.contract()
        if self.sensors is not None:
            from .sensors import SENSOR_CONTRACT

            contract["sensors"] = SENSOR_CONTRACT
            contract["sensor_randomization"] = self.sensors.contract()
        if self.motion_pool is not None:
            contract["motion_start"] = {
                "prob": self.cfg.motion_start_prob,
                "seconds": self.cfg.motion_start_s,
            }
        if self.joint_speed_limits is not None:
            contract["joint_speed_limits_rad_s"] = self.joint_speed_limits.tolist()
            contract["governor_joint_speed_margin"] = self.cfg.governor_joint_speed_margin
        return contract

    @torch.no_grad()
    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        cfg, plant = self.cfg, self.plant
        progress = min(1.0, self.common_step_counter / max(1, cfg.curriculum_steps))

        def lerp(pair):
            return pair[0] + (pair[1] - pair[0]) * progress

        sigma, sigma_w = lerp(cfg.track_sigma), lerp(cfg.angular_sigma)
        previous = plant.u_cmd.clone()
        plant.begin_action(actions)
        du = plant.u_cmd - previous

        track = torch.zeros(self.num_envs, device=self.device)
        angular = torch.zeros_like(track)
        speed_error = torch.zeros_like(track)
        angle_rate_error = torch.zeros_like(track)
        collided = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for _ in range(self.decimation):
            plant.step()
            if self.sensors is not None:
                self.sensors.step()
            _, twist = plant.tip_state()
            error = twist - self.command
            e2 = error[:, :2].square().sum(1)
            track += torch.exp(-e2 / sigma**2) + cfg.track_fine_weight * torch.exp(
                -e2 / cfg.track_fine_sigma**2
            )
            angular += torch.exp(-error[:, 2].square() / sigma_w**2)
            speed_error += e2.sqrt()
            angle_rate_error += error[:, 2].abs()
            collided |= self.kin.colliding(plant.q)
        for value in (track, angular, speed_error, angle_rate_error):
            value /= self.decimation

        rate_scale = lerp(cfg.action_rate_scale)
        reward = self.time_scale * (
            track + cfg.angular_weight * angular - cfg.effort * plant.u_cmd.square().sum(1)
        )
        reward -= rate_scale * (
            cfg.action_rate_l1 * du.abs().sum(1) + cfg.action_rate_l2 * du.square().sum(1)
        )
        reward -= self.time_scale * (
            cfg.linear_error_weight * speed_error / cfg.speed_max
            + cfg.angular_error_weight * angle_rate_error / cfg.pitch_rate_max
        )
        terminated = plant.invalid | plant.limit_hit | collided
        reward -= cfg.termination_penalty * terminated.float()

        self.episode_length_buf += 1
        timeout = (self.episode_length_buf >= self.max_episode_length) & ~terminated
        dones = terminated | timeout
        moving = self.command[:, :2].norm(dim=1) > 0.005
        extras = {
            "time_outs": timeout,
            "log": {
                "Tracking/speed_error_mm_s": speed_error.mean() * 1000,
                "Tracking/speed_error_moving_mm_s": (speed_error * moving).sum()
                / moving.sum().clamp_min(1)
                * 1000,
                "Tracking/angular_rate_error": angle_rate_error.mean(),
                "Tracking/sigma_mm_s": torch.tensor(sigma * 1000),
                "Valves/du_mean": du.abs().mean(),
                "Valves/du_p99": torch.quantile(du.abs().flatten(), 0.99),
                "Valves/saturated": (plant.u_cmd.abs() > 0.98).float().mean(),
                "Valves/abs_mean": plant.u_cmd.abs().mean(),
                "Termination/limit": plant.limit_hit.float().mean(),
                "Termination/collision": collided.float().mean(),
                "Termination/invalid": plant.invalid.float().mean(),
                "Command/governor_fraction": self.intervention.mean(),
                "Curriculum/progress": torch.tensor(progress),
            },
        }
        done_ids = dones.nonzero().flatten()
        if len(done_ids):
            self._reset_idx(done_ids)
        self._update_command(resample=True)
        self._obs = self._compute_observations()
        self.common_step_counter += 1
        if self.motion_pool is not None and self.common_step_counter % cfg.motion_pool_refresh_steps == 0:
            self.motion_pool.refresh()
        return self._obs, reward, dones.long(), extras

    # ---- Task internals -----------------------------------------------------------------------------------------

    def _noise_scale(self) -> torch.Tensor:
        """Per-feature observation noise amplitude, laid out as :meth:`HydraulicPlant.observe`."""
        cfg = self.cfg
        v_taps = (self.plant.model.v_history.shape[1] + 1) // 2
        u_taps = len(range(0, self.plant.u_cmd_history.shape[1], self.plant.model.source.u_stride))
        parts = [
            (4, cfg.noise_q),
            (4 * v_taps, cfg.noise_v),
            (3 * u_taps, 0.0),
            (2, cfg.noise_pose),
            (2, cfg.noise_q),
            (3, cfg.noise_twist),
            (3, 0.0),
            (3, cfg.noise_twist),
        ]
        scale = torch.cat([torch.full((count,), value, device=self.device) for count, value in parts])
        if len(scale) != self.plant.observe(self.command[:1].expand(self.num_envs, 3)).shape[1]:
            raise RuntimeError("Observation noise layout does not match HydraulicPlant.observe")
        return scale

    def _compute_observations(self) -> TensorDict:
        clean = self.plant.observe(self.command)
        policy = clean
        if self.sensors is not None:
            policy = self.sensors.observe(self.command)
        elif self._noise is not None:
            policy = clean + (2 * torch.rand_like(clean) - 1) * self._noise
        plant = self.plant
        privileged = torch.cat(
            (
                plant.valve_gain - 1,
                plant.valve_offset,
                plant.speed_scale[:, :3] - 1,
                plant.action_delay[:, None].float() / MAX_ACTION_DELAY_STEPS,
            ),
            dim=1,
        )
        critic = torch.cat((clean, privileged), dim=1)
        return TensorDict({"policy": policy, "critic": critic}, batch_size=[self.num_envs])

    def _sample_poses(self, count: int) -> torch.Tensor:
        """Valid random arm poses [rad] with carriage pitch near its recorded rest value."""
        cfg, kin = self.cfg, self.kin
        q = torch.tensor(HOME, device=self.device).repeat(count, 1)
        pending = torch.arange(count, device=self.device)
        lo = kin.limits[:3, 0] + cfg.reset_joint_margin
        hi = kin.limits[:3, 1] - cfg.reset_joint_margin
        for _ in range(10):
            if not len(pending):
                break
            samples = torch.empty(len(pending), 4, device=self.device)
            samples[:, :3] = lo + (hi - lo) * torch.rand(len(pending), 3, device=self.device)
            samples[:, 3] = torch.randn(len(pending), device=self.device) * cfg.reset_pitch_std
            good = kin.valid(samples, self.settings.joint_margin)
            q[pending[good]] = samples[good]
            pending = pending[~good]
        return q

    def _reset_idx(self, ids: torch.Tensor) -> None:
        cfg = self.cfg
        count = len(ids)
        self.plant.reset(ids, self._sample_poses(count))
        self.episode_length_buf[ids] = 0
        if cfg.randomize:
            low, high = cfg.valve_gain_range
            self.plant.valve_gain[ids] = low + (high - low) * torch.rand(count, 3, device=self.device)
            self.plant.valve_offset[ids] = (
                2 * torch.rand(count, 3, device=self.device) - 1
            ) * cfg.valve_offset_max
            low, high = cfg.joint_speed_scale_range
            self.plant.speed_scale[ids, :3] = low + (high - low) * torch.rand(count, 3, device=self.device)
            self.plant.action_delay[ids] = torch.randint(
                0, cfg.action_delay_max_steps + 1, (count,), device=self.device
            )
        moving = ids[:0]
        if self.motion_pool is not None:
            moving = ids[torch.rand(count, device=self.device) < cfg.motion_start_prob]
            moving = self.motion_pool.load(moving)
        if self.sensors is not None:
            self.sensors.reset(ids)
            self.sensors.prime(moving)
        self.requested[ids] = 0
        if len(moving):
            # The request starts at the motion already under way (a stale command), then slews to the new target.
            _, twist = self.plant.tip_state()
            twist = twist[moving]
            twist[:, :2] *= (
                cfg.speed_max / twist[:, :2].norm(dim=1, keepdim=True).clamp_min(1e-9)
            ).clamp_max(1)
            twist[:, 2].clamp_(-cfg.pitch_rate_max, cfg.pitch_rate_max)
            self.requested[moving] = twist
        self._sample_targets(ids)

    def _sample_targets(self, ids: torch.Tensor) -> None:
        cfg = self.cfg
        count = len(ids)
        direction = torch.rand(count, device=self.device) * (2 * torch.pi)
        if self.speed_table is None:
            speed = cfg.speed_min + (cfg.speed_max - cfg.speed_min) * torch.rand(count, device=self.device)
        else:
            reach = self.speed_table.lookup(self.plant.q[ids], direction)
            low, high = cfg.achievable_fraction
            fraction = low + (high - low) * torch.rand(count, device=self.device)
            over = torch.rand(count, device=self.device) < cfg.over_limit_prob
            low, high = cfg.over_limit_fraction
            fraction[over] = low + (high - low) * torch.rand(int(over.sum()), device=self.device)
            speed = (fraction * reach).clamp(cfg.achievable_speed_min, cfg.speed_max)
            # Keep covering pose-blind requests (static-speed paths, gamepad), far beyond the limit at hard poses.
            blind = torch.rand(count, device=self.device) < cfg.uniform_mix_prob
            uniform = cfg.speed_min + (cfg.speed_max - cfg.speed_min) * torch.rand(count, device=self.device)
            speed = torch.where(blind, uniform, speed)
        speed *= torch.rand(count, device=self.device) >= cfg.zero_speed_prob
        angular = (2 * torch.rand(count, device=self.device) - 1) * cfg.pitch_rate_max
        angular *= torch.rand(count, device=self.device) >= cfg.zero_pitch_rate_prob
        self.target[ids] = torch.stack((speed * direction.cos(), speed * direction.sin(), angular), dim=1)
        low, high = cfg.command_hold_s
        self.hold_timer[ids] = low + (high - low) * torch.rand(count, device=self.device)
        step = torch.rand(count, device=self.device) < cfg.command_step_prob
        self.requested[ids[step]] = self.target[ids[step]]

    def _update_command(self, resample: bool) -> None:
        cfg, dt = self.cfg, self.policy_dt
        if resample:
            self.hold_timer -= dt
            expired = (self.hold_timer <= 0).nonzero().flatten()
            if len(expired):
                self._sample_targets(expired)
        delta = self.target - self.requested
        linear = delta[:, :2]
        linear = linear * (
            cfg.linear_accel * dt / linear.norm(dim=1, keepdim=True).clamp_min(1e-9)
        ).clamp_max(1)
        self.requested[:, :2] += linear
        self.requested[:, 2] += delta[:, 2].clamp(-cfg.angular_accel * dt, cfg.angular_accel * dt)
        self.command[:], self.intervention[:] = self.governor(self.plant.q, self.plant.v, self.requested)


class MotionStartPool:
    """States already in motion, for episodes that should not start at rest (Egli & Hutter start theirs after
    1.2 s of a position controller). A separate batch of the same plant, with its own model per row matching
    the environments' (row ``i`` runs model ``i % models``), is driven by a joint-space P controller toward
    random nearby poses. Rows that end invalid, at an end stop or in collision are never handed out.
    """

    def __init__(self, env: HydraulicControlEnv, extra_models: tuple):
        cfg = env.cfg
        self.env = env
        self.models = 1 + len(extra_models)
        size = cfg.motion_pool_size - cfg.motion_pool_size % self.models
        if size < self.models:
            raise ValueError("motion_pool_size must hold at least one row per actuator model")
        self.plant = HydraulicPlant(cfg.model_path, env.kin, size, env.device, env.settings, extra_models)
        self.ticks = round(cfg.motion_start_s / env.settings.dt)
        self.good = torch.zeros(size, dtype=torch.bool, device=env.device)
        self.refresh()

    @torch.no_grad()
    def refresh(self) -> None:
        env, plant = self.env, self.plant
        size, device = plant.count, env.device
        rows = torch.arange(size, device=device)
        start = env._sample_poses(size)
        plant.reset(rows, start)
        limits = env.kin.limits[:3]
        margin = env.cfg.reset_joint_margin
        goal = (start[:, :3] + (2 * torch.rand(size, 3, device=device) - 1) * 0.6).clamp(
            limits[:, 0] + margin, limits[:, 1] - margin
        )
        gain = 1.0 + 4.0 * torch.rand(size, 1, device=device)
        reach = 0.3 + 0.7 * torch.rand(size, 1, device=device)  # largest valve opening per row
        bad = torch.zeros(size, dtype=torch.bool, device=device)
        # A raw action of atanh(u) makes the plant's tanh send exactly u.
        for tick in range(self.ticks):
            u = (gain * (goal - plant.q[:, :3])).clamp(-1, 1) * reach
            plant.begin_action(torch.atanh(u.clamp(-0.999, 0.999)))
            plant.step()
            bad |= plant.invalid | plant.limit_hit
            if tick % 10 == 9 or tick == self.ticks - 1:  # the arm moves well under a link width in 0.1 s
                bad |= env.kin.colliding(plant.q)
        self.good = ~bad

    def load(self, ids: torch.Tensor) -> torch.Tensor:
        """Continue environments ``ids`` from random good pool rows of their own model; returns those loaded."""
        if not len(ids):
            return ids
        group = ids % self.models
        loaded = torch.zeros(len(ids), dtype=torch.bool, device=ids.device)
        rows = torch.zeros_like(ids)
        for k in range(self.models):
            mine = (group == k).nonzero().flatten()
            candidates = (
                self.good & (torch.arange(len(self.good), device=ids.device) % self.models == k)
            ).nonzero()
            if not len(mine) or not len(candidates):
                continue
            pick = torch.randint(len(candidates), (len(mine),), device=ids.device)
            rows[mine] = candidates.flatten()[pick]
            loaded[mine] = True
        ids, rows = ids[loaded], rows[loaded]
        self.env.plant.copy_state(ids, self.plant, rows)
        return ids
