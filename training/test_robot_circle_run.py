"""run_robot_circle.py run() end to end against a fake robot: valve write path, arming and logging."""

import csv
import json
import sys
import threading
import time
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import run_robot_circle
from hydraulic_controller.core import HOME

ROOT = Path(__file__).resolve().parents[1]
ROBOT_REPO = ROOT.parent / "kaivuriprokkis"
LOGS = ROOT / "drive_logs/20261007_circle_mlp_vs_pid"
PROFILE = "jetson_bucket_dz0"
pytestmark = pytest.mark.skipif(
    not (ROBOT_REPO / "configuration_files/profiles" / PROFILE).is_dir(), reason="robot repo missing"
)
RATE = 0.3  # fake joint speed [rad/s] per unit valve; positive valves raise the angle, as on the robot


class FakeHardware:
    """Integrates written valves into joint motion; IMU reads see the result."""

    def __init__(self, **kwargs):
        self.q = np.array(HOME, dtype=np.float64)
        self.u = np.zeros(3)
        self.last = time.monotonic()
        self.lock = threading.Lock()
        self.writes = []  # (thread name, boom/arm/bucket)
        self._imu_joint_roles = ("boom", "arm", "bucket")

    def send_named_pwm_commands(self, commands, **kwargs):
        with self.lock:
            self.advance()
            self.u = np.array([commands[name] for name in ("boom", "arm", "bucket")])
            self.writes.append((threading.current_thread().name, self.u.copy()))
        return True

    def advance(self):
        now = time.monotonic()
        self.q[:3] += RATE * self.u * (now - self.last)
        self.last = now

    def set_pump_enabled(self, enabled):
        return True

    def reset(self, reset_pump=False):
        with self.lock:
            self.advance()
            self.u[:] = 0

    def is_hardware_ready(self):
        return True

    def shutdown(self):
        pass


class FakeReader:
    """Stands in for ImuReader: angles and rates straight from the fake plant."""

    stamp = 0

    def __init__(self, hardware, max_age_s=0.05):
        self.hardware, self.fresh_time = hardware, None
        quat = [1.0, 0.0, 0.0, 0.0]
        self.snapshot = SimpleNamespace(
            imu_by_role={role: quat for role in ("base", "boom", "arm", "bucket")},
            imu_gyro=[[0.0, 0.0, 0.0]] * 3,
            base_imu_gyro=[0.0, 0.0, 0.0],
        )

    def read(self, now=None):
        hw = self.hardware
        with hw.lock:
            hw.advance()
            q, v = hw.q.astype(np.float32), np.r_[RATE * hw.u, 0.0].astype(np.float32)
        FakeReader.stamp += 10000
        self.fresh_time = time.monotonic() if now is None else now
        return q, v, FakeReader.stamp


class FakePad:
    def __init__(self):
        self.created = time.monotonic()
        self.A = self.B = False

    @property
    def LeftBumper(self):  # released at first, held from 1.2 s on
        return time.monotonic() - self.created > 1.2

    def is_connected(self):
        return True

    def stop_monitoring(self):
        pass


class FakeController:
    """ExcavatorController stand-in; in direct-command mode its own thread writes the held setpoint."""

    instances = []

    def __init__(self, hardware, control_config_file=None):
        self.hardware, self.events, self.setpoint = hardware, [], None
        self.stop_event = threading.Event()
        FakeController.instances.append(self)

    def enter_direct_command_mode(self, **kwargs):
        self.events.append("direct")

    def suspend_ik_output(self):
        self.events.append("suspend")

    def give_direct_commands(self, commands):
        self.setpoint = dict(commands)

    def start(self):
        self.events.append("start")
        if "direct" in self.events:
            threading.Thread(target=self.loop, name="controller", daemon=True).start()

    def loop(self):
        while not self.stop_event.wait(0.01):
            self.hardware.send_named_pwm_commands(self.setpoint or {"boom": 0, "arm": 0, "bucket": 0})

    def stop(self):
        self.stop_event.set()


def run_circle(monkeypatch, tmp_path, valve_writes):
    package = types.ModuleType("modules")
    fakes = {
        "board": {
            "resolve_profile": lambda name: {
                "servo_config_file": f"configuration_files/profiles/{name}/servo_config.yaml",
                "control_config_file": f"configuration_files/profiles/{name}/control_config.yaml",
                "pwm_i2c_bus": 7,
                "pwm_i2c_addr": 0x40,
            }
        },
        "bringup": {"wait_for_hardware_ready": lambda hardware: None},
        "excavator_controller": {"ExcavatorController": FakeController},
        "gamepad": {"XboxController": FakePad},
        "hardware_interface": {"HardwareInterface": FakeHardware},
    }
    monkeypatch.setitem(sys.modules, "modules", package)
    for name, attributes in fakes.items():
        module = types.ModuleType(f"modules.{name}")
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, f"modules.{name}", module)
    monkeypatch.setattr(run_robot_circle, "ImuReader", FakeReader)
    monkeypatch.setattr(run_robot_circle, "raw_imu_values", lambda snapshot: [0.0] * 24)
    FakeController.instances.clear()
    log = tmp_path / f"circle_{valve_writes}.csv"
    argv = [
        "run_robot_circle.py", "run",
        "--bundle", str(LOGS / "bundles/bucket_dz0"),
        "--robot_repo", str(ROBOT_REPO),
        "--robot", PROFILE,
        "--controller", "pid_tuned",
        "--pid_gains", str(LOGS / "pid_gains.yaml"),
        "--radius_mm", "5",
        "--valve_writes", valve_writes,
        "--log", str(log),
    ]  # fmt: skip
    monkeypatch.setattr(sys, "argv", argv)
    run_robot_circle.main()
    with log.open(newline="") as stream:
        table = list(csv.DictReader(stream))
    rows = {
        key: np.array(
            [float(row[key] == "True") if row[key] in ("True", "False") else float(row[key]) for row in table]
        )
        for key in table[0]
    }
    report = json.loads(log.with_suffix(".json").read_text())
    return rows, report, FakeController.instances[0]


def test_loop_writes_each_command_in_the_tick_it_is_computed(monkeypatch, tmp_path):
    rows, report, controller = run_circle(monkeypatch, tmp_path, "loop")
    assert report["result"]["completed"] and report["valve_writes"] == "loop"
    assert controller.events == ["suspend", "start"]
    writers = {name for name, _ in controller.hardware.writes}
    assert writers == {"MainThread"}  # only the measurement loop drives the valves
    joints = ("boom", "arm", "bucket")
    requested = np.stack([rows[f"{j}_requested_u"] for j in joints], 1)
    emitted = np.stack([rows[f"{j}_emitted_u"] for j in joints], 1)
    armed = rows["armed"] > 0
    moving = armed[:-1] & armed[1:] & (np.abs(requested[:-1]).sum(1) > 0)
    assert moving.sum() > 100
    # The value read back at the start of a tick is exactly what the previous tick computed and wrote.
    np.testing.assert_allclose(emitted[1:][moving], requested[:-1][moving], atol=1e-6)


def test_thread_mode_keeps_the_direct_command_path(monkeypatch, tmp_path):
    rows, report, controller = run_circle(monkeypatch, tmp_path, "thread")
    assert report["result"]["completed"] and report["valve_writes"] == "thread"
    assert controller.events == ["direct", "start"]
    assert any(name == "controller" for name, _ in controller.hardware.writes)
