"""Xbox gamepad adapter for the interactive Isaac demo."""

from __future__ import annotations

import weakref

import numpy as np

import carb
import carb.input
import omni.appwindow

DEAD_ZONE = 0.30

# ── gamepad ────────────────────────────────────────────────────────────────

def _deadzone(v: float) -> float:
    if abs(v) < DEAD_ZONE:
        return 0.0
    s = 1.0 if v > 0.0 else -1.0
    return s * (abs(v) - DEAD_ZONE) / (1.0 - DEAD_ZONE)


class XboxController:
    """Minimal gamepad reader for direct valve-command teleop."""

    def __init__(self):
        carb.settings.get_settings().set_bool(
            "/persistent/app/omniverse/gamepadCameraControl", False
        )
        self._appwindow  = omni.appwindow.get_default_app_window()
        self._input      = carb.input.acquire_input_interface()
        self._gamepad    = self._appwindow.get_gamepad(0)
        # 0/1=tilt, 2/3=carriage, 4/5=lift, 6/7=tool (positive/negative).
        self._axes = np.zeros(8, dtype=np.float32)
        self._a_pressed  = False
        self._prev_a     = False
        self._b_pressed  = False
        self._prev_b     = False

        self._sub = self._input.subscribe_to_gamepad_events(
            self._gamepad,
            lambda ev, *a, obj=weakref.proxy(self): obj._on_event(ev),
        )
        name = self._input.get_gamepad_name(self._gamepad)
        print(f"[Gamepad] {'Connected: ' + name if name else 'No gamepad detected'}")

    def close(self) -> None:
        """Release the Carb subscription deterministically and idempotently."""
        subscription = getattr(self, "_sub", None)
        if subscription is None:
            return
        self._sub = None
        try:
            self._input.unsubscribe_to_gamepad_events(self._gamepad, subscription)
        except (AttributeError, RuntimeError):
            pass

    def __del__(self):
        self.close()

    def _on_event(self, ev):
        GI = carb.input.GamepadInput
        v  = ev.value
        m  = {
            GI.LEFT_STICK_UP:    (0, v),
            GI.LEFT_STICK_DOWN:  (1, v),
            GI.LEFT_STICK_LEFT:  (2, v),
            GI.LEFT_STICK_RIGHT: (3, v),
            GI.RIGHT_STICK_UP:   (4, v),
            GI.RIGHT_STICK_DOWN: (5, v),
            GI.RIGHT_STICK_RIGHT: (6, v),
            GI.RIGHT_STICK_LEFT: (7, v),
        }
        if ev.input in m:
            idx, val = m[ev.input]
            self._axes[idx] = val
        elif ev.input == GI.A:
            self._a_pressed = v > 0.5
        elif ev.input == GI.B:
            self._b_pressed = v > 0.5
        return True

    @staticmethod
    def _axis(pos: float, neg: float) -> float:
        raw = float(pos) - float(neg)
        return _deadzone(raw)

    def read(self) -> tuple[float, float, float, float]:
        """Returns (lift_cmd, carriage_vel_norm, tilt_cmd, tool_cmd) each in [-1, 1]."""
        tilt     = -self._axis(self._axes[0], self._axes[1])  # left  Y
        carriage = -self._axis(self._axes[3], self._axes[2])  # left  X
        lift     =  self._axis(self._axes[4], self._axes[5])  # right Y
        tool     = -self._axis(self._axes[6], self._axes[7])  # right X
        return lift, carriage, tilt, tool

    def mode_toggle_requested(self) -> bool:
        cur    = self._a_pressed
        rising = cur and not self._prev_a
        self._prev_a = cur
        return rising

    def reset_requested(self) -> bool:
        cur    = self._b_pressed
        rising = cur and not self._prev_b
        self._prev_b = cur
        return rising


__all__ = ["XboxController"]
