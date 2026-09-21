"""Kit gamepad input for end-effector velocity playback."""

from __future__ import annotations

import math
import weakref

import carb
import carb.input
import numpy as np
from isaaclab.app.settings_manager import get_settings_manager


def _deadzone(value: float, threshold: float = 0.2) -> float:
    if abs(value) <= threshold:
        return 0.0
    return math.copysign((abs(value) - threshold) / (1 - threshold), value)


class GamepadVelocity:
    """Xbox-style X/Z/pitch velocity source with a rising-edge reset button."""

    def __init__(self, speed: float, pitch_rate: float):
        get_settings_manager().set_bool("/persistent/app/omniverse/gamepadCameraControl", False)
        self.speed, self.pitch_rate = speed, pitch_rate
        self.axes = np.zeros(6, dtype=np.float32)
        self.reset_pressed = self.previous_reset = False
        self.interface = self.gamepad = self.subscription = None
        try:
            import omni.appwindow

            window = omni.appwindow.get_default_app_window()
        except ImportError:
            window = None
        if window is None:
            print("[Gamepad] No Kit window; commands remain zero")
            return
        self.interface = carb.input.acquire_input_interface()
        self.gamepad = window.get_gamepad(0)
        self.subscription = self.interface.subscribe_to_gamepad_events(
            self.gamepad, lambda event, *unused, obj=weakref.proxy(self): obj._event(event)
        )
        name = self.interface.get_gamepad_name(self.gamepad)
        print(f"[Gamepad] {name or 'No controller detected'}")
        print("[Gamepad] left stick = tip X/Z, right stick X = bucket pitch, B = reset")

    def close(self):
        if self.interface is not None and self.subscription is not None:
            self.interface.unsubscribe_to_gamepad_events(self.gamepad, self.subscription)
            self.subscription = None

    def _event(self, event):
        key = carb.input.GamepadInput
        mapping = {
            key.LEFT_STICK_RIGHT: (0, event.value),
            key.LEFT_STICK_LEFT: (1, event.value),
            key.LEFT_STICK_UP: (2, event.value),
            key.LEFT_STICK_DOWN: (3, event.value),
            key.RIGHT_STICK_RIGHT: (4, event.value),
            key.RIGHT_STICK_LEFT: (5, event.value),
        }
        if event.input in mapping:
            index, value = mapping[event.input]
            self.axes[index] = value
        elif event.input == key.B:
            self.reset_pressed = event.value > 0.5
        return True

    @staticmethod
    def _axis(positive, negative):
        return _deadzone(float(positive) - float(negative))

    def command(self):
        return np.array(
            [
                self.speed * self._axis(self.axes[0], self.axes[1]),
                self.speed * self._axis(self.axes[2], self.axes[3]),
                self.pitch_rate * self._axis(self.axes[4], self.axes[5]),
            ],
            dtype=np.float32,
        )

    def wants_reset(self):
        rising = self.reset_pressed and not self.previous_reset
        self.previous_reset = self.reset_pressed
        return rising
