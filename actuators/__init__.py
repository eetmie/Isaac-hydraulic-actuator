"""Reusable, robot-agnostic actuators for learned hydraulic forward models.

Nothing in this package knows what an excavator is. Joint names, gamepad
mappings, tool assets and the decision of what to do when a model has not been
trained yet all live in the demo script that drives these classes.

``HydraulicActuatorNet`` needs only numpy and torch and can be imported outside
Isaac Sim; the two integrating actuators need Isaac Lab.
"""

from importlib import import_module
from typing import TYPE_CHECKING

from .hydraulic_actuator import HydraulicActuatorNet

if TYPE_CHECKING:
    from .direct_integration_actuator import DirectIntegrationActuator, DirectIntegrationActuatorCfg
    from .integrating_actuator_base import IntegratingActuatorBase, IntegratingActuatorCfg
    from .velocity_integrated_actuator import VelocityIntegratedActuator, VelocityIntegratedActuatorCfg

_SIM_EXPORTS = {
    "DirectIntegrationActuator": ".direct_integration_actuator",
    "DirectIntegrationActuatorCfg": ".direct_integration_actuator",
    "IntegratingActuatorBase": ".integrating_actuator_base",
    "IntegratingActuatorCfg": ".integrating_actuator_base",
    "VelocityIntegratedActuator": ".velocity_integrated_actuator",
    "VelocityIntegratedActuatorCfg": ".velocity_integrated_actuator",
}


def __getattr__(name: str) -> object:
    """Load simulation classes only when requested, preserving package exports."""
    if name not in _SIM_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_SIM_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "DirectIntegrationActuator",
    "DirectIntegrationActuatorCfg",
    "HydraulicActuatorNet",
    "IntegratingActuatorBase",
    "IntegratingActuatorCfg",
    "VelocityIntegratedActuator",
    "VelocityIntegratedActuatorCfg",
]
