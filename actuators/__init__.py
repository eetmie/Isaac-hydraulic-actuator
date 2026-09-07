"""Reusable, robot-agnostic actuators for learned hydraulic forward models.

Nothing in this package knows what an excavator is. Joint names, gamepad
mappings, tool assets and the decision of what to do when a model has not been
trained yet all live in the demo script that drives these classes.

``HydraulicActuatorNet`` needs only numpy and torch and can be imported outside
Isaac Sim; the two integrating actuators need Isaac Lab.
"""

from .direct_integration_actuator import DirectIntegrationActuator, DirectIntegrationActuatorCfg
from .hydraulic_actuator import HydraulicActuatorNet
from .integrating_actuator_base import IntegratingActuatorBase, IntegratingActuatorCfg
from .velocity_integrated_actuator import VelocityIntegratedActuator, VelocityIntegratedActuatorCfg

__all__ = [
    "DirectIntegrationActuator",
    "DirectIntegrationActuatorCfg",
    "HydraulicActuatorNet",
    "IntegratingActuatorBase",
    "IntegratingActuatorCfg",
    "VelocityIntegratedActuator",
    "VelocityIntegratedActuatorCfg",
]
