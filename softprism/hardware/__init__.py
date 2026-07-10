# TendonMotorBridge has no hardware/CAN dependency and is safe to
# import eagerly.
from .tendon_motor_bridge import TendonMotorBridge
 
# On the real platform, import them explicitly when needed:
#
#     from softprism.hardware.motor import Motor
#     from softprism.hardware.system_motors import SystemMotors
 