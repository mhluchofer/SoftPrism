"""
tendon_motor_bridge.py

Converts generalized coordinates q = [theta, phi] into motor angular
positions, for a direct-pulley winch actuation mechanism (radius
r_pulley, no gearbox, confirmed hardware):

    L_i(q) = L - theta*R_c*cos(gamma_i - phi)     (eq. 18, Sec. II.D)
    theta_motor_i = (L0_i - L_i(q)) / r_pulley

This module has no CAN or hardware dependency: it only computes the
motor angle to send, replacing the equivalent computation previously
done by an external kinematics module (e.g. InverseKinematics), using
the verified PCC tendon-length model instead.
"""

import numpy as np


class TendonMotorBridge:

    def __init__(self, robot, pulley_radius, L0=None):
        """
        Parameters
        ----------
        robot : SoftPrism
            The robot instance providing the actuation model
            (robot.actuation.tendon_lengths(q)).
        pulley_radius : float
            Winch/pulley radius on the motor shaft (m). Confirmed
            hardware: 0.01 m (2 cm diameter), direct pulley, no
            gearbox.
        L0 : array-like of shape (3,), optional
            Tendon length at the motor's zero angular position, per
            tendon. Defaults to [L, L, L] (robot.geometry.length),
            consistent with tendon_lengths(q) at theta=0 (Sec. II.D:
            all tendons equal L when the neck is straight). Override
            if the physical zero calibration differs.
        """
        self.robot = robot
        self.pulley_radius = float(pulley_radius)

        if L0 is None:
            L = robot.geometry.length
            self.L0 = np.array([L, L, L], dtype=float)
        else:
            self.L0 = np.asarray(L0, dtype=float)
            if self.L0.shape != (3,):
                raise ValueError("L0 must have shape (3,)")

    def motor_angles_from_q(self, q):
        """
        Desired motor angles for a target q = [theta, phi]:

            theta_motor_i = (L0_i - L_i(q)) / r_pulley

        Returns
        -------
        numpy.ndarray of shape (3,)
        """
        L = self.robot.actuation.tendon_lengths(q)
        return (self.L0 - L) / self.pulley_radius