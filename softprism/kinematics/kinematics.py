"""
kinematics.py

Implements the Piecewise Constant Curvature (PCC) kinematic model of the
soft robotic neck, as derived in Section II.B of the accompanying paper:

    - Backbone parameterization:        r(s, q)      -> eq. (2)
    - End-effector orientation:         R(theta, phi) -> eq. (3)
    - Homogeneous transformation:       T(q)          -> eq. (4)
    - Linear-velocity Jacobian:         J_v(s)        -> eq. (5)
    - Angular-velocity Jacobian:        J_omega(q)    -> eq. (6)

All closed-form expressions reproduce the symbolic derivations of the
research notebook (modelo_dinamico_v2.ipynb). A small-theta safeguard is
applied to r(s,q) and J_v(s), which present a removable 0/0 singularity
at theta = 0; the analytic Taylor-series limit is used instead of the
raw formula whenever |theta| falls below THETA_EPS, consistent with the
limit derivation given in the paper.
"""

import numpy as np

# Threshold below which the closed-form expressions for r(s,q) and Jv(s)
# are replaced by their analytic theta -> 0 limit, to avoid the 0/0
# floating-point indeterminacy of the raw PCC formula.
THETA_EPS = 1e-6


class Kinematics:

    def __init__(self, robot):
        self.robot = robot

    # ------------------------------------------------------------------
    # Backbone parameterization  r(s, q)  -- eq. (2)
    # ------------------------------------------------------------------
    def backbone_position(self, s, theta, phi):
        """
        Position of the backbone point at arc-length s, under the PCC
        hypothesis (eq. 2). Returns a 3-element numpy array [x, y, z].

        Uses the analytic theta -> 0 limit for |theta| < THETA_EPS,
        recovering the undeformed configuration r(s,q) -> [0, 0, s].
        """
        if abs(theta) < THETA_EPS:
            return np.array([0.0, 0.0, s])

        c = theta * s / self.robot.geometry.length
        L_over_theta = self.robot.geometry.length / theta

        x = L_over_theta * (1 - np.cos(c)) * np.cos(phi)
        y = L_over_theta * (1 - np.cos(c)) * np.sin(phi)
        z = L_over_theta * np.sin(c)

        return np.array([x, y, z])

    def tip_position(self, theta, phi):
        """Tip position p_tip(q) = r(L, q)."""
        return self.backbone_position(self.robot.geometry.length, theta, phi)

    # ------------------------------------------------------------------
    # End-effector orientation  R(theta, phi)  -- eq. (3)
    # ------------------------------------------------------------------
    def rotation_matrix(self, theta, phi):
        """
        ZYZ Euler-angle rotation matrix R(theta,phi) = Rz(phi) Ry(theta) Rz(-phi),
        describing the orientation of the mobile frame oxyz relative to the
        fixed frame OXYZ (eq. 3). No small-theta safeguard is needed here,
        since R(theta,phi) is well defined (identity) at theta = 0.
        """
        Rz_phi = self._rot_z(phi)
        Ry_theta = self._rot_y(theta)
        Rz_minus_phi = self._rot_z(-phi)

        return Rz_phi @ Ry_theta @ Rz_minus_phi

    @staticmethod
    def _rot_z(angle):
        c, s = np.cos(angle), np.sin(angle)
        return np.array([
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0]
        ])

    @staticmethod
    def _rot_y(angle):
        c, s = np.cos(angle), np.sin(angle)
        return np.array([
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c]
        ])

    # ------------------------------------------------------------------
    # Homogeneous transformation  T(q)  -- eq. (4)
    # ------------------------------------------------------------------
    def homogeneous_transform(self, theta, phi):
        """
        Full end-effector pose T(q) combining tip position and orientation
        (eq. 4). Returns a 4x4 numpy array.
        """
        R = self.rotation_matrix(theta, phi)
        p = self.tip_position(theta, phi)

        T = np.eye(4)
        T[0:3, 0:3] = R
        T[0:3, 3] = p
        return T

    # ------------------------------------------------------------------
    # Linear-velocity Jacobian  J_v(s)  -- eq. (5)
    # ------------------------------------------------------------------
    def linear_velocity_jacobian(self, s, theta, phi):
        """
        J_v(s) = d r(s,q) / dq, in R^{3x2} (eq. 5).

        Uses the analytic theta -> 0 limit for |theta| < THETA_EPS:
            J_v(s) -> [[s^2 cos(phi)/(2L), 0],
                       [s^2 sin(phi)/(2L), 0],
                       [0,                 0]]
        """
        L = self.robot.geometry.length

        if abs(theta) < THETA_EPS:
            dxdtheta = (s**2) * np.cos(phi) / (2.0 * L)
            dydtheta = (s**2) * np.sin(phi) / (2.0 * L)
            dzdtheta = 0.0
            dxdphi = 0.0
            dydphi = 0.0
            dzdphi = 0.0
        else:
            c = theta * s / L
            one_minus_cos = 1 - np.cos(c)
            sin_c = np.sin(c)
            cos_c = np.cos(c)

            # d/dtheta of (L/theta)(1 - cos(theta s / L))
            dfdtheta = -(L / theta**2) * one_minus_cos + (s / theta) * sin_c
            # d/dtheta of (L/theta) sin(theta s / L)
            dzdtheta = -(L / theta**2) * sin_c + (s / theta) * cos_c

            dxdtheta = dfdtheta * np.cos(phi)
            dydtheta = dfdtheta * np.sin(phi)

            f = (L / theta) * one_minus_cos
            dxdphi = -f * np.sin(phi)
            dydphi = f * np.cos(phi)
            dzdphi = 0.0

        return np.array([
            [dxdtheta, dxdphi],
            [dydtheta, dydphi],
            [dzdtheta, dzdphi]
        ])

    # ------------------------------------------------------------------
    # Angular-velocity Jacobian  J_omega(theta, phi)  -- eq. (6)
    # ------------------------------------------------------------------
    def angular_velocity_jacobian(self, theta, phi):
        """
        J_omega(theta,phi), obtained from the kinematic identity
        Rdot R^T = [omega x] applied to R(theta,phi) (eq. 6). Well defined
        for all (theta,phi), including theta = 0 (no safeguard needed).
        """
        return np.array([
            [-np.sin(phi),               -np.sin(theta) * np.cos(phi)],
            [ np.cos(phi),               -np.sin(theta) * np.sin(phi)],
            [0.0,                         1 - np.cos(theta)]
        ])

    def angular_velocity(self, theta, phi, theta_dot, phi_dot):
        """omega = J_omega(theta,phi) @ qdot, as a 3-element numpy array."""
        Jw = self.angular_velocity_jacobian(theta, phi)
        qdot = np.array([theta_dot, phi_dot])
        return Jw @ qdot

    # ------------------------------------------------------------------
    # Legacy interface (kept for backward compatibility with existing
    # callers; NOT part of the PCC model derived in the paper -- see note)
    # ------------------------------------------------------------------
    def forward(self, theta, phi):
        """
        Returns the tip position (x, y, z) using the full PCC model
        (eq. 2), replacing the previous small-angle approximation
        (r = L*theta/2, z = L) that did not match the paper's model.
        """
        return tuple(self.tip_position(theta, phi))


    def inverse_exact(self, target, theta0=None, phi0=None,
                       tol=1e-10, max_iter=50):
        """
        Exact inverse kinematics, obtained by numerically solving
        tip_position(theta, phi) = target for (theta, phi), via Newton
        iteration on the full PCC forward map (eq. 2 at s = L), using
        the already-derived linear-velocity Jacobian J_v(L) (eq. 5) in
        place of a finite-difference Jacobian.

        Parameters
        ----------
        target : array-like of shape (3,)
            Desired tip position (x, y, z).
        theta0, phi0 : float, optional
            Initial guess. If omitted, inverse_approx(x, y) is used to
            obtain a small-bending initial estimate for theta, and phi
            is taken directly from atan2(y, x).
        tol : float
            Convergence tolerance on the position residual (meters).
        max_iter : int
            Maximum number of Newton iterations.

        Returns
        -------
        theta, phi : float
            Solution.
        n_iter : int
            Number of iterations performed.
        residual_norm : float
            Norm of the final position residual, for convergence
            diagnostics.
        """
        target = np.asarray(target, dtype=float)
        x, y = target[0], target[1]

        if theta0 is None or phi0 is None:
            theta0, phi0 = self.inverse_approx(x, y)

        theta, phi = float(theta0), float(phi0)

        for n_iter in range(max_iter):
            p = self.tip_position(theta, phi)
            residual = target - p
            residual_norm = np.linalg.norm(residual)

            if residual_norm < tol:
                return theta, phi, n_iter, residual_norm

            Jv = self.linear_velocity_jacobian(
                self.robot.geometry.length, theta, phi
            )
            # Least-squares step: Jv is 3x2 (overdetermined system).
            dq, *_ = np.linalg.lstsq(Jv, residual, rcond=None)
            theta += dq[0]
            phi += dq[1]

        p = self.tip_position(theta, phi)
        residual_norm = np.linalg.norm(target - p)
        return theta, phi, max_iter, residual_norm
    

    def inverse_approx(self, x, y):
        """
        First-order (small-bending) approximation of the inverse
        kinematics, obtained by expanding 1 - cos(theta) ~ theta^2/2
        in the exact tip-position map (eq. 2 at s = L):

            theta ~= 2 sqrt(x^2 + y^2) / L,   phi = atan2(y, x).
        """
        L = self.robot.geometry.length
        r_xy = np.sqrt(x**2 + y**2)
        theta = float(2.0 * r_xy / L)
        phi = float(np.arctan2(y, x))
        return theta, phi