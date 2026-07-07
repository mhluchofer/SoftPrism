# euler_lagrange.py
# This file defines the EulerLagrangeModel class, which implements the dynamics of the soft robotic prism using the Euler-Lagrange formulation.
# The class computes the mass matrix, Coriolis matrix, gravity vector, and elasticity vector based on the robot's parameters and state variables.

from .euler_lagrange_model_functions import (M_func_api, C_func_api, G_func_api, K_func_api)
import numpy as np

# Threshold below which M_func_api(theta, ...) suffers catastrophic
# cancellation (it contains a division by theta**5). Calibrated so that,
# above this threshold, M_func_api matches the analytic theta -> 0 limit
# to within ~1e-6 relative error; below it, the raw formula degrades
# rapidly (observed relative error > 1e-2 by theta ~ 5e-4, and
# unphysical/negative values by theta ~ 1e-4).
THETA_EPS_DYNAMICS = 1e-2

# Regularization factor applied to the (1,1) [phi,phi] entry of M(q)
# when |theta| < THETA_EPS_DYNAMICS. M_22(theta) -> 0 as theta -> 0: this
# is not a numerical artifact but a genuine kinematic singularity of the
# PCC parameterization (phi is undefined at theta = 0, so it carries no
# generalized inertia in that limit). The regularization keeps M(q)
# invertible in a neighborhood of theta = 0 at the cost of treating
# phi_ddot as approximate (Tikhonov-style regularization), scaled to the
# robot's own M_11 so it adapts to different physical parameters instead
# of relying on a fixed absolute constant.
M22_REGULARIZATION_FACTOR = 1e-4


class EulerLagrangeModel:

    def __init__(self, robot):

        self.robot = robot

    def mass_matrix(self, q):
        """
        Mass matrix M(q) (eq. 12 / Appendix A). For |theta| below
        THETA_EPS_DYNAMICS, the closed-form M_func_api expression is
        replaced by its analytic theta -> 0 limit, to avoid the
        division-by-theta**5 catastrophic cancellation of the raw
        formula.

        Note: M(q) is symmetric positive definite for theta != 0, but
        M_22 -> 0 as theta -> 0 (kinematic singularity: phi is
        undefined at zero bending). Near that point, M_22 is instead
        set to a small Tikhonov regularization term proportional to
        M_11, to keep M(q) invertible; phi_ddot obtained from a
        regularized M(q) should be treated as approximate in that
        regime, not as an exact dynamic prediction.
        """
        theta, phi = q
        p = self.robot._dynamic_parameters()

        if abs(theta) < THETA_EPS_DYNAMICS:
            return self._mass_matrix_small_theta_limit(p)

        return np.asarray(M_func_api(
            theta,
            p["L"],
            p["a"],
            p["h"],
            p["m_backbone"],
            p["m_prism"],
            p["r_hole"]
        ), dtype=float)

    @staticmethod
    def _mass_matrix_small_theta_limit(p):
        """
        Analytic theta -> 0 limit of M(q), derived from the closed-form
        blocks in Appendix A:

            M_flex,11  -> L^2 m_backbone / 20
            M_trans,11 -> 979 L^2 m_prism / 5184   (5 prisms at s_k = kL/6)
            M_rot,11   -> 5 I_xx                    (theta-independent)

        M_22 -> 0 exactly in this limit (M_flex,22, M_trans,22 and
        M_rot,22 all vanish as theta -> 0); it is regularized here
        rather than set to zero, to keep M(q) invertible.
        """
        L = p["L"]
        a = p["a"]
        h = p["h"]
        m_backbone = p["m_backbone"]
        m_prism = p["m_prism"]
        r_hole = p["r_hole"]

        denom = np.sqrt(3) * a**2 - 4 * np.pi * r_hole**2
        Ixx = m_prism * (
            np.sqrt(3) * a**4
            + 2 * h**2 * denom
            - 24 * np.pi * r_hole**4
        ) / (24 * denom)

        M11 = (L**2 * m_backbone) / 20 + (979 * L**2 * m_prism) / 5184 + 5 * Ixx
        M22 = M22_REGULARIZATION_FACTOR * M11

        return np.array([
            [M11, 0.0],
            [0.0, M22]
        ])

    def coriolis(self, q, qdot):
        """
        Coriolis/centrifugal matrix C(q,qdot), obtained from the
        standard Christoffel-symbol construction on M(q).

        For |theta| < THETA_EPS_DYNAMICS, C is set to the zero matrix.
        This is not an independent approximation: in that same regime,
        mass_matrix() already treats M(q) as locally constant (see
        _mass_matrix_small_theta_limit), and C(q,qdot) is by
        construction the term arising from the q-dependence of M(q).
        Consistency therefore requires C = 0 wherever M is treated as
        constant; using the raw C_func_api there would reintroduce the
        same theta**6 catastrophic cancellation already avoided in
        mass_matrix(), and would in any case be physically inconsistent
        with a constant M(q) in that neighborhood.
        """
        theta, phi = q
        theta_dot, phi_dot = qdot

        if abs(theta) < THETA_EPS_DYNAMICS:
            return np.zeros((2, 2))

        p = self.robot._dynamic_parameters()

        return np.asarray(C_func_api(
            theta,
            theta_dot,
            phi_dot,
            p["L"],
            p["m_backbone"],
            p["m_prism"],
            p["a"],
            p["h"],
            p["r_hole"]
        ), dtype=float)

    def gravity(self, q):
        """
        Gravitational generalized-force vector G(q) = dV_grav/dq.

        For |theta| < THETA_EPS_DYNAMICS, G is set to the zero vector.
        Unlike M(q) and C(q,qdot), G_func_api does not suffer
        catastrophic cancellation in this range (its raw values scale
        linearly and stably down to theta ~ 1e-4); the zero-vector
        limit is simply the exact analytic value as theta -> 0
        (a straight backbone has no preferred bending direction under
        gravity, so the generalized force conjugate to theta vanishes
        there), used mainly to avoid the literal division-by-zero in
        G_func_api at theta = 0 exactly.
        """
        theta, phi = q

        if abs(theta) < THETA_EPS_DYNAMICS:
            return np.zeros((2, 1))

        p = self.robot._dynamic_parameters()

        return G_func_api(
            theta,
            p["L"],
            p["m_backbone"],
            p["m_prism"],
            p["g"]
        )

    def elasticity(self, q):
        theta, phi = q

        p = self.robot._dynamic_parameters()

        return K_func_api(
            theta,
            p["E_mod"],
            p["I_geom"],
            p["L"]
        )

    def inverse_dynamics(self, q, qdot, qddot):

        M = self.mass_matrix(q)
        C = self.coriolis(q, qdot)
        G = self.gravity(q)
        K = self.elasticity(q)

        qdot = np.asarray(qdot).reshape(2, 1)
        qddot = np.asarray(qddot).reshape(2, 1)

        tau = (M @ qddot + C @ qdot + G + K)

        return tau

    def forward_dynamics(
        self,
        q,
        qdot,
        tensions
    ):
        """
        Forward dynamics of the neck: given the current state (q, qdot)
        and the applied tendon tensions, returns the generalized
        acceleration

            qddot = M(q)^{-1} [ J_L(q)^T T - C(q,qdot) qdot - G(q) - K(q) ],

        i.e. the actuated equation of motion solved for qddot. J_L(q)^T T
        is obtained from the actuation module, which owns the tendon
        routing geometry and Jacobian; this method only assembles and
        solves the resulting linear system, without duplicating that
        geometry here.

        Parameters
        ----------
        q : array-like of shape (2,)
            Current generalized coordinates [theta, phi].
        qdot : array-like of shape (2,)
            Current generalized velocities [theta_dot, phi_dot].
        tensions : array-like of shape (3,)
            Applied tendon tensions [T1, T2, T3].

        Returns
        -------
        numpy.ndarray of shape (2,)
            Generalized acceleration [theta_ddot, phi_ddot]. Near the
            kinematic singularity theta = 0, phi_ddot is obtained from
            a regularized M(q) (see mass_matrix()) and should be
            treated as approximate rather than exact.
        """
        M = self.mass_matrix(q)
        C = self.coriolis(q, qdot)
        G = np.asarray(self.gravity(q)).reshape(2, 1)
        K = np.asarray(self.elasticity(q)).reshape(2, 1)

        qdot_vec = np.asarray(qdot, dtype=float).reshape(2, 1)

        tau = self.robot.actuation.tensions_to_generalized(q, tensions)
        tau = np.asarray(tau, dtype=float).reshape(2, 1)

        rhs = tau - C @ qdot_vec - G - K

        # M(q) is symmetric positive definite for theta != 0; near the
        # theta = 0 kinematic singularity it is regularized by
        # mass_matrix() to remain invertible (see note above).
        qddot = np.linalg.solve(M, rhs)

        return qddot.flatten()