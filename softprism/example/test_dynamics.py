"""
test_dynamics.py

Unit tests for softprism.dynamics.Dynamics / EulerLagrangeModel, verifying
that the implementation reproduces the closed-form dynamic model derived
in Section II.C of the accompanying paper and its Appendix A:

    M(q)  -- eq. (12) / Appendix A
    C(q,qdot)   -- standard Christoffel-symbol construction
    G(q)  -- eq. (11), gravitational term
    K(q)  -- eq. (11), elastic term
    inverse_dynamics -- eq. (14): tau = M qddot + C qdot + G + K
    forward_dynamics -- eq. (23) solved for qddot, with tendon tensions

Tests are organized in five groups:
    1. Structural properties of M(q) (symmetry, positive-definiteness).
    2. The theta -> 0 kinematic singularity and its regularization.
    3. Cross-validation between forward and inverse dynamics.
    4. Physical sanity checks (equilibrium, energy-consistent signs).
    5. Numerical robustness (finite differences, time integration).
"""

import numpy as np
import pytest
from scipy.optimize import lsq_linear
from scipy.integrate import solve_ivp

from softprism import Geometry, Material, ActuationConfig, SoftPrism
from softprism.dynamics.euler_lagrange import THETA_EPS_DYNAMICS


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture
def robot():
    geometry = Geometry(
        length=0.105,
        n_prisms=5,
        prism_side=0.07,
        prism_thickness=0.014,
        backbone_radius=0.01,
    )
    material = Material(
        backbone_mass=0.03,
        prism_mass=0.01,
        young_modulus=2e6,
    )
    actuationconfig = ActuationConfig(
        n_tendons=3,
        tendon_distance=0.005,
    )
    return SoftPrism(geometry, material, actuationconfig)


# Representative non-singular configurations, spanning positive and
# negative phi and a range of bending magnitudes safely above
# THETA_EPS_DYNAMICS.
CONFIGS = [
    (0.10, 0.00),
    (0.30, 0.50),
    (0.60, -1.20),
    (0.95, 2.80),
]


# ========================================================================
# 1. Structural properties of M(q)
# ========================================================================
class TestMassMatrixStructure:

    @pytest.mark.parametrize("theta,phi", CONFIGS)
    def test_symmetric(self, robot, theta, phi):
        M = np.asarray(robot.dynamics.mass_matrix([theta, phi]))
        assert np.allclose(M, M.T, atol=1e-12)

    @pytest.mark.parametrize("theta,phi", CONFIGS)
    def test_diagonal(self, robot, theta, phi):
        """The (1,2) off-diagonal entry vanishes identically (Appendix A)."""
        M = np.asarray(robot.dynamics.mass_matrix([theta, phi]))
        assert np.isclose(M[0, 1], 0.0, atol=1e-12)
        assert np.isclose(M[1, 0], 0.0, atol=1e-12)

    @pytest.mark.parametrize("theta,phi", CONFIGS)
    def test_positive_definite_away_from_singularity(self, robot, theta, phi):
        """M(q) must be positive definite for theta != 0 (Appendix A)."""
        M = np.asarray(robot.dynamics.mass_matrix([theta, phi]))
        eigvals = np.linalg.eigvalsh(M)
        assert np.all(eigvals > 0)

    def test_independent_of_phi(self, robot):
        """
        M(q) depends only on theta (Sec. II.C / Appendix A): evaluating
        it at different phi with the same theta must give identical
        results.
        """
        theta = 0.4
        M_phi0 = np.asarray(robot.dynamics.mass_matrix([theta, 0.0]))
        M_phi1 = np.asarray(robot.dynamics.mass_matrix([theta, 1.7]))
        assert np.allclose(M_phi0, M_phi1, atol=1e-14)


# ========================================================================
# 2. The theta -> 0 kinematic singularity and its regularization
# ========================================================================
class TestSingularityRegularization:

    def test_no_crash_at_exact_zero(self, robot):
        """mass_matrix, coriolis, gravity must all be evaluable at theta=0."""
        M = robot.dynamics.mass_matrix([0.0, 0.3])
        C = robot.dynamics.coriolis([0.0, 0.3], [0.1, 0.1])
        G = robot.dynamics.gravity([0.0, 0.3])
        assert np.all(np.isfinite(np.asarray(M)))
        assert np.all(np.isfinite(np.asarray(C)))
        assert np.all(np.isfinite(np.asarray(G)))

    def test_M22_degenerates_at_theta_zero(self, robot):
        """
        M_22 must vanish relative to M_11 at theta=0 (regularized to a
        small but nonzero value, per the Tikhonov regularization),
        reflecting the genuine kinematic degeneracy of phi at zero
        bending (Appendix A, Sec. "Total Inertia Matrix").
        """
        M = np.asarray(robot.dynamics.mass_matrix([0.0, 0.0]))
        assert M[1, 1] < 1e-3 * M[0, 0]
        assert M[1, 1] > 0.0  # regularized, not exactly singular

    def test_condition_number_bounded_at_theta_zero(self, robot):
        """
        Unlike the raw (unregularized) closed form -- which diverges as
        theta -> 0 -- the regularized M(q) must have a finite,
        bounded condition number at theta = 0.
        """
        M = np.asarray(robot.dynamics.mass_matrix([0.0, 0.0]))
        cond = np.linalg.cond(M)
        assert np.isfinite(cond)
        assert cond < 1e6

    def test_coriolis_vanishes_below_threshold(self, robot):
        """
        Consistent with treating M(q) as locally constant for
        |theta| < THETA_EPS_DYNAMICS, C(q,qdot) must be exactly zero
        in that same regime.
        """
        for theta in [0.0, THETA_EPS_DYNAMICS / 2, -THETA_EPS_DYNAMICS / 3]:
            C = np.asarray(robot.dynamics.coriolis([theta, 0.2], [0.3, -0.4]))
            assert np.allclose(C, np.zeros((2, 2)), atol=1e-15)

    def test_gravity_vanishes_below_threshold(self, robot):
        """
        A straight backbone has no preferred bending direction under
        gravity, so G(q) must be exactly zero for |theta| < THETA_EPS_DYNAMICS.
        """
        for theta in [0.0, THETA_EPS_DYNAMICS / 2]:
            G = np.asarray(robot.dynamics.gravity([theta, 0.4]))
            assert np.allclose(G, np.zeros((2, 1)), atol=1e-15)

    def test_mass_matrix_continuous_across_threshold(self, robot):
        """
        M_11 must not show a large jump when crossing THETA_EPS_DYNAMICS,
        since the analytic limit and the raw closed form agree closely
        near the switching boundary (verified analytically beforehand).
        """
        M_below = np.asarray(
            robot.dynamics.mass_matrix([THETA_EPS_DYNAMICS * 0.99, 0.0])
        )
        M_above = np.asarray(
            robot.dynamics.mass_matrix([THETA_EPS_DYNAMICS * 1.01, 0.0])
        )
        rel_diff = abs(M_below[0, 0] - M_above[0, 0]) / M_above[0, 0]
        assert rel_diff < 1e-3

    def test_negative_theta_handled_symmetrically(self, robot):
        """The |theta| < THETA_EPS_DYNAMICS check must also cover the
        negative side, mirroring the kinematics module's behavior."""
        M_pos = np.asarray(robot.dynamics.mass_matrix([1e-5, 0.3]))
        M_neg = np.asarray(robot.dynamics.mass_matrix([-1e-5, 0.3]))
        assert np.allclose(M_pos, M_neg, atol=1e-14)


# ========================================================================
# 3. Cross-validation between forward and inverse dynamics
# ========================================================================
class TestForwardInverseConsistency:

    @pytest.mark.parametrize("theta,phi", CONFIGS)
    def test_round_trip_recovers_qddot(self, robot, theta, phi):
        """
        Given a known (q, qdot, qddot), inverse_dynamics must produce a
        generalized-force vector that, when mapped to non-negative
        tendon tensions and passed back through forward_dynamics,
        recovers the original qddot to high precision.
        """
        q = [theta, phi]
        qdot = [0.1, -0.15]
        qddot_true = [0.05, -0.08]

        tau_required = robot.dynamics.inverse_dynamics(q, qdot, qddot_true)

        J = robot.actuation.tendon_jacobian(q)
        res = lsq_linear(J.T, np.asarray(tau_required).flatten(),
                          bounds=(0, np.inf))
        assert res.success
        T = res.x

        qddot_recovered = robot.dynamics.forward_dynamics(q, qdot, T)
        assert np.allclose(qddot_true, qddot_recovered, atol=1e-8)

    def test_zero_tension_case(self, robot):
        """
        With zero tendon tension, forward_dynamics must reduce to the
        unactuated structural response: qddot = M^{-1}(-C qdot - G - K).
        """
        q = [0.4, 0.2]
        qdot = [0.1, 0.0]
        T = [0.0, 0.0, 0.0]

        qddot = robot.dynamics.forward_dynamics(q, qdot, T)

        M = np.asarray(robot.dynamics.mass_matrix(q))
        C = np.asarray(robot.dynamics.coriolis(q, qdot))
        G = np.asarray(robot.dynamics.gravity(q)).flatten()
        K = np.asarray(robot.dynamics.elasticity(q)).flatten()
        qdot_arr = np.asarray(qdot)

        expected = np.linalg.solve(M, -C @ qdot_arr - G - K)
        assert np.allclose(qddot, expected, atol=1e-10)

    def test_equal_tensions_produce_zero_generalized_force(self, robot):
        """
        Equal tension on all three tendons must produce zero generalized
        actuation force (eq. 22 discussion), so forward_dynamics with
        equal tensions must match the zero-tension case exactly.
        """
        q = [0.5, -0.3]
        qdot = [0.0, 0.0]

        qddot_zero = robot.dynamics.forward_dynamics(q, qdot, [0.0, 0.0, 0.0])
        qddot_equal = robot.dynamics.forward_dynamics(q, qdot, [2.0, 2.0, 2.0])
        assert np.allclose(qddot_zero, qddot_equal, atol=1e-10)


# ========================================================================
# 4. Physical sanity checks
# ========================================================================
class TestPhysicalSanity:

    def test_static_equilibrium_zero_acceleration(self, robot):
        """
        At a configuration/tension pair satisfying G(q)+K(q) = J_L(q)^T T
        with qdot = 0, forward_dynamics must return (numerically) zero
        acceleration.
        """
        q = [0.3, 0.1]
        G = np.asarray(robot.dynamics.gravity(q)).flatten()
        K = np.asarray(robot.dynamics.elasticity(q)).flatten()
        Q_required = G + K  # = J_L^T T at equilibrium (qddot=0, qdot=0)

        J = robot.actuation.tendon_jacobian(q)
        res = lsq_linear(J.T, Q_required, bounds=(0, np.inf))
        assert res.success
        T = res.x

        qddot = robot.dynamics.forward_dynamics(q, [0.0, 0.0], T)
        assert np.allclose(qddot, [0.0, 0.0], atol=1e-8)

    def test_elasticity_restoring_sign(self, robot):
        """
        The elastic term K(q) must act as a restoring force: with no
        tension and no velocity, positive theta must yield a negative
        theta-acceleration contribution from elasticity alone (checked
        by comparing forward_dynamics at small vs. larger theta, holding
        gravity's contribution fixed by using theta small enough that
        G(q) is negligible relative to K(q)).
        """
        q = [0.05, 0.0]
        qdot = [0.0, 0.0]
        T = [0.0, 0.0, 0.0]
        qddot = robot.dynamics.forward_dynamics(q, qdot, T)
        # Elastic restoring torque pulls theta back toward 0.
        assert qddot[0] < 0.0


# ========================================================================
# 5. Numerical robustness
# ========================================================================
class TestNumericalRobustness:

    def test_time_integration_through_singularity(self, robot):
        """
        Integrating forward_dynamics over a trajectory that passes
        through theta = 0 must not produce NaN/Inf and must complete
        successfully.
        """
        def rhs(t, x):
            theta, phi, theta_dot, phi_dot = x
            qddot = robot.dynamics.forward_dynamics(
                [theta, phi], [theta_dot, phi_dot], [0.05, 0.05, 0.05]
            )
            return [theta_dot, phi_dot, qddot[0], qddot[1]]

        sol = solve_ivp(rhs, [0, 1], [0.05, 0.0, -0.2, 0.0], max_step=1e-3)
        assert sol.success
        assert np.all(np.isfinite(sol.y))

    @pytest.mark.parametrize("theta,phi", CONFIGS)
    def test_inverse_dynamics_matches_manual_assembly(self, robot, theta, phi):
        """
        inverse_dynamics(q,qdot,qddot) must equal the manual assembly
        M @ qddot + C @ qdot + G + K, independently recomputed here from
        the individual M/C/G/K accessors.
        """
        q = [theta, phi]
        qdot = [0.2, -0.1]
        qddot = [0.03, 0.07]

        tau = np.asarray(robot.dynamics.inverse_dynamics(q, qdot, qddot)).flatten()

        M = np.asarray(robot.dynamics.mass_matrix(q))
        C = np.asarray(robot.dynamics.coriolis(q, qdot))
        G = np.asarray(robot.dynamics.gravity(q)).flatten()
        K = np.asarray(robot.dynamics.elasticity(q)).flatten()

        tau_manual = M @ np.asarray(qddot) + C @ np.asarray(qdot) + G + K
        assert np.allclose(tau, tau_manual, atol=1e-12)

    def test_forward_dynamics_shape(self, robot):
        qddot = robot.dynamics.forward_dynamics([0.4, 0.1], [0.0, 0.0],
                                                 [0.1, 0.2, 0.3])
        assert np.asarray(qddot).shape == (2,)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
