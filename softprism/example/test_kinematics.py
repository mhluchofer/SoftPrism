"""
test_kinematics.py

Unit tests for softprism.kinematics.Kinematics, verifying that the
implementation reproduces the closed-form PCC kinematic model derived in
Section II.B of the accompanying paper:

    r(s, q)        -- eq. (2)
    R(theta, phi)  -- eq. (3)
    T(q)           -- eq. (4)
    J_v(s)         -- eq. (5)
    J_omega(q)     -- eq. (6)

Tests are organized in four groups:
    1. Analytical / geometric consistency checks (closed-form identities).
    2. Cross-validation against independent numerical differentiation.
    3. Structural properties (orthogonality, symmetry, singular limits).
    4. Backward-compatibility of the legacy forward()/inverse() interface.
"""

import numpy as np
import pytest

from softprism import Geometry, Material, ActuationConfig, SoftPrism
from softprism.kinematics.kinematics import THETA_EPS


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


@pytest.fixture
def kin(robot):
    return robot.kinematics


# A representative set of non-degenerate configurations used across
# multiple tests, chosen to avoid theta = 0 (tested separately) and to
# span more than one quadrant of phi.
CONFIGS = [
    (0.10, 0.00),
    (0.30, 0.50),
    (0.60, -1.20),
    (0.95, 2.80),
]


# ========================================================================
# 1. Analytical / geometric consistency
# ========================================================================
class TestBackboneParameterization:

    def test_base_is_at_origin(self, kin):
        """r(0, q) must equal the origin for any (theta, phi)."""
        for theta, phi in CONFIGS:
            r0 = kin.backbone_position(0.0, theta, phi)
            assert np.allclose(r0, [0.0, 0.0, 0.0], atol=1e-12)

    def test_tip_matches_continelli_geometric_formula(self, kin, robot):
        """
        p_tip(q) must reduce to the point-wise geometric solution
        s0 = (L/theta)(1 - cos theta), t0 = (L/theta) sin theta,
        reported in Continelli et al. (2023) for the mobile platform.
        """
        L = robot.geometry.length
        for theta, phi in CONFIGS:
            p = kin.tip_position(theta, phi)
            s0 = (L / theta) * (1 - np.cos(theta))
            t0 = (L / theta) * np.sin(theta)
            expected = np.array([s0 * np.cos(phi), s0 * np.sin(phi), t0])
            assert np.allclose(p, expected, atol=1e-10)

    def test_straight_backbone_limit(self, kin):
        """
        As theta -> 0, r(s,q) must recover the undeformed straight
        configuration [0, 0, s], independently of phi.
        """
        s = 0.07
        for phi in [0.0, 1.3, -2.5]:
            r = kin.backbone_position(s, 0.0, phi)
            assert np.allclose(r, [0.0, 0.0, s], atol=1e-12)

    def test_continuity_across_theta_eps_boundary(self, kin):
        """
        r(s,q) must be continuous when crossing the THETA_EPS threshold
        that switches between the closed-form expression and its
        analytic small-theta limit; no discontinuity should appear.
        """
        s, phi = 0.05, 0.4
        r_below = kin.backbone_position(s, THETA_EPS / 2, phi)
        r_above = kin.backbone_position(s, THETA_EPS * 2, phi)
        assert np.allclose(r_below, r_above, atol=1e-6)


class TestRotationMatrix:

    def test_identity_at_zero_bending(self, kin):
        """R(0, phi) must equal the identity for any phi."""
        for phi in [0.0, 0.7, -1.9, 3.0]:
            R = kin.rotation_matrix(0.0, phi)
            assert np.allclose(R, np.eye(3), atol=1e-12)

    @pytest.mark.parametrize("theta,phi", CONFIGS)
    def test_orthogonality_and_determinant(self, kin, theta, phi):
        """R must be a proper rotation: R R^T = I and det(R) = 1."""
        R = kin.rotation_matrix(theta, phi)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-10)

    def test_third_column_maps_z_to_bending_direction(self, kin):
        """
        R(theta,phi) @ [0,0,1] must point along the direction obtained by
        bending the z-axis by theta within the plane at azimuth phi,
        i.e. (sin(theta)cos(phi), sin(theta)sin(phi), cos(theta)).
        """
        for theta, phi in CONFIGS:
            R = kin.rotation_matrix(theta, phi)
            z_mapped = R @ np.array([0.0, 0.0, 1.0])
            expected = np.array([
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta),
            ])
            assert np.allclose(z_mapped, expected, atol=1e-10)


class TestHomogeneousTransform:

    def test_block_structure(self, kin):
        """T(q) must embed R(q) and p_tip(q) in the expected 4x4 blocks,
        with a bottom row of [0,0,0,1]."""
        for theta, phi in CONFIGS:
            T = kin.homogeneous_transform(theta, phi)
            R = kin.rotation_matrix(theta, phi)
            p = kin.tip_position(theta, phi)

            assert np.allclose(T[0:3, 0:3], R, atol=1e-12)
            assert np.allclose(T[0:3, 3], p, atol=1e-12)
            assert np.allclose(T[3, :], [0.0, 0.0, 0.0, 1.0], atol=1e-12)

    def test_identity_pose_at_theta_zero(self, kin):
        T = kin.homogeneous_transform(0.0, 0.6)
        assert np.allclose(T[0:3, 0:3], np.eye(3), atol=1e-12)


# ========================================================================
# 2. Cross-validation against independent numerical differentiation
# ========================================================================
class TestLinearVelocityJacobian:

    @pytest.mark.parametrize("theta,phi", CONFIGS)
    @pytest.mark.parametrize("s", [0.02, 0.07, 0.105])
    def test_matches_finite_differences(self, kin, theta, phi, s):
        """
        J_v(s) must match a central finite-difference approximation of
        d r(s,q)/dq, independently of the closed-form derivation.
        """
        eps = 1e-6
        Jv = kin.linear_velocity_jacobian(s, theta, phi)

        r_theta_plus = kin.backbone_position(s, theta + eps, phi)
        r_theta_minus = kin.backbone_position(s, theta - eps, phi)
        dr_dtheta = (r_theta_plus - r_theta_minus) / (2 * eps)

        r_phi_plus = kin.backbone_position(s, theta, phi + eps)
        r_phi_minus = kin.backbone_position(s, theta, phi - eps)
        dr_dphi = (r_phi_plus - r_phi_minus) / (2 * eps)

        Jv_numeric = np.column_stack([dr_dtheta, dr_dphi])
        assert np.allclose(Jv, Jv_numeric, atol=1e-6)

    def test_small_theta_limit_matches_formula(self, kin, robot):
        """
        The theta -> 0 branch of J_v(s) must equal the analytic limit
        [[s^2 cos(phi)/(2L), 0], [s^2 sin(phi)/(2L), 0], [0, 0]].
        """
        L = robot.geometry.length
        s, phi = 0.06, 0.9
        Jv = kin.linear_velocity_jacobian(s, 0.0, phi)
        expected = np.array([
            [s**2 * np.cos(phi) / (2 * L), 0.0],
            [s**2 * np.sin(phi) / (2 * L), 0.0],
            [0.0, 0.0],
        ])
        assert np.allclose(Jv, expected, atol=1e-12)

    def test_zero_at_base(self, kin):
        """J_v(0) must be the zero matrix: the base does not move."""
        for theta, phi in CONFIGS:
            Jv = kin.linear_velocity_jacobian(0.0, theta, phi)
            assert np.allclose(Jv, np.zeros((3, 2)), atol=1e-12)


class TestAngularVelocityJacobian:

    @pytest.mark.parametrize("theta,phi", CONFIGS)
    def test_matches_Rdot_RT_identity(self, kin, theta, phi):
        """
        omega = J_omega(q) qdot must match the angular velocity obtained
        from the kinematic identity Rdot R^T = [omega x], computed by
        numerically differentiating R(theta,phi) along a given qdot.
        This directly validates eq. (6) independently of its algebraic
        derivation from R(theta,phi).
        """
        theta_dot, phi_dot = 0.35, -0.6
        eps = 1e-6

        omega_analytic = kin.angular_velocity(theta, phi, theta_dot, phi_dot)

        R0 = kin.rotation_matrix(theta, phi)
        R1 = kin.rotation_matrix(theta + theta_dot * eps, phi + phi_dot * eps)
        Rdot = (R1 - R0) / eps
        omega_skew = Rdot @ R0.T

        omega_numeric = np.array([
            omega_skew[2, 1],
            omega_skew[0, 2],
            omega_skew[1, 0],
        ])
        assert np.allclose(omega_analytic, omega_numeric, atol=1e-5)

    def test_well_defined_at_theta_zero(self, kin):
        """
        Unlike J_v(s), J_omega(theta,phi) requires no small-theta
        safeguard; it must evaluate to a finite matrix at theta = 0.
        """
        for phi in [0.0, 1.0, -2.2]:
            Jw = kin.angular_velocity_jacobian(0.0, phi)
            assert np.all(np.isfinite(Jw))
            expected = np.array([
                [-np.sin(phi), 0.0],
                [np.cos(phi), 0.0],
                [0.0, 0.0],
            ])
            assert np.allclose(Jw, expected, atol=1e-12)

    def test_pure_phi_rotation_cancels_cross_term(self, kin):
        """
        Sanity check tied to the cross-term cancellation discussed in
        Sec. II.C: for theta_dot = 0, the z-component of omega must equal
        phi_dot * (1 - cos(theta)) exactly, with no theta-dependence
        leaking into the x,y components beyond the expected sin/cos form.
        """
        theta, phi, phi_dot = 0.5, 0.2, 0.4
        omega = kin.angular_velocity(theta, phi, 0.0, phi_dot)
        assert np.isclose(omega[2], phi_dot * (1 - np.cos(theta)), atol=1e-12)


# ========================================================================
# 3. Structural / edge-case properties
# ========================================================================
class TestSingularityHandling:

    def test_no_nan_or_inf_near_theta_zero(self, kin):
        """
        Evaluating r(s,q) and J_v(s) at and around theta = 0 must never
        produce NaN or Inf, confirming the THETA_EPS safeguard is active.
        """
        s, phi = 0.05, 1.0
        for theta in [0.0, 1e-9, -1e-9, THETA_EPS / 10]:
            r = kin.backbone_position(s, theta, phi)
            Jv = kin.linear_velocity_jacobian(s, theta, phi)
            assert np.all(np.isfinite(r))
            assert np.all(np.isfinite(Jv))

    def test_negative_theta_is_handled(self, kin):
        """
        The |theta| < THETA_EPS check must also cover small negative
        theta, not only the positive side.
        """
        r = kin.backbone_position(0.05, -1e-9, 0.5)
        assert np.all(np.isfinite(r))
        assert np.allclose(r, [0.0, 0.0, 0.05], atol=1e-8)


# ========================================================================
# 4. Forward/inverse kinematics interface
# ========================================================================
class TestForward:

    def test_forward_matches_tip_position(self, kin):
        """forward() must be consistent with tip_position() (eq. 2 at s=L)."""
        for theta, phi in CONFIGS:
            fwd = np.array(kin.forward(theta, phi))
            tip = kin.tip_position(theta, phi)
            assert np.allclose(fwd, tip, atol=1e-12)


class TestInverseApprox:

    def test_phi_is_exact(self, kin):
        """
        phi = atan2(y,x) carries no approximation error: recomputing phi
        from a tip generated at known (theta,phi) must recover phi
        exactly, regardless of theta.
        """
        for theta_true, phi_true in CONFIGS:
            p = kin.tip_position(theta_true, phi_true)
            _, phi_approx = kin.inverse_approx(p[0], p[1])
            assert np.isclose(phi_approx, phi_true, atol=1e-10)

    @pytest.mark.parametrize("theta_true,max_rel_error", [
        (0.05, 0.005),
        (0.20, 0.01),
        (0.60, 0.05),
        (1.00, 0.10),
        (1.50, 0.20),
    ])
    def test_theta_error_bounds(self, kin, theta_true, max_rel_error):
        """
        Quantifies the relative error of the first-order small-bending
        approximation theta ~= 2*r_xy/L against the exact tip_position
        model, at representative bending angles. These bounds document
        the growing error away from theta = 0 and must not be loosened
        without re-deriving the approximation.
        """
        phi_true = 0.3
        p = kin.tip_position(theta_true, phi_true)
        theta_approx, _ = kin.inverse_approx(p[0], p[1])
        rel_error = abs(theta_approx - theta_true) / theta_true
        assert rel_error < max_rel_error

    def test_error_grows_monotonically_with_theta(self, kin):
        """
        Sanity check on the qualitative behavior of the approximation:
        the relative error must increase as theta grows away from 0,
        confirming it is a small-bending (not global) approximation.
        """
        phi_true = 0.3
        errors = []
        for theta_true in [0.05, 0.2, 0.6, 1.0, 1.5]:
            p = kin.tip_position(theta_true, phi_true)
            theta_approx, _ = kin.inverse_approx(p[0], p[1])
            errors.append(abs(theta_approx - theta_true) / theta_true)
        assert all(e2 > e1 for e1, e2 in zip(errors, errors[1:]))


class TestInverseExact:

    @pytest.mark.parametrize("theta_true,phi_true", CONFIGS)
    def test_recovers_known_configuration(self, kin, theta_true, phi_true):
        """
        Given a tip position generated from a known (theta,phi), the
        Newton solver must recover that same configuration to high
        precision, regardless of the bending magnitude.
        """
        target = kin.tip_position(theta_true, phi_true)
        theta_sol, phi_sol, n_iter, res_norm = kin.inverse_exact(target)

        assert res_norm < 1e-9
        assert np.isclose(theta_sol, theta_true, atol=1e-7)
        assert np.isclose(phi_sol, phi_true, atol=1e-7)

    def test_accurate_beyond_small_bending_regime(self, kin):
        """
        Unlike inverse_approx(), inverse_exact() must remain accurate
        at large bending angles where the first-order approximation
        degrades significantly (theta = 1.5 rad, ~17% error above).
        """
        theta_true, phi_true = 1.5, -0.8
        target = kin.tip_position(theta_true, phi_true)
        theta_sol, phi_sol, n_iter, res_norm = kin.inverse_exact(target)

        assert res_norm < 1e-9
        assert np.isclose(theta_sol, theta_true, atol=1e-8)
        assert np.isclose(phi_sol, phi_true, atol=1e-8)

    def test_converges_in_few_iterations(self, kin):
        """
        Using inverse_approx() as the initial guess (default behavior),
        Newton iteration should converge in well under max_iter steps
        for a well-conditioned target inside the workspace.
        """
        target = kin.tip_position(0.7, -1.0)
        _, _, n_iter, _ = kin.inverse_exact(target)
        assert n_iter < 15

    def test_explicit_initial_guess_is_respected(self, kin):
        """Providing theta0/phi0 explicitly must override inverse_approx()."""
        target = kin.tip_position(0.4, 0.9)
        theta_sol, phi_sol, n_iter, res_norm = kin.inverse_exact(
            target, theta0=0.1, phi0=0.0
        )
        assert res_norm < 1e-9
        assert np.isclose(theta_sol, 0.4, atol=1e-7)
        assert np.isclose(phi_sol, 0.9, atol=1e-7)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))