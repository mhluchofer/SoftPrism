"""
test_robot.py

Minimal end-to-end demonstration of the SoftPrism pipeline:
kinematics -> dynamics -> actuation, using the physical parameters of
Table 1 in the accompanying paper.
"""

from softprism import Geometry, Material, ActuationConfig, SoftPrism


geometry = Geometry(
    length=0.105,           # backbone length (m)
    n_prisms=5,             # number of prisms
    prism_side=0.07,        # prism side length (m)
    prism_thickness=0.014,  # prism thickness (m)
    backbone_radius=0.01    # backbone radius (m)
)

material = Material(
    backbone_mass=0.03,   # backbone mass (kg)
    prism_mass=0.01,       # prism mass (kg)
    young_modulus=2e6      # Young's modulus (Pa)
)

actuationconfig = ActuationConfig(
    n_tendons=3,           # number of tendons
    tendon_distance=0.005  # tendon radial distance R_c (m)
)

robot = SoftPrism(
    geometry,
    material,
    actuationconfig
)

# ----------------------------------
# Kinematics
# ----------------------------------
# The exact inverse kinematics requires a target tip position that is
# actually reachable, i.e. a point on the image of tip_position(theta,phi).
# Rather than proposing an arbitrary (x, y) pair (which need not lie on
# that surface), the target here is generated from a known configuration
# via the forward map, then recovered through inverse_exact() to
# demonstrate the round trip.

theta_ref, phi_ref = 0.6, 0.4
target = robot.kinematics.tip_position(theta_ref, phi_ref)

theta, phi, n_iter, residual_norm = robot.kinematics.inverse_exact(target)

q = [theta, phi]

print("Reference (theta, phi) =", (theta_ref, phi_ref))
print("Recovered (theta, phi) =", (theta, phi))
print("Newton iterations =", n_iter, " | residual norm =", residual_norm)

# ----------------------------------
# Dynamics
# ----------------------------------

qdot = [0.5, 0.0]
qddot = [0.0, 0.0]

Q = robot.dynamics.inverse_dynamics(
    q,
    qdot,
    qddot
)

# ----------------------------------
# Actuation
# ----------------------------------

T = robot.actuation.solve_tensions(
    q,
    Q
)

lengths = robot.actuation.tendon_lengths(
    q
)

# ----------------------------------
# Results
# ----------------------------------

print("q =", q)

print("Q =")
print(Q)

print("Tensions =")
print(T)

print("Lengths =")
print(lengths)