# softprism/robot.py
# This file defines the main SoftPrism class, which encapsulates the geometry, material properties, actuation, kinematics, and dynamics of the soft robotic prism.
# The class also computes derived parameters and provides an interface for accessing the robot's dynamics.

from .geometry import Geometry
from .material import Material
from .actuation_config import ActuationConfig
from .actuation import TendonActuation
from .kinematics import Kinematics
from .dynamics import Dynamics

import numpy as np


class SoftPrism:

    def __init__(
        self,
        geometry: Geometry,
        material: Material,
        actuationconfig: ActuationConfig
    ):

        self.geometry = geometry
        self.material = material
        self.actuationconfig = actuationconfig

        # calcular automáticamente
        self._compute_derived_parameters()

        self.kinematics = Kinematics(self)
        self.dynamics = Dynamics(self)
        self.actuation = TendonActuation(self)


    def _compute_derived_parameters(self):

        self.gravity = 9.81

        self.segment_length = (
            self.geometry.length /
            self.geometry.n_prisms
        )

        self.total_mass = (
            self.material.backbone_mass +
            self.geometry.n_prisms *
            self.material.prism_mass
        )

        self.I_geom = (
            np.pi *
            self.geometry.backbone_radius**4
            / 4
        )



    def _dynamic_parameters(self):

        return {

            "L":self.geometry.length,
            "a":self.geometry.prism_side,
            "h":self.geometry.prism_thickness,
            "r_hole":self.geometry.backbone_radius,
            "m_backbone":self.material.backbone_mass,
            "m_prism":self.material.prism_mass,
            "E_mod":self.material.young_modulus,
            "R_c": self.actuationconfig.tendon_distance,
            "I_geom":self.I_geom,
            "g":self.gravity
        }