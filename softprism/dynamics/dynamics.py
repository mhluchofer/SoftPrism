# softprism/dynamics/dynamics.py
from .euler_lagrange import (EulerLagrangeModel)


class Dynamics:

    def __init__(self, robot):

        self.robot = robot

        self.model = EulerLagrangeModel(robot)


    def mass_matrix(self, q):

        return self.model.mass_matrix(q)


    def coriolis(self, q, qdot):

        return self.model.coriolis(q, qdot)


    def gravity(self, q):

        return self.model.gravity(q)


    def elasticity(self, q):

        return self.model.elasticity(q)


    def inverse_dynamics(self,q,qdot,qddot):

        return self.model.inverse_dynamics(
            q,
            qdot,
            qddot
        )


    def forward_dynamics(self,q,qdot,tensions):

        return self.model.forward_dynamics(
            q,
            qdot,
            tensions
        )