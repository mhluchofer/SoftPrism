import numpy as np
from scipy.optimize import lsq_linear
from .tendon_model_functions import J_L_func_api, tendon_lengths_api

class TendonActuation:

    def __init__(self, robot):

        self.robot = robot

    def tendon_jacobian(self,q):

        theta, phi = q
        p = self.robot._dynamic_parameters()

        return J_L_func_api(theta,phi,p["R_c"])
    
    def tensions_to_generalized(self,q,tensions):

        J = self.tendon_jacobian(q)
        T = np.asarray(tensions).reshape(3,1)

        return J.T @ T
    
    def solve_tensions(self,q,Q):

        J = self.tendon_jacobian(q)

        Q = np.asarray(Q).flatten()
        if Q.size != 2:
            raise ValueError("Q must have length 2")
        
        result = lsq_linear(J.T,Q,bounds=(0,np.inf))
        if not result.success:
            raise RuntimeError("Tension optimization failed")

        return result.x
    
    def tendon_lengths(self,q):

        theta, phi = q
        p = self.robot._dynamic_parameters()

        return tendon_lengths_api(theta,phi,p["L"],p["R_c"])