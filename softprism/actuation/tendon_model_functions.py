import numpy as np
import math

def J_L_func_api(theta,phi,R_c):
    return np.array([
        [-R_c*math.sin(phi), -R_c*theta*math.cos(phi)],
        [R_c*math.sin(phi + (1/3)*math.pi), R_c*theta*math.cos(phi + (1/3)*math.pi)],
        [-R_c*math.cos(phi + (1/6)*math.pi), R_c*theta*math.sin(phi + (1/6)*math.pi)]
    ])


def tendon_lengths_api(theta,phi,L,R_c):
    return np.array([
        L - R_c*theta*math.sin(phi),
        L + R_c*theta*math.sin(phi + (1/3)*math.pi),
        L - R_c*theta*math.cos(phi + (1/6)*math.pi)
    ])
