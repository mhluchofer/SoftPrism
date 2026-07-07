from dataclasses import dataclass

@dataclass
class Geometry:
    length: float
    n_prisms: int
    prism_side: float
    prism_thickness: float
    backbone_radius: float