"""
3D vector class and angle, dihedral and distance helpers. The implementation
lives in the ``turtlend`` package, https://github.com/Flux-Frontiers/turtlend.
This module re-exports it so that ``waverider.vector3D`` keeps resolving.

Author: Eric G. Suchanek, PhD
"""

from turtlend.vector3D import (
    Vector3D,
    calc_angle,
    calc_dihedral,
    calculate_bond_angle,
    distance3d,
    rms_difference,
)

__all__ = [
    "Vector3D",
    "calc_angle",
    "calc_dihedral",
    "calculate_bond_angle",
    "distance3d",
    "rms_difference",
]
