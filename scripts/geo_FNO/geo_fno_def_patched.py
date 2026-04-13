"""
Patched Geo-FNO import surface.

This module exposes the current local Geo-FNO implementation under the
`geo_fno_def_patched` name so training scripts can depend on a stable
patched import path without pulling in the older vanilla module name.
"""

from geo_FNO_def import FNO2d, IPHI, SpectralConv2d, get_global_L_from_h5, set_seed

__all__ = [
    "FNO2d",
    "IPHI",
    "SpectralConv2d",
    "get_global_L_from_h5",
    "set_seed",
]
