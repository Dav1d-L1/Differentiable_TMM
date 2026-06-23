"""Convenience re-exports for the differentiable_tmm public API."""

from . import DE94, Spectrum2XYZ, TMM, XYZ2Lab, XYZ2sRGB, spectrum_interpolation

__all__ = [
    "TMM",
    "spectrum_interpolation",
    "Spectrum2XYZ",
    "XYZ2Lab",
    "XYZ2sRGB",
    "DE94",
]
