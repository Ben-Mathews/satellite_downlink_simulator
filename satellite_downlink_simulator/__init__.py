"""Satellite Spectrum Emulator - A tool for simulating satellite communication spectrums."""

from .enums import CarrierType, ModulationType, CarrierStandard, Band, Polarization, BeamDirection
from .carrier import Carrier
from .transponder import Transponder
from .beam import Beam
from .metadata import IQMetadata, PSDMetadata
from .generation import generate_psd, generate_iq

__version__ = "0.1.0"

__all__ = [
    "CarrierType",
    "ModulationType",
    "CarrierStandard",
    "Band",
    "Polarization",
    "BeamDirection",
    "Carrier",
    "Transponder",
    "Beam",
    "IQMetadata",
    "PSDMetadata",
    "generate_psd",
    "generate_iq",
]
