"""Satellite Spectrum Emulator - A tool for simulating satellite communication spectrums."""

# Import from new structure
from .objects.enums import CarrierType, ModulationType, CarrierStandard, Band, Polarization, BeamDirection
from .objects.carrier import Carrier
from .objects.transponder import Transponder
from .objects.beam import Beam
from .objects.metadata import IQMetadata, PSDMetadata
from .simulation.psd import generate_psd
from .simulation.iq import generate_iq

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
