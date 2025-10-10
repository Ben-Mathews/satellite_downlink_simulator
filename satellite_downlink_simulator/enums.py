"""Enumerations for satellite spectrum emulator."""

from enum import Enum


class CarrierType(Enum):
    """Type of carrier access method."""
    FDMA = "FDMA"
    TDMA = "TDMA"


class ModulationType(Enum):
    """Modulation type for carriers."""
    BPSK = "BPSK"
    QPSK = "QPSK"
    QAM16 = "16QAM"
    APSK16 = "16APSK"
    APSK32 = "32APSK"


class CarrierStandard(Enum):
    """Communication standard for carriers."""
    NONE = "NONE"
    DVB_S = "DVB-S"
    DVB_S2 = "DVB-S2"
    IESS_308 = "IESS-308"


class Band(Enum):
    """Satellite communication frequency bands."""
    L = "L"
    C = "C"
    X = "X"
    KA = "KA"


class Polarization(Enum):
    """Polarization of satellite beam."""
    LHCP = "LHCP"
    RHCP = "RHCP"
    HORIZONTAL = "HORIZONTAL"
    VERTICAL = "VERTICAL"


class BeamDirection(Enum):
    """Direction of satellite beam."""
    DOWNLINK = "DOWNLINK"
    UPLINK = "UPLINK"
